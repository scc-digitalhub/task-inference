# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime object-detection dialect adapters.

Supported dialects
------------------
transformers-detr
    DETR (DEtection TRansformer) and compatible encoder-decoder detectors.
    Input tensor named ``images``; outputs ``logits [B, Q, C+1]`` (last
    column = background) and ``pred_boxes [B, Q, 4]`` in normalised
    ``(cx, cy, w, h)`` format.  Applicable to: facebook/detr-*,
    microsoft/conditional-detr-*, etc.

transformers-yolos
    YOLOS (You Only Look at One Sequence) and any other ViT-based detector
    that follows the same output contract as DETR but names the image input
    ``pixel_values``.  Applicable to: hustvl/yolos-*.

torchvision-detection
    TorchVision FasterRCNN / SSD exported with explicit tensor names.
    Outputs ``pred_boxes`` (or ``boxes``) in pixel-space ``(xmin, ymin,
    xmax, ymax)`` format, ``labels`` (integer class ids), and ``scores``
    (or ``logits``) as per-detection confidence values.  No background
    class in the label vocabulary.

yolov8
    Ultralytics YOLOv8 (and compatible YOLOv5/v9/v10 exports).
    Input ``images``; single output tensor ``[B, 4+C, N]`` or ``[B, N,
    4+C]`` (auto-detected from shape) with unnormalised ``(cx, cy, w, h)``
    box coordinates in *letterbox image space*.  Includes per-class score
    columns; max-score + argmax labelling + greedy NMS are applied.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from ..base import OnnxDialectAdapter, resolve_adapter, io_names_from_session
from ...base import OnnxRuntimeTaskMixin as _M
from .....tasks.vision.object_detection import BoundingBox, DetectedObject


# ---------------------------------------------------------------------------
# Per-task abstract base
# ---------------------------------------------------------------------------

class ObjectDetectionAdapter(OnnxDialectAdapter):
    """Abstract adapter for object-detection tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        self._session = session
        self._pp_cfg = pp_cfg

    @abstractmethod
    def detect(
        self,
        pil_image: Image.Image,
        threshold: float,
        id2label: dict[int, str],
    ) -> list[DetectedObject]:
        """Run inference and return detections above *threshold*."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cxcywh_to_xyxy(
    boxes: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    """Convert normalised ``(cx, cy, w, h)`` to pixel ``(xmin, ymin, xmax, ymax)``."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack(
        [(cx - w / 2) * img_w,
         (cy - h / 2) * img_h,
         (cx + w / 2) * img_w,
         (cy + h / 2) * img_h],
        axis=-1,
    )


def _build_detections(
    boxes_xyxy: np.ndarray,
    class_ids: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    id2label: dict[int, str],
    orig_w: int,
    orig_h: int,
) -> list[DetectedObject]:
    dets = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < threshold:
            continue
        b = boxes_xyxy[i]
        dets.append(
            DetectedObject(
                label=id2label.get(int(class_ids[i]), str(int(class_ids[i]))),
                score=score,
                box=BoundingBox(
                    xmin=float(np.clip(b[0], 0, orig_w)),
                    ymin=float(np.clip(b[1], 0, orig_h)),
                    xmax=float(np.clip(b[2], 0, orig_w)),
                    ymax=float(np.clip(b[3], 0, orig_h)),
                ),
            )
        )
    return dets


# ---------------------------------------------------------------------------
# Dialect: transformers-detr  (and transformers-yolos via subclass)
# ---------------------------------------------------------------------------

class DetrAdapter(ObjectDetectionAdapter):
    """DETR-family dialect adapter.

    Input key  : ``images``
    Output keys: ``logits [B, Q, C+1]``, ``pred_boxes [B, Q, 4]``
    Box format : normalised ``(cx, cy, w, h)``; last logit column = background.
    """

    DIALECT: ClassVar[str] = "transformers-detr"
    _INPUT_KEY: ClassVar[str] = "images"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return (
            cls._INPUT_KEY in input_names
            and "logits" in output_names
            and "pred_boxes" in output_names
        )

    def detect(
        self,
        pil_image: Image.Image,
        threshold: float,
        id2label: dict[int, str],
    ) -> list[DetectedObject]:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        orig_w, orig_h = pil_image.size

        out = self._session.run(
            ["logits", "pred_boxes"], {self._INPUT_KEY: pixel_values}
        )
        logits = out[0][0]      # [Q, C+1]
        pred_boxes = out[1][0]  # [Q, 4]

        probs = _M._softmax(logits)                          # [Q, C+1]
        class_ids = np.argmax(probs[:, :-1], axis=-1)        # [Q]   (exclude background)
        scores = probs[np.arange(len(probs)), class_ids]     # [Q]

        boxes_xyxy = _cxcywh_to_xyxy(pred_boxes, orig_w, orig_h)
        return _build_detections(boxes_xyxy, class_ids, scores, threshold, id2label, orig_w, orig_h)


class YolosAdapter(DetrAdapter):
    """YOLOS-family dialect adapter.

    Identical output contract to DETR; only the image input is named
    ``pixel_values`` instead of ``images``.
    """

    DIALECT: ClassVar[str] = "transformers-yolos"
    _INPUT_KEY: ClassVar[str] = "pixel_values"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return (
            "pixel_values" in input_names
            and "logits" in output_names
            and "pred_boxes" in output_names
        )


# ---------------------------------------------------------------------------
# Dialect: torchvision-detection
# ---------------------------------------------------------------------------

class TorchVisionDetectionAdapter(ObjectDetectionAdapter):
    """TorchVision FasterRCNN / SSD dialect adapter.

    Outputs: ``pred_boxes`` (or ``boxes``) in pixel ``(xmin, ymin, xmax,
    ymax)`` format, ``labels`` (int class ids, 1-indexed), ``scores`` (or
    ``logits``) as float confidence values.

    No background class in label vocab; ``id2label`` should map 1-based
    indices to class names (COCO convention).
    """

    DIALECT: ClassVar[str] = "torchvision-detection"

    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        super().__init__(session, pp_cfg)
        out_names = [o.name for o in session.get_outputs()]
        self._box_key = "pred_boxes" if "pred_boxes" in out_names else "boxes"
        self._score_key = "scores" if "scores" in out_names else "logits"
        inp_names = [i.name for i in session.get_inputs()]
        self._input_key = "pixel_values" if "pixel_values" in inp_names else "images"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        has_boxes = "pred_boxes" in output_names or "boxes" in output_names
        return has_boxes and "labels" in output_names

    def detect(
        self,
        pil_image: Image.Image,
        threshold: float,
        id2label: dict[int, str],
    ) -> list[DetectedObject]:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        orig_w, orig_h = pil_image.size

        out = self._session.run(
            [self._box_key, "labels", self._score_key],
            {self._input_key: pixel_values},
        )
        boxes = out[0]   # [N, 4]  xyxy pixel coords (no batch dim)
        labels = out[1]  # [N]
        scores = out[2]  # [N]

        dets = []
        for i in range(len(scores)):
            score = float(scores[i])
            if score < threshold:
                continue
            b = boxes[i]
            label_id = int(labels[i])
            dets.append(
                DetectedObject(
                    label=id2label.get(label_id, str(label_id)),
                    score=score,
                    box=BoundingBox(
                        xmin=float(np.clip(b[0], 0, orig_w)),
                        ymin=float(np.clip(b[1], 0, orig_h)),
                        xmax=float(np.clip(b[2], 0, orig_w)),
                        ymax=float(np.clip(b[3], 0, orig_h)),
                    ),
                )
            )
        return dets


# ---------------------------------------------------------------------------
# Dialect: yolov8
# ---------------------------------------------------------------------------

def _letterbox(
    pil_image: Image.Image,
    target_h: int,
    target_w: int,
) -> tuple[np.ndarray, float, int, int]:
    """Resize with grey padding to maintain aspect ratio (YOLOv8 convention).

    Returns
    -------
    padded : np.ndarray
        uint8 HWC array of shape ``(target_h, target_w, 3)``.
    scale : float
        Scaling factor applied to original dimensions.
    pad_top, pad_left : int
        Pixel offsets of the original image within the padded canvas.
    """
    orig_w, orig_h = pil_image.size
    scale = min(target_h / orig_h, target_w / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = np.array(pil_image.resize((new_w, new_h), Image.BILINEAR), dtype=np.uint8)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return padded, scale, pad_top, pad_left


def _nms(
    boxes_cxcywh: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    """Greedy NMS operating on ``(cx, cy, w, h)`` boxes (pixel space)."""
    if len(boxes_cxcywh) == 0:
        return []
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou < iou_threshold]
    return keep


class YoloV8Adapter(ObjectDetectionAdapter):
    """Ultralytics YOLOv8 dialect adapter.

    Input  : ``images [B, 3, H, W]`` (normalised to ``[0, 1]``,
             letterbox-padded to the model's native resolution).
    Output : single tensor ``output0 [B, 4+C, N]`` or ``[B, N, 4+C]``
             (layout auto-detected from ONNX metadata at init time).

    Box coords ``(cx, cy, w, h)`` are in letterbox-image pixel space and
    are un-letterboxed to the original image coordinates before returning.
    Per-class scores are raw; argmax labelling + greedy NMS at
    ``iou_threshold=0.45`` are applied.
    """

    DIALECT: ClassVar[str] = "yolov8"

    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        super().__init__(session, pp_cfg)
        # Determine model input resolution
        in_shape = session.get_inputs()[0].shape  # [1, 3, H, W] or dynamic
        self._img_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 640
        self._img_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 640
        # Determine output layout: [B, 4+C, N] vs [B, N, 4+C]
        out_shape = session.get_outputs()[0].shape
        s1 = out_shape[1]
        s2 = out_shape[2]
        if isinstance(s1, int) and isinstance(s2, int):
            self._transposed = s1 > s2  # True → [B, N, 4+C]
        else:
            self._transposed = False    # default [B, 4+C, N]
        self._out_name = session.get_outputs()[0].name

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        # Catch-all for YOLO-like single-output models with 'images' input
        return (
            "images" in input_names
            and len(output_names) == 1
            and "pred_boxes" not in output_names
            and "logits" not in output_names
        )

    def detect(
        self,
        pil_image: Image.Image,
        threshold: float,
        id2label: dict[int, str],
    ) -> list[DetectedObject]:
        orig_w, orig_h = pil_image.size
        padded, scale, pad_top, pad_left = _letterbox(pil_image, self._img_h, self._img_w)

        # Normalize to [0, 1] and convert to CHW float32
        arr = padded.astype(np.float32) / 255.0
        pixel_values = arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]

        raw = self._session.run([self._out_name], {"images": pixel_values})[0]  # [1, *, *]

        if self._transposed:
            preds = raw[0]           # [N, 4+C]
        else:
            preds = raw[0].T         # transpose [4+C, N] → [N, 4+C]

        boxes_cxcywh = preds[:, :4]       # [N, 4]  cx,cy,w,h in letterbox space
        class_scores = preds[:, 4:]        # [N, C]

        class_ids = np.argmax(class_scores, axis=-1)   # [N]
        scores = class_scores[np.arange(len(preds)), class_ids]  # [N]

        # Confidence filter
        mask = scores >= threshold
        if not mask.any():
            return []
        boxes_cxcywh = boxes_cxcywh[mask]
        class_ids = class_ids[mask]
        scores = scores[mask]

        # NMS
        keep = _nms(boxes_cxcywh, scores)
        boxes_cxcywh = boxes_cxcywh[keep]
        class_ids = class_ids[keep]
        scores = scores[keep]

        # Un-letterbox: remove padding offset, rescale to original image
        boxes_cxcywh[:, 0] = (boxes_cxcywh[:, 0] - pad_left) / scale
        boxes_cxcywh[:, 1] = (boxes_cxcywh[:, 1] - pad_top) / scale
        boxes_cxcywh[:, 2] /= scale
        boxes_cxcywh[:, 3] /= scale

        dets = []
        for i in range(len(scores)):
            cx, cy, w, h = boxes_cxcywh[i]
            dets.append(
                DetectedObject(
                    label=id2label.get(int(class_ids[i]), str(int(class_ids[i]))),
                    score=float(scores[i]),
                    box=BoundingBox(
                        xmin=float(np.clip(cx - w / 2, 0, orig_w)),
                        ymin=float(np.clip(cy - h / 2, 0, orig_h)),
                        xmax=float(np.clip(cx + w / 2, 0, orig_w)),
                        ymax=float(np.clip(cy + h / 2, 0, orig_h)),
                    ),
                )
            )
        return dets


# ---------------------------------------------------------------------------
# Registry & factory  (ordering: specific → generic)
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[ObjectDetectionAdapter]] = [
    DetrAdapter,             # images + logits + pred_boxes
    YolosAdapter,            # pixel_values + logits + pred_boxes
    TorchVisionDetectionAdapter,  # any input + labels + boxes/scores
    YoloV8Adapter,           # images + single output (catch-all)
]


def resolve_detection_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
) -> ObjectDetectionAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg)
