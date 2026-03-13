# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - image segmentation (pure ORT, no transformers).

Dialect is auto-detected from the model's ONNX tensor names.

Supported dialects:

* **semantic-logits** — ``pixel_values`` → ``logits [B, C, H', W']``
  Covers SegFormer (stride-4), DeepLabV3 / DeepLabV3+ (full resolution),
  and MobileViT-seg.

Panoptic and instance sub-tasks are not yet supported; use the
``transformers`` backend for those.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ....tasks.vision.image_segmentation import (
    ImageSegmentationInput,
    ImageSegmentationOutput,
    ImageSegmentationTask,
    SegmentResult,
)
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.segmentation import resolve_segmentation_adapter


class OnnxImageSegmentationTask(OnnxRuntimeTaskMixin, ImageSegmentationTask):
    """Image segmentation using a local ONNX model.

    Dialect is auto-detected from the model's ONNX input/output tensor names.
    Supported dialects:

    * **semantic-logits** — ``pixel_values`` → ``logits [B, C, H', W']``
      (SegFormer stride-4, DeepLabV3 full-resolution, MobileViT-seg, …)

    The model directory must contain:

    * ``model.onnx``
    * ``config.json`` — includes ``id2label``
    * ``preprocessor_config.json``

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device.

    Example
    -------
    ::

        optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 ./onnx/segformer/
        task = OnnxImageSegmentationTask(model_name="./onnx/segformer/")
    """

    def __init__(self, model_name: str, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        cfg = self._load_config(model_name)
        self._id2label: dict[int, str] = {
            int(k): v for k, v in cfg.get("id2label", {}).items()
        }
        pp_cfg = self._load_preprocessor_config(model_name)
        providers = self._device_to_providers(device)
        session = self._create_session(self._find_onnx_file(model_name), providers)
        self._adapter = resolve_segmentation_adapter(session, pp_cfg, cfg)

    @staticmethod
    def _mask_to_png(mask: np.ndarray) -> bytes:
        buf = io.BytesIO()
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(buf, format="PNG")
        return buf.getvalue()

    def process(self, inputs: ImageSegmentationInput) -> ImageSegmentationOutput:
        all_segments = []
        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            logits_up = self._adapter.segment(pil_image)  # [C, orig_H, orig_W]

            probs = self._softmax(logits_up.transpose(1, 2, 0))  # [H, W, C]
            seg_map = np.argmax(probs, axis=-1)                   # [H, W]

            segs = []
            for class_id in np.unique(seg_map):
                mask = (seg_map == class_id).astype(np.uint8)
                if mask.sum() == 0:
                    continue
                score = float(probs[:, :, class_id][mask == 1].mean())
                if score < inputs.mask_threshold:
                    continue
                segs.append(
                    SegmentResult(
                        label=self._id2label.get(int(class_id), str(int(class_id))),
                        score=score,
                        mask=self._mask_to_png(mask),
                    )
                )
            all_segments.append(segs)
        return ImageSegmentationOutput(segments=all_segments)
