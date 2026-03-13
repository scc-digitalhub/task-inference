# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - object detection (pure ORT, no transformers)."""
from __future__ import annotations

from ....tasks.vision.object_detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
    ObjectDetectionTask,
)
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.detection import resolve_detection_adapter


class OnnxObjectDetectionTask(OnnxRuntimeTaskMixin, ObjectDetectionTask):
    """Object detection using a local ONNX model.

    Dialect is auto-detected from the model's ONNX input/output tensor names.
    Supported dialects:

    * **transformers-detr** — ``images`` → ``logits [B,Q,C+1]`` + ``pred_boxes [B,Q,4]``
      (DETR, Conditional-DETR, DAB-DETR, …)
    * **transformers-yolos** — ``pixel_values`` → same output contract as DETR
      (YOLOS, hustvl/yolos-tiny, …)
    * **torchvision-detection** — any input → ``pred_boxes``/``boxes`` + ``labels`` +
      ``scores``/``logits`` in pixel ``(xmin,ymin,xmax,ymax)`` format
      (FasterRCNN, SSD, RetinaNet exported via ``torch.onnx.export``)
    * **yolov8** — ``images`` → single output tensor ``[B,4+C,N]``
      (Ultralytics YOLOv8/v9/v10, letterbox preprocessing + NMS applied)

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device.

    Example
    -------
    ::

        optimum-cli export onnx --model facebook/detr-resnet-50 ./onnx/detr/
        task = OnnxObjectDetectionTask(model_name="./onnx/detr/")
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
        self._adapter = resolve_detection_adapter(session, pp_cfg, cfg)

    def process(self, inputs: ObjectDetectionInput) -> ObjectDetectionOutput:
        all_detections = []
        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            dets = self._adapter.detect(pil_image, inputs.threshold, self._id2label)
            all_detections.append(dets)
        return ObjectDetectionOutput(detections=all_detections)
