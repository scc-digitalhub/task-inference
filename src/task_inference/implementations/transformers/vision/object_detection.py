# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - object detection."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.vision.object_detection import (
    BoundingBox,
    DetectedObject,
    ObjectDetectionInput,
    ObjectDetectionOutput,
    ObjectDetectionTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "facebook/detr-resnet-50"


class TransformersObjectDetectionTask(TransformersTaskMixin, ObjectDetectionTask):
    """Object detection using a HuggingFace ``object-detection`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``object-detection`` compatible model on the Hub.
        Defaults to ``facebook/detr-resnet-50``.
    device:
        Inference device.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline("object-detection", model=model_name, device=device)

    def process(self, inputs: ObjectDetectionInput) -> ObjectDetectionOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        batch_results = self._pipe(pil_images, threshold=inputs.threshold)
        # pipeline returns list[list[dict]] for a batch of images
        all_detections = []
        for results in batch_results:
            dets = []
            for r in results:
                box = r.get("box", {})
                dets.append(
                    DetectedObject(
                        label=r.get("label", ""),
                        score=float(r.get("score", 0.0)),
                        box=BoundingBox(
                            xmin=float(box.get("xmin", 0)),
                            ymin=float(box.get("ymin", 0)),
                            xmax=float(box.get("xmax", 0)),
                            ymax=float(box.get("ymax", 0)),
                        ),
                    )
                )
            all_detections.append(dets)
        return ObjectDetectionOutput(detections=all_detections)
