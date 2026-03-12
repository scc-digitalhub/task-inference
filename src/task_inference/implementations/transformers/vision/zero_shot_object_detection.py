# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - zero-shot object detection."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.vision.object_detection import BoundingBox, DetectedObject
from ....tasks.vision.zero_shot_object_detection import (
    ZeroShotObjectDetectionInput,
    ZeroShotObjectDetectionOutput,
    ZeroShotObjectDetectionTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "google/owlvit-base-patch32"


class TransformersZeroShotObjectDetectionTask(TransformersTaskMixin, ZeroShotObjectDetectionTask):
    """Zero-shot object detection using a HuggingFace ``zero-shot-object-detection`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``zero-shot-object-detection`` compatible model on the Hub.
        Defaults to ``google/owlvit-base-patch32``.
    device:
        Inference device.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline("zero-shot-object-detection", model=model_name, device=device)

    def process(self, inputs: ZeroShotObjectDetectionInput) -> ZeroShotObjectDetectionOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        all_detections = []
        for pil_image in pil_images:
            results = self._pipe(
                pil_image,
                candidate_labels=inputs.candidate_labels,
                threshold=inputs.threshold,
            )
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
        return ZeroShotObjectDetectionOutput(detections=all_detections)
