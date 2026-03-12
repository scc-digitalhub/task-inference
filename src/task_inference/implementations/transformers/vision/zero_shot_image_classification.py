# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - zero-shot image classification."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.vision.zero_shot_image_classification import (
    ZeroShotClassificationResult,
    ZeroShotImageClassificationInput,
    ZeroShotImageClassificationOutput,
    ZeroShotImageClassificationTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class TransformersZeroShotImageClassificationTask(TransformersTaskMixin, ZeroShotImageClassificationTask):
    """Zero-shot image classification using a HuggingFace ``zero-shot-image-classification`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``zero-shot-image-classification`` compatible model on the Hub.
        Defaults to ``openai/clip-vit-base-patch32``.
    device:
        Inference device.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline("zero-shot-image-classification", model=model_name, device=device)

    def process(self, inputs: ZeroShotImageClassificationInput) -> ZeroShotImageClassificationOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        results_per_image = []
        for pil_image in pil_images:
            per_image = self._pipe(pil_image, candidate_labels=inputs.candidate_labels)
            results_per_image.append(
                [
                    ZeroShotClassificationResult(label=r["label"], score=float(r["score"]))
                    for r in per_image
                ]
            )
        return ZeroShotImageClassificationOutput(results=results_per_image)
