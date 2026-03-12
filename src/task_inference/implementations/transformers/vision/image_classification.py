# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - image classification."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.vision.image_classification import (
    ClassificationResult,
    ImageClassificationInput,
    ImageClassificationOutput,
    ImageClassificationTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "google/vit-base-patch16-224"


class TransformersImageClassificationTask(TransformersTaskMixin, ImageClassificationTask):
    """Image classification using a HuggingFace ``image-classification`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``image-classification`` compatible model on the Hub.
        Defaults to ``google/vit-base-patch16-224``.
    device:
        Inference device.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline("image-classification", model=model_name, device=device)

    def process(self, inputs: ImageClassificationInput) -> ImageClassificationOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        batch_results = self._pipe(pil_images, top_k=inputs.top_k)
        # pipeline returns list[list[dict]] for a batch of images
        return ImageClassificationOutput(
            results=[
                [ClassificationResult(label=r["label"], score=float(r["score"])) for r in per_image]
                for per_image in batch_results
            ]
        )
