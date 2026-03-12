# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - Visual Question Answering."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.vision.visual_question_answering import VQAAnswer, VQAInput, VQAOutput, VQATask
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "dandelin/vilt-b32-finetuned-vqa"


class TransformersVQATask(TransformersTaskMixin, VQATask):
    """Visual Question Answering using a HuggingFace ``visual-question-answering`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``visual-question-answering`` compatible model on the Hub.
        Defaults to ``dandelin/vilt-b32-finetuned-vqa``.
    device:
        Inference device.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline("visual-question-answering", model=model_name, device=device)

    def process(self, inputs: VQAInput) -> VQAOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        # VQA pipeline is called per-image with the shared question
        answers_per_image = []
        for pil_image in pil_images:
            results = self._pipe(pil_image, inputs.question, top_k=inputs.top_k)
            answers_per_image.append(
                [VQAAnswer(answer=r["answer"], score=float(r["score"])) for r in results]
            )
        return VQAOutput(answers=answers_per_image)
