# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Visual Question Answering task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"
_IN_QUESTION = "question"
_OUT_ANSWERS = "answer"
_OUT_SCORES = "score"

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class VQAInput(BaseModel):
    """Inputs for a Visual Question Answering request."""

    images: list[bytes] = Field(..., description="Raw image bytes")
    question: str = Field(..., description="Natural language question about the images")
    top_k: int = Field(1, ge=1, description="Number of candidate answers to return")

    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name=_IN_IMAGE,
                    shape=[len(self.images)],
                    datatype=Datatype.BYTES,
                    data=[encode_image(img) for img in self.images],
                ),
                RequestInput(
                    name=_IN_QUESTION,
                    shape=[1],
                    datatype=Datatype.BYTES,
                    data=[self.question],
                ),
            ],
            parameters={"top_k": self.top_k},
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "VQAInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        images = [decode_image(b) for b in (next(t for t in request.inputs if t.name == _IN_IMAGE).data or [])]
        question = (next(t for t in request.inputs if t.name == _IN_QUESTION).data or [""])[0]
        top_k = (request.parameters or {}).get("top_k", 1)
        return cls(images=images, question=question, top_k=top_k)


class VQAAnswer(BaseModel):
    """A single candidate answer."""

    answer: str
    score: float


class VQAOutput(BaseModel):
    """Outputs of a VQA request.

    ``answers`` is a list of per-image answer lists.
    """

    answers: list[list[VQAAnswer]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        answers_per = [[a.answer for a in img_answers] for img_answers in self.answers]
        scores_per = [[a.score for a in img_answers] for img_answers in self.answers]
        k = len(self.answers)
        m = min(len(img_answers) for img_answers in self.answers) if self.answers else 0
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_ANSWERS, shape=[k, m], datatype=Datatype.BYTES, data=answers_per),
                ResponseOutput(name=_OUT_SCORES, shape=[k, m], datatype=Datatype.FP32, data=scores_per),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "VQAOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        answers_per = response.get_output(_OUT_ANSWERS).data or []
        scores_per = response.get_output(_OUT_SCORES).data or []
        return cls(
            answers=[
                [VQAAnswer(answer=a, score=sc) for a, sc in zip(img_answers, img_scores, strict=False)]
                for img_answers, img_scores in zip(answers_per, scores_per, strict=False)
            ]
        )


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class VQATask(BaseTask[VQAInput, VQAOutput]):
    """Task definition for Visual Question Answering.

    OIP v2 request tensors
    ----------------------
    * ``image``     -- ``BYTES  [-1]``     images (batch)
    * ``question``  -- ``BYTES [1]``      the question string (same for all images)
    * parameter ``top_k`` -- ``int``

    OIP v2 response tensors
    -----------------------
    * ``answers``  -- ``BYTES [-1]``   per-image list of candidate answer strings
    * ``scores``   -- ``FP32   [-1]``   per-image list of answer confidence scores
    """

    TASK_NAME = "visual-question-answering"
    INPUT_SCHEMA = VQAInput
    OUTPUT_SCHEMA = VQAOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
        MetadataTensor(name="question", datatype=Datatype.BYTES, shape=[1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="answer", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="top_k", datatype="int64", required=False, default=1, description="Number of top candidate answers to return per image"),
    ]
