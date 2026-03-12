# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Zero-shot image classification task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"                    # BYTES [-1]      images (batch)
_IN_CANDIDATE_LABELS = "candidate_labels"  # BYTES [-1]  candidate class label strings
_OUT_LABELS = "label"                  # BYTES [-1, -1]  per-image list of matched labels
_OUT_SCORES = "score"                  # FP32  [-1, -1]  per-image list of match scores

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class ZeroShotClassificationResult(BaseModel):
    """A single label-score pair from zero-shot classification."""

    label: str
    score: float


class ZeroShotImageClassificationInput(BaseModel):
    """Inputs for a zero-shot image classification request."""

    images: list[bytes] = Field(..., description="Raw image bytes (PNG / JPEG / ...)")
    candidate_labels: list[str] = Field(..., min_length=1, description="Candidate class labels to match against")

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
                    name=_IN_CANDIDATE_LABELS,
                    shape=[len(self.candidate_labels)],
                    datatype=Datatype.BYTES,
                    data=self.candidate_labels,
                ),
            ],
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "ZeroShotImageClassificationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        images = [decode_image(b) for b in (next(t for t in request.inputs if t.name == _IN_IMAGE).data or [])]
        label_inputs = [t for t in request.inputs if t.name == _IN_CANDIDATE_LABELS]
        candidate_labels = list(label_inputs[0].data or []) if label_inputs else []
        return cls(images=images, candidate_labels=candidate_labels)


class ZeroShotImageClassificationOutput(BaseModel):
    """Outputs of a zero-shot image classification request.

    ``results`` is a list of per-image result lists, ordered by descending score.
    """

    results: list[list[ZeroShotClassificationResult]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        labels = [[r.label for r in per_img] for per_img in self.results]
        scores = [[r.score for r in per_img] for per_img in self.results]
        n = len(self.results)
        m = len(self.results[0]) if self.results else 0
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_LABELS, shape=[n, m], datatype=Datatype.BYTES, data=labels),
                ResponseOutput(name=_OUT_SCORES, shape=[n, m], datatype=Datatype.FP32, data=scores),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "ZeroShotImageClassificationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        labels_per_image = response.get_output(_OUT_LABELS).data or []
        scores_per_image = response.get_output(_OUT_SCORES).data or []
        return cls(
            results=[
                [
                    ZeroShotClassificationResult(label=lbl, score=sc)
                    for lbl, sc in zip(lbls, scs, strict=False)
                ]
                for lbls, scs in zip(labels_per_image, scores_per_image, strict=False)
            ]
        )


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class ZeroShotImageClassificationTask(BaseTask[ZeroShotImageClassificationInput, ZeroShotImageClassificationOutput]):
    """Task definition for zero-shot image classification.

    OIP v2 request tensors
    ----------------------
    * ``image``             -- ``BYTES [-1]``      images (batch)
    * ``candidate_labels``  -- ``BYTES [-1]``      candidate class label strings

    OIP v2 response tensors
    -----------------------
    * ``label``  -- ``BYTES [-1, -1]``   per-image list of matched class labels
    * ``score``  -- ``FP32  [-1, -1]``   per-image list of match scores
    """

    TASK_NAME = "zero-shot-image-classification"
    INPUT_SCHEMA = ZeroShotImageClassificationInput
    OUTPUT_SCHEMA = ZeroShotImageClassificationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
        MetadataTensor(name="candidate_labels", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
    ]
