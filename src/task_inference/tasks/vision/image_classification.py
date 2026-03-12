# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Image classification task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"      # BYTES [-1]            input images
_OUT_LABELS = "label"    # STRING [-1, -1]       per-image list of predicted class labels   
_OUT_SCORES = "score"    # FP32   [-1, -1]       per-image list of confidence scores

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class ImageClassificationInput(BaseModel):
    """Inputs for an image classification request."""

    images: list[bytes] = Field(..., description="Raw image bytes (PNG / JPEG / ...)")
    top_k: int = Field(5, ge=1, description="Number of top predictions to return")

    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name=_IN_IMAGE,
                    shape=[len(self.images)],
                    datatype=Datatype.BYTES,
                    data=[encode_image(img) for img in self.images],
                )
            ],
            parameters={"top_k": self.top_k},
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "ImageClassificationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        images = [decode_image(b) for b in (request.inputs[0].data or [])]
        top_k = (request.parameters or {}).get("top_k", 5)
        return cls(images=images, top_k=top_k)


class ClassificationResult(BaseModel):
    """A single label-score pair."""

    label: str
    score: float


class ImageClassificationOutput(BaseModel):
    """Outputs of an image classification request.

    ``results`` is a list of per-image result lists.  Each inner list contains
    up to ``top_k`` :class:`ClassificationResult` objects ordered by descending
    score.
    """

    results: list[list[ClassificationResult]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        labels = [[r.label for r in per_img] for per_img in self.results]
        scores = [[r.score for r in per_img] for per_img in self.results]
        n = len(self.results)
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_LABELS, shape=[n, len(self.results[0]) if self.results else 0], datatype=Datatype.BYTES, data=labels),
                ResponseOutput(name=_OUT_SCORES, shape=[n, len(self.results[0]) if self.results else 0], datatype=Datatype.FP32, data=scores),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "ImageClassificationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        labels_per_image = response.get_output(_OUT_LABELS).data or []
        scores_per_image = response.get_output(_OUT_SCORES).data or []
        return cls(
            results=[
                [
                    ClassificationResult(label=lbl, score=sc)
                    for lbl, sc in zip(lbls, scs, strict=False)
                ]
                for lbls, scs in zip(labels_per_image, scores_per_image, strict=False)
            ]
        )


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class ImageClassificationTask(BaseTask[ImageClassificationInput, ImageClassificationOutput]):
    """Task definition for image classification.

    OIP v2 request tensors
    ----------------------
    * ``image``   -- ``BYTES [-1]``   images (batch)
    * parameter ``top_k`` -- ``int``

    OIP v2 response tensors
    -----------------------
    * ``labels``  -- ``STRING [-1, -1]``  per-image list of predicted class labels
    * ``scores``  -- ``FP32   [-1, -1]``  per-image list of confidence scores
    """

    TASK_NAME = "image-classification"
    INPUT_SCHEMA = ImageClassificationInput
    OUTPUT_SCHEMA = ImageClassificationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="top_k", datatype="int64", required=False, default=5, description="Number of top predictions to return per image"),
    ]
