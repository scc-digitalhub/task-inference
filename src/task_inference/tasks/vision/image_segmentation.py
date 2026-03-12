# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Image segmentation task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"     # BYTES [-1] images (batch)
_OUT_LABELS = "label"   # STRING [-1, -1] per-image list of segment class labels
_OUT_SCORES = "score"   # FLOAT [-1, -1] per-image list of segment scores
_OUT_MASKS = "mask"     # BYTES [-1, -1] per-image list of PNG-encoded binary masks

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class ImageSegmentationInput(BaseModel):
    """Inputs for an image segmentation request."""

    images: list[bytes] = Field(..., description="Raw image bytes")
    threshold: float = Field(0.9, ge=0.0, le=1.0, description="Score threshold")
    mask_threshold: float = Field(0.5, ge=0.0, le=1.0, description=" Threshold to use when turning the predicted masks into binary values")
    overlap_mask_area_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Overlap area threshold for mask merging"
    )
    subtask: str | None = Field(
        None,
        description="Segmentation sub-task: 'panoptic', 'instance', or 'semantic'",
    )

    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        params: dict = {
            "threshold": self.threshold,
            "mask_threshold": self.mask_threshold,
            "overlap_mask_area_threshold": self.overlap_mask_area_threshold,
        }
        if self.subtask is not None:
            params["subtask"] = self.subtask
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name=_IN_IMAGE,
                    shape=[len(self.images)],
                    datatype=Datatype.BYTES,
                    data=[encode_image(img) for img in self.images],
                )
            ],
            parameters=params,
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "ImageSegmentationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        images = [decode_image(b) for b in (request.inputs[0].data or [])]
        params = request.parameters or {}
        return cls(
            images=images,
            threshold=float(params.get("threshold", 0.9)),
            mask_threshold=float(params.get("mask_threshold", 0.5)),
            overlap_mask_area_threshold=float(params.get("overlap_mask_area_threshold", 0.5)),
            subtask=params.get("subtask"),
        )


class SegmentResult(BaseModel):
    """A single detected segment."""

    label: str
    score: float
    mask: bytes = Field(..., description="PNG-encoded binary mask")


class ImageSegmentationOutput(BaseModel):
    """Outputs of an image segmentation request.

    ``segments`` is a list of per-image segment lists.
    """

    segments: list[list[SegmentResult]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        labels = [[s.label for s in segs] for segs in self.segments]
        scores = [[s.score for s in segs] for segs in self.segments]
        masks = [[encode_image(s.mask) for s in segs] for segs in self.segments]
        n = len(self.segments)
        m = min(len(segs) for segs in self.segments) if self.segments else 0
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_LABELS, shape=[n, m], datatype=Datatype.BYTES, data=labels),
                ResponseOutput(name=_OUT_SCORES, shape=[n, m], datatype=Datatype.FP32, data=scores),
                ResponseOutput(name=_OUT_MASKS, shape=[n, m], datatype=Datatype.BYTES, data=masks),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "ImageSegmentationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        labels_per = response.get_output(_OUT_LABELS).data or []
        scores_per = response.get_output(_OUT_SCORES).data or []
        masks_per = response.get_output(_OUT_MASKS).data or []
        return cls(
            segments=[
                [
                    SegmentResult(label=lbl, score=sc, mask=decode_image(m))
                    for lbl, sc, m in zip(lbls, scs, msks, strict=False)
                ]
                for lbls, scs, msks in zip(labels_per, scores_per, masks_per, strict=False)
            ]
        )


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class ImageSegmentationTask(BaseTask[ImageSegmentationInput, ImageSegmentationOutput]):
    """Task definition for image segmentation.

    OIP v2 request tensors
    ----------------------
    * ``image``   -- ``BYTES [-1]``    images (batch)
    * parameters: ``threshold``, ``mask_threshold``, ``overlap_mask_area_threshold``, ``subtask``

    OIP v2 response tensors
    -----------------------
    * ``labels``  -- ``BYTES  [-1, -1]``   per-image list of segment class labels
    * ``scores``  -- ``FP32   [-1, -1]``   per-image list of segment confidence scores
    * ``masks``   -- ``BYTES  [-1, -1]``   per-image list of PNG-encoded binary masks
    """

    TASK_NAME = "image-segmentation"
    INPUT_SCHEMA = ImageSegmentationInput
    OUTPUT_SCHEMA = ImageSegmentationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
        MetadataTensor(name="mask", datatype=Datatype.BYTES, shape=[-1, -1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="threshold", datatype="fp64", required=False, default=0.9, description="Minimum confidence score for a segment to be returned"),
        ParameterMetadata(name="mask_threshold", datatype="fp64", required=False, default=0.5, description="Threshold to binarise predicted mask logits"),
        ParameterMetadata(name="overlap_mask_area_threshold", datatype="fp64", required=False, default=0.5, description="Overlap area threshold used when merging masks"),
        ParameterMetadata(name="subtask", datatype="string", required=False, default=None, description="Segmentation sub-task: 'panoptic', 'instance', or 'semantic'"),
    ]
