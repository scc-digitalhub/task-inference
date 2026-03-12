# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Object detection task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"     # BYTES [-1]        images (batch)
_OUT_LABELS = "label"   # BYTES [-1, -1]    per-image list of detected class labels
_OUT_SCORES = "score"   # FP32 [-1, -1]    per-image list of detection confidence scores
_OUT_BOXES = "box"      # FP32 [-1, -1, 4]  per-image list of bounding boxes [xmin, ymin, xmax, ymax]

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class ObjectDetectionInput(BaseModel):
    """Inputs for an object detection request."""

    images: list[bytes] = Field(..., description="Raw image bytes")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Score threshold")

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
            parameters={"threshold": self.threshold},
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "ObjectDetectionInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        images = [decode_image(b) for b in (request.inputs[0].data or [])]
        threshold = (request.parameters or {}).get("threshold", 0.5)
        return cls(images=images, threshold=threshold)


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float


class DetectedObject(BaseModel):
    """A single detected object."""

    label: str
    score: float
    box: BoundingBox


class ObjectDetectionOutput(BaseModel):
    """Outputs of an object detection request.

    ``detections`` is a list of per-image detection lists.
    """

    detections: list[list[DetectedObject]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        labels = [[d.label for d in dets] for dets in self.detections]
        scores = [[d.score for d in dets] for dets in self.detections]
        boxes = [
            [v for d in dets for v in (d.box.xmin, d.box.ymin, d.box.xmax, d.box.ymax)]
            for dets in self.detections
        ]
        n = len(self.detections)
        m = min(len(lbls) for lbls in labels) if labels else 0
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_LABELS, shape=[n, m], datatype=Datatype.BYTES, data=labels),
                ResponseOutput(name=_OUT_SCORES, shape=[n, m], datatype=Datatype.FP32, data=scores),
                ResponseOutput(name=_OUT_BOXES, shape=[n, m, 4], datatype=Datatype.FP32, data=boxes),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "ObjectDetectionOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        labels_per = response.get_output(_OUT_LABELS).data or []
        scores_per = response.get_output(_OUT_SCORES).data or []
        boxes_per = response.get_output(_OUT_BOXES).data or []
        detections = []
        for lbls, scs, bxs in zip(labels_per, scores_per, boxes_per, strict=False):
            img_dets = []
            for i, (lbl, sc) in enumerate(zip(lbls, scs, strict=False)):
                xmin, ymin, xmax, ymax = bxs[i * 4 : i * 4 + 4]
                img_dets.append(
                    DetectedObject(
                        label=lbl,
                        score=sc,
                        box=BoundingBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
                    )
                )
            detections.append(img_dets)
        return cls(detections=detections)


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class ObjectDetectionTask(BaseTask[ObjectDetectionInput, ObjectDetectionOutput]):
    """Task definition for object detection.

    OIP v2 request tensors
    ----------------------
    * ``image``   -- ``BYTES [-1]``     images (batch)
    * parameter ``threshold`` -- ``float``

    OIP v2 response tensors
    -----------------------
    * ``label``  -- ``STRING [-1, -1]``    per-image list of detected class labels
    * ``score``  -- ``FP32   [-1, -1]``    per-image list of detection confidence scores
    * ``box``    -- ``FP32   [-1, -1, 4]``    per-image list of bounding boxes [xmin, ymin, xmax, ymax]
    """

    TASK_NAME = "object-detection"
    INPUT_SCHEMA = ObjectDetectionInput
    OUTPUT_SCHEMA = ObjectDetectionOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
        MetadataTensor(name="box", datatype=Datatype.FP32, shape=[-1, -1, 4]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="threshold", datatype="fp64", required=False, default=0.5, description="Minimum confidence score for a detection to be returned"),
    ]
