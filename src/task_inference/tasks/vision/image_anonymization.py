# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Image anonymization task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"                           # BYTES [-1]     input images
_OUT_IMAGE = "anonymized_image"               # BYTES [-1]     anonymized images
_OUT_NUM_REGIONS = "num_regions_anonymized"   # INT [-1]       number of regions anonymized

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class AnonymizationStrategy(str, Enum):
    """How to obscure the detected sensitive regions."""

    BLUR = "blur"
    PIXELATE = "pixelate"
    BLACK_BOX = "black_box"


class ImageAnonymizationInput(BaseModel):
    """Inputs for an image anonymization request."""

    images: list[bytes] = Field(..., description="Raw image bytes")
    strategy: AnonymizationStrategy = Field(
        AnonymizationStrategy.BLUR,
        description="Anonymization strategy applied to detected regions",
    )
    blur_radius: int = Field(
        51,
        ge=1,
        description="Blur kernel radius (used for BLUR and PIXELATE strategies)",
    )
    threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Score threshold for region detection"
    )
    classes: list[str] | None = Field(
        None,
        description=(
            "Object classes to anonymize, e.g. ['person', 'face']. "
            "When None all detected objects are anonymized."
        ),
    )

    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        params: dict = {
            "strategy": self.strategy.value,
            "blur_radius": self.blur_radius,
            "threshold": self.threshold,
        }
        if self.classes is not None:
            params["classes"] = self.classes
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
    def from_inference_request(cls, request: InferenceRequest) -> "ImageAnonymizationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        image_b64_list = request.inputs[0].data or []

        params = request.parameters or {}
        classes = params.get("classes")
        return cls(
            images=[decode_image(img_b64) for img_b64 in image_b64_list],
            strategy=AnonymizationStrategy(params.get("strategy", AnonymizationStrategy.BLUR)),
            blur_radius=int(params.get("blur_radius", 51)),
            threshold=float(params.get("threshold", 0.5)),
            classes=list(classes) if classes is not None else None,
        )


class ImageAnonymizationOutput(BaseModel):
    """Outputs of an image anonymization request."""

    images: list[bytes] = Field(..., description="PNG-encoded anonymized images")
    num_regions_anonymized: list[int] = Field(
        ..., description="Numbers of regions that were anonymized"
    )

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(
                    name=_OUT_IMAGE,
                    shape=[len(self.images)],
                    datatype=Datatype.BYTES,
                    data=[encode_image(img) for img in self.images],
                ),
                ResponseOutput(
                    name=_OUT_NUM_REGIONS,
                    shape=[len(self.num_regions_anonymized)],

                    datatype=Datatype.INT32,
                    data=self.num_regions_anonymized,
                ),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "ImageAnonymizationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        img_b64_list = response.get_output(_OUT_IMAGE).data or [""]
        num_regions_list = response.get_output(_OUT_NUM_REGIONS).data or [0]
        return cls(
            images=[decode_image(img_b64) for img_b64 in img_b64_list],
            num_regions_anonymized=num_regions_list,
        )



# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class ImageAnonymizationTask(BaseTask[ImageAnonymizationInput, ImageAnonymizationOutput]):
    """Task definition for image anonymization.

    The task relies on an object detector to locate sensitive regions
    (e.g. faces, persons) and then applies the chosen :class:`AnonymizationStrategy`.

    OIP v2 request tensors
    ----------------------
    * ``image``  - ``BYTES [-1]``  input image
    * parameters: ``strategy``, ``blur_radius``, ``threshold``,
      ``classes``

    OIP v2 response tensors
    -----------------------
    * ``anonymized_image``       - ``BYTES  [-1]``  output image
    * ``num_regions_anonymized`` - ``INT32  [-1]``  count of anonymized regions
    """

    TASK_NAME = "image-anonymization"
    INPUT_SCHEMA = ImageAnonymizationInput
    OUTPUT_SCHEMA = ImageAnonymizationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="anonymized_image", datatype=Datatype.BYTES, shape=[-1]),
        MetadataTensor(name="num_regions_anonymized", datatype=Datatype.INT32, shape=[-1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(
            name="strategy",
            datatype="string",
            required=False,
            default="blur",
            description="Anonymization strategy: 'blur', 'pixelate', or 'black_box'",
        ),
        ParameterMetadata(
            name="blur_radius",
            datatype="int64",
            required=False,
            default=51,
            description="Blur kernel radius (used for BLUR and PIXELATE strategies)",
        ),
        ParameterMetadata(
            name="threshold",
            datatype="fp64",
            required=False,
            default=0.5,
            description="Score threshold for region detection",
        ),
        ParameterMetadata(
            name="classes",
            datatype="string",
            required=False,
            default=None,
            description="Object classes to anonymize; when absent all detected objects are anonymized",
        ),
    ]
