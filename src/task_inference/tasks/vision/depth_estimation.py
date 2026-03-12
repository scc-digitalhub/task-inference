# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Depth estimation task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from task_inference.protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, RequestInput, ResponseOutput
from task_inference.utils import decode_image, encode_image
from task_inference.tasks.base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"                         # BYTES [-1]        input images
_OUT_PREDICTED_DEPTH = "predicted_depth"    # FP32 [-1, H, W]   flat row-major predicted depth values in meters
_OUT_DEPTH = "depth"                        # BYTES [-1]        optional PNG visualisations

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------

class DepthEstimationInput(BaseModel):
    """Inputs for a depth estimation request."""

    images: list[bytes] = Field(..., description="Raw image bytes")

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
            ]
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "DepthEstimationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        image_b64_list = request.inputs[0].data or []
        return cls(images=[decode_image(img_b64) for img_b64 in image_b64_list])


class DepthEstimationOutput(BaseModel):
    """Outputs of a depth estimation request.

    ``depth`` is a row-major flat list of ``float`` values representing the
    predicted relative depth map.  Reshape to ``(height, width)`` to obtain
    the 2-D map.
    """

    predicted_depth: list[list[float]] = Field(..., description="Flat row-major depth maps values in meters")
    width: int
    height: int
    depth: list[bytes] | None = Field(
        None, description="Optional PNG-encoded 8-bit visualisation of the depth maps"
    )

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        outputs = [
            ResponseOutput(
                name=_OUT_PREDICTED_DEPTH,
                shape=[len(self.predicted_depth), self.height, self.width],
                datatype=Datatype.FP32,
                data=self.predicted_depth,
            )
        ]
        if self.depth is not None:
            outputs.append(
                ResponseOutput(
                    name=_OUT_DEPTH,
                    shape=[len(self.depth)],
                    datatype=Datatype.BYTES,
                    data=[encode_image(img) for img in self.depth],
                )
            )
        return InferenceResponse(model_name=model_name, outputs=outputs)

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "DepthEstimationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        depth_tensor = response.get_output(_OUT_PREDICTED_DEPTH)
        num, h, w = depth_tensor.shape
        depth_flat = depth_tensor.data or []
        depth_imgs: list[bytes] | None = None
        try:
            image_tensor = response.get_output(_OUT_DEPTH)
            depth_imgs = [decode_image(image_tensor.data[i]) for i in range(num)]
        except (KeyError, IndexError, TypeError):
            pass
        return cls(predicted_depth=depth_flat, height=h, width=w, depth=depth_imgs)


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class DepthEstimationTask(BaseTask[DepthEstimationInput, DepthEstimationOutput]):
    """Task definition for monocular depth estimation.

    OIP v2 request tensors
    ----------------------
    * ``image``             - ``BYTES [-1]``         input images

    OIP v2 response tensors
    -----------------------
    * ``predicted_depth``   - ``FP32  [-1, H, W]``  flat row-major predicted depth values
    * ``depth``             - ``BYTES [-1]``        (optional) PNG visualisation
    """

    TASK_NAME = "depth-estimation"
    INPUT_SCHEMA = DepthEstimationInput
    OUTPUT_SCHEMA = DepthEstimationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="predicted_depth", datatype=Datatype.FP32, shape=[-1, -1, -1]),
        MetadataTensor(name="depth", datatype=Datatype.BYTES, shape=[-1]),
    ]
