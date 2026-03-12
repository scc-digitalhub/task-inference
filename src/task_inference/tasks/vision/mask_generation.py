# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Mask generation (SAM-style) task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_image, encode_image
from ..base import BaseTask

# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_IMAGE = "image"
_OUT_MASKS = "mask"           # list[list[bytes]]  per-image list of PNG masks  shape [-1]
_OUT_SCORES = "score"         # list[list[float]]  per-image list of quality scores     shape [-1]

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class Point(BaseModel):
    """A 2-D prompt point in pixel coordinates."""

    x: float
    y: float


class MaskGenerationInput(BaseModel):
    """Inputs for a mask generation request.

    """

    images: list[bytes] = Field(..., description="Raw image bytes")
    mask_threshold: float = Field(0.0, ge=0.0, le=1.0, description=" Threshold to use when turning the predicted masks into binary values.")
    pred_iou_thresh: float = Field(0.88, ge=0.0, le=1.0, description="Predicted IoU threshold")
    stability_score_thresh: float = Field(0.95, ge=0.0, le=1.0, description="Mask stability score threshold")
    stability_score_offset: int = Field(1, ge=0, description="The amount to shift the cutoff when calculated the stability score")
    crops_nms_thresh: float = Field(0.7, ge=0.0, le=1.0, description="The box IoU cutoff used by non-maximal suppression to filter duplicate masks.")
    crops_n_layers: int = Field(0, ge=0, description="If crops_n_layers>0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crops.")
    crop_overlap_ratio: float = Field(512 / 1500, ge=0.0, le=1.0, description="Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of the image length. Later layers with more crops scale down this overlap.")
    crop_n_points_downscale_factor: int = Field(1, ge=1, description="The number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n.")
    
    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        request_inputs: list[RequestInput] = [
            RequestInput(
                name=_IN_IMAGE,
                shape=[len(self.images)],
                datatype=Datatype.BYTES,
                data=[encode_image(img) for img in self.images],
            )
        ]
        return InferenceRequest(
            inputs=request_inputs,
            parameters={
                "mask_threshold": self.mask_threshold,
                "pred_iou_thresh": self.pred_iou_thresh, 
                "stability_score_thresh": self.stability_score_thresh,
                "stability_score_offset": self.stability_score_offset,
                "crops_nms_thresh": self.crops_nms_thresh,
                "crops_n_layers": self.crops_n_layers,
                "crop_overlap_ratio": self.crop_overlap_ratio,
                "crop_n_points_downscale_factor": self.crop_n_points_downscale_factor,
            },
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "MaskGenerationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        image_b64_list = (next(t for t in request.inputs if t.name == _IN_IMAGE).data or [])
        params = request.parameters or {}
        return cls(
            images=[decode_image(b) for b in image_b64_list],
            mask_threshold=float(params.get("mask_threshold", 0.0)),
            pred_iou_thresh=float(params.get("pred_iou_thresh", 0.88)),
            stability_score_thresh=float(params.get("stability_score_thresh", 0.95)),
            stability_score_offset=int(params.get("stability_score_offset", 1)),
            crops_nms_thresh=float(params.get("crops_nms_thresh", 0.7)),
            crops_n_layers=int(params.get("crops_n_layers", 0)),
            crop_overlap_ratio=float(params.get("crop_overlap_ratio", 512 / 1500)),
            crop_n_points_downscale_factor=int(params.get("crop_n_points_downscale_factor", 1)),
        )


class GeneratedMask(BaseModel):
    """A single generated mask."""

    mask: bytes = Field(..., description="PNG-encoded binary mask")
    score: float = Field(..., description="Predicted IoU / quality score")


class MaskGenerationOutput(BaseModel):
    """Outputs of a mask generation request.

    ``masks`` is a list of per-image mask lists.
    """

    masks: list[list[GeneratedMask]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        masks_per = [[encode_image(m.mask) for m in img_masks] for img_masks in self.masks]
        scores_per = [[m.score for m in img_masks] for img_masks in self.masks]
        n = len(self.masks)
        m = min(len(mask_list) for mask_list in masks_per) if masks_per else 0
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_MASKS, shape=[n, m], datatype=Datatype.BYTES, data=masks_per),
                ResponseOutput(name=_OUT_SCORES, shape=[n, m], datatype=Datatype.FP32, data=scores_per),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "MaskGenerationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        masks_per = response.get_output(_OUT_MASKS).data or []
        scores_per = response.get_output(_OUT_SCORES).data or []
        return cls(
            masks=[
                [GeneratedMask(mask=decode_image(m), score=sc) for m, sc in zip(img_masks, img_scores, strict=False)]
                for img_masks, img_scores in zip(masks_per, scores_per, strict=False)
            ]
        )


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class MaskGenerationTask(BaseTask[MaskGenerationInput, MaskGenerationOutput]):
    """Task definition for mask generation (SAM-style).

    OIP v2 request tensors
    ----------------------
    * ``image``         -- ``BYTES  [-1]``      images (batch)
    * parameters: ``pred_iou_thresh``, ``stability_score_thresh``

    OIP v2 response tensors
    -----------------------
    * ``mask``   -- ``BYTES [-1, -1]``   per-image list of PNG binary masks
    * ``score``  -- ``FP32  [-1, -1]``   per-image list of predicted quality scores
    """

    TASK_NAME = "mask-generation"
    INPUT_SCHEMA = MaskGenerationInput
    OUTPUT_SCHEMA = MaskGenerationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="mask", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="mask_threshold", datatype="fp64", required=False, default=0.0, description="Threshold to binarise predicted mask logits"),
        ParameterMetadata(name="pred_iou_thresh", datatype="fp64", required=False, default=0.88, description="Predicted IoU quality threshold; masks below this score are discarded"),
        ParameterMetadata(name="stability_score_thresh", datatype="fp64", required=False, default=0.95, description="Stability score threshold; masks below this value are discarded"),
        ParameterMetadata(name="stability_score_offset", datatype="int64", required=False, default=1, description="Shift applied to the cutoff when computing the stability score"),
        ParameterMetadata(name="crops_nms_thresh", datatype="fp64", required=False, default=0.7, description="IoU cutoff used by NMS to deduplicate masks across crops"),
        ParameterMetadata(name="crops_n_layers", datatype="int64", required=False, default=0, description="Number of crop layers; 0 disables cropped re-inference"),
        ParameterMetadata(name="crop_overlap_ratio", datatype="fp64", required=False, default=round(512 / 1500, 6), description="Fraction of image length by which adjacent crops overlap"),
        ParameterMetadata(name="crop_n_points_downscale_factor", datatype="int64", required=False, default=1, description="Down-scale factor for the number of points sampled per crop layer"),
    ]
