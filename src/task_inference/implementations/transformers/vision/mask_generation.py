# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - mask generation (SAM)."""
from __future__ import annotations

import numpy as np
from PIL import Image
from transformers import pipeline

from ....tasks.vision.mask_generation import (
    GeneratedMask,
    MaskGenerationInput,
    MaskGenerationOutput,
    MaskGenerationTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "facebook/sam-vit-base"


class TransformersMaskGenerationTask(TransformersTaskMixin, MaskGenerationTask):
    """Mask generation (SAM-style) using a HuggingFace ``mask-generation`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``mask-generation`` compatible model on the Hub.
        Defaults to ``facebook/sam-vit-base``.
    device:
        Inference device.
    points_per_batch:
        Number of points processed in parallel (automatic mode only).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | int = "cpu",
        points_per_batch: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._points_per_batch = points_per_batch
        self._pipe = pipeline("mask-generation", model=model_name, device=device, points_per_batch=points_per_batch)

    def _process_single(self, pil_image: Image.Image, inputs: MaskGenerationInput) -> list[GeneratedMask]:
        """Run the SAM pipeline on a single image."""
        pipeline_kwargs: dict = {
            "mask_threshold": inputs.mask_threshold,
            "pred_iou_thresh": inputs.pred_iou_thresh,
            "stability_score_thresh": inputs.stability_score_thresh,
            "stability_score_offset": inputs.stability_score_offset,
            "crops_nms_thresh": inputs.crops_nms_thresh,
            "crops_n_layers": inputs.crops_n_layers,
            "crop_overlap_ratio": inputs.crop_overlap_ratio,
            "crop_n_points_downscale_factor": inputs.crop_n_points_downscale_factor,
        }

        result = self._pipe(pil_image, **pipeline_kwargs)

        generated: list[GeneratedMask] = []
        for mask, score in zip(result.get("masks", []), result.get("scores", []), strict=False):
            if isinstance(mask, np.ndarray):
                pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            elif isinstance(mask, Image.Image):
                pil_mask = mask
            else:
                pil_mask = Image.fromarray(np.array(mask, dtype=np.uint8))
            generated.append(GeneratedMask(mask=self._pil_to_bytes(pil_mask), score=float(score)))
        return generated

    def process(self, inputs: MaskGenerationInput) -> MaskGenerationOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        # SAM pipeline processes one image at a time; loop over the batch
        return MaskGenerationOutput(
            masks=[self._process_single(pil_img, inputs) for pil_img in pil_images]
        )
