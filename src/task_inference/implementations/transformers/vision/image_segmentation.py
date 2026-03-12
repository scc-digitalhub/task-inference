# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - image segmentation."""
from __future__ import annotations

import numpy as np
from PIL import Image
from transformers import pipeline

from ....tasks.vision.image_segmentation import (
    ImageSegmentationInput,
    ImageSegmentationOutput,
    ImageSegmentationTask,
    SegmentResult,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "facebook/mask2former-swin-large-coco-panoptic"


class TransformersImageSegmentationTask(TransformersTaskMixin, ImageSegmentationTask):
    """Image segmentation using a HuggingFace ``image-segmentation`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``image-segmentation`` compatible model on the Hub.
        Defaults to ``facebook/mask2former-swin-large-coco-panoptic``.
    device:
        Inference device.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline("image-segmentation", model=model_name, device=device)

    def process(self, inputs: ImageSegmentationInput) -> ImageSegmentationOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        kwargs: dict = {
            "threshold": inputs.threshold,
            "mask_threshold": inputs.mask_threshold,
            "overlap_mask_area_threshold": inputs.overlap_mask_area_threshold,
        }
        if inputs.subtask is not None:
            kwargs["subtask"] = inputs.subtask

        # pipeline accepts a batch (list of images) and returns list[list[dict]]
        batch_results = self._pipe(pil_images, **kwargs)

        all_segments = []
        for results in batch_results:
            segs = []
            for r in results:
                mask_pil: Image.Image = r["mask"]
                mask_bytes = self._pil_to_bytes(mask_pil, fmt="PNG")
                segs.append(
                    SegmentResult(
                        label=r.get("label", ""),
                        score=float(r.get("score", 0.0)),
                        mask=mask_bytes,
                    )
                )
            all_segments.append(segs)
        return ImageSegmentationOutput(segments=all_segments)
