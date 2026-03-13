# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime image-segmentation dialect adapters.

Supported dialects
------------------
semantic-logits
    Encoder-only semantic segmentation models.  Input ``pixel_values
    [B, 3, H, W]``; output ``logits [B, C, H', W']``.  Covers:

    * **SegFormer** — H' = H/4, W' = W/4 (stride-4); logits are bilinearly
      upsampled to the original resolution before argmax.
    * **DeepLabV3 / DeepLabV3+** (torchvision export) — H' = H, W' = W
      (full resolution); the same upsample path is effectively a no-op.
    * **MobileViT-seg** and other models that follow the same contract.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from ..base import OnnxDialectAdapter, resolve_adapter, io_names_from_session
from ...base import OnnxRuntimeTaskMixin as _M


# ---------------------------------------------------------------------------
# Per-task abstract base
# ---------------------------------------------------------------------------

class ImageSegmentationAdapter(OnnxDialectAdapter):
    """Abstract adapter for semantic image-segmentation tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        self._session = session
        self._pp_cfg = pp_cfg

    @abstractmethod
    def segment(self, pil_image: Image.Image) -> np.ndarray:
        """Run inference and return class logits ``[C, orig_H, orig_W]``
        upsampled to the original image resolution."""


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _upsample_logits(
    logits: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Bilinear-upsample class logits ``[C, H, W]`` → ``[C, target_h, target_w]``."""
    num_classes = logits.shape[0]
    upsampled = np.zeros((num_classes, target_h, target_w), dtype=np.float32)
    for c in range(num_classes):
        upsampled[c] = np.array(
            Image.fromarray(logits[c]).resize((target_w, target_h), Image.BILINEAR),
            dtype=np.float32,
        )
    return upsampled


# ---------------------------------------------------------------------------
# Dialect: semantic-logits
# ---------------------------------------------------------------------------

class SemanticLogitsAdapter(ImageSegmentationAdapter):
    """SegFormer / DeepLabV3 semantic-segmentation dialect adapter.

    Expects:
    * Input  ``pixel_values [B, 3, H, W]``
    * Output ``logits [B, C, H', W']``

    ``H'`` / ``W'`` may be smaller than the input (e.g. stride-4 models)
    or equal; bilinear upsampling is always applied regardless.
    """

    DIALECT: ClassVar[str] = "semantic-logits"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return "pixel_values" in input_names and "logits" in output_names

    def segment(self, pil_image: Image.Image) -> np.ndarray:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        orig_w, orig_h = pil_image.size

        out = self._session.run(["logits"], {"pixel_values": pixel_values})
        logits = out[0][0]  # [C, H', W']

        return _upsample_logits(logits, orig_h, orig_w)  # [C, orig_H, orig_W]


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[ImageSegmentationAdapter]] = [
    SemanticLogitsAdapter,
]


def resolve_segmentation_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
) -> ImageSegmentationAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg)
