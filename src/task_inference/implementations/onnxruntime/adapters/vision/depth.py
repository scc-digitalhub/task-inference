# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime depth-estimation dialect adapters.

Supported dialects
------------------
predicted-depth
    DPT / GLPN style.  Input ``pixel_values [B, 3, H, W]``; output
    ``predicted_depth [B, H', W']``.  The implementation upsamples the
    raw depth map back to the original image resolution before returning.
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

class DepthEstimationAdapter(OnnxDialectAdapter):
    """Abstract adapter for depth-estimation tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        self._session = session
        self._pp_cfg = pp_cfg

    @abstractmethod
    def estimate(self, pil_image: Image.Image) -> np.ndarray:
        """Run inference and return a float32 depth map ``[H, W]``
        upsampled to the original image dimensions."""


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _resize_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear-resize a float32 depth map to ``(target_h, target_w)``."""
    d_min, d_max = float(depth.min()), float(depth.max())
    if d_max > d_min:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        resized = np.array(
            Image.fromarray(norm, mode="L").resize((target_w, target_h), Image.BILINEAR),
            dtype=np.float32,
        )
        return resized / 255.0 * (d_max - d_min) + d_min
    return np.full((target_h, target_w), d_min, dtype=np.float32)


# ---------------------------------------------------------------------------
# Dialect: predicted-depth
# ---------------------------------------------------------------------------

class PredictedDepthAdapter(DepthEstimationAdapter):
    """DPT / GLPN dialect adapter.

    Expects:
    * Input  ``pixel_values [B, 3, H, W]``
    * Output ``predicted_depth [B, H', W']``
    """

    DIALECT: ClassVar[str] = "predicted-depth"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return "pixel_values" in input_names and "predicted_depth" in output_names

    def estimate(self, pil_image: Image.Image) -> np.ndarray:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        orig_w, orig_h = pil_image.size

        out = self._session.run(["predicted_depth"], {"pixel_values": pixel_values})
        depth = out[0][0]  # [H', W']

        return _resize_depth(depth, orig_h, orig_w)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[DepthEstimationAdapter]] = [
    PredictedDepthAdapter,
]


def resolve_depth_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
) -> DepthEstimationAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg)
