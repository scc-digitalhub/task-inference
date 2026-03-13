# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime image-classification dialect adapters.

Supported dialects
------------------
pixel-logits
    Standard transformer / torchvision classifier.  Single image tensor
    ``pixel_values`` in, ``logits`` out.  Covers ViT, ResNet, EfficientNet,
    ConvNeXt, Swin, DenseNet, and any other model whose exported ONNX
    follows this contract.
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

class ImageClassificationAdapter(OnnxDialectAdapter):
    """Abstract adapter for image-classification tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        self._session = session
        self._pp_cfg = pp_cfg

    @abstractmethod
    def classify(self, pil_image: Image.Image) -> np.ndarray:
        """Run inference and return raw logits ``[num_classes]``."""


# ---------------------------------------------------------------------------
# Dialect: pixel-logits
# ---------------------------------------------------------------------------

class PixelLogitsAdapter(ImageClassificationAdapter):
    """Standard image-classification dialect.

    Expects:
    * Input ``pixel_values`` of shape ``[B, 3, H, W]``
    * Output ``logits`` of shape ``[B, num_classes]``

    Covers: ViT, ResNet (torchvision + transformers), EfficientNet,
    ConvNeXt, Swin Transformer, DenseNet, …
    """

    DIALECT: ClassVar[str] = "pixel-logits"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return "pixel_values" in input_names and "logits" in output_names

    def classify(self, pil_image: Image.Image) -> np.ndarray:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        out = self._session.run(["logits"], {"pixel_values": pixel_values})
        return out[0][0]  # [num_classes]


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[ImageClassificationAdapter]] = [
    PixelLogitsAdapter,
]


def resolve_classification_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
) -> ImageClassificationAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg)
