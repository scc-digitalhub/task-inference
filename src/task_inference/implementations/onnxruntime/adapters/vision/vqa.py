# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime VQA dialect adapters.

Supported dialects
------------------
transformers-vilt
    ViLT (Vision-and-Language Transformer) and compatible multi-modal
    classifiers.  Five inputs: ``input_ids``, ``attention_mask``,
    ``token_type_ids``, ``pixel_values``, ``pixel_mask``; output
    ``logits [B, num_labels]``.

transformers-vilt-no-token-type
    ViLT variant without ``token_type_ids`` (some fine-tuned checkpoints
    omit this input).  Falls back gracefully when the full ViLT dialect
    does not match.
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

class VQAAdapter(OnnxDialectAdapter):
    """Abstract adapter for visual question-answering tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any], tokenizer: Any) -> None:
        self._session = session
        self._pp_cfg = pp_cfg
        self._tokenizer = tokenizer

    @abstractmethod
    def answer(self, pil_image: Image.Image, question: str) -> np.ndarray:
        """Run inference and return raw logits ``[num_labels]``."""


# ---------------------------------------------------------------------------
# Dialect: transformers-vilt  (with token_type_ids)
# ---------------------------------------------------------------------------

class ViltAdapter(VQAAdapter):
    """Full ViLT dialect adapter.

    Inputs : ``input_ids``, ``attention_mask``, ``token_type_ids``,
             ``pixel_values``, ``pixel_mask``
    Output : ``logits [1, num_labels]``
    """

    DIALECT: ClassVar[str] = "transformers-vilt"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return (
            "input_ids" in input_names
            and "pixel_values" in input_names
            and "pixel_mask" in input_names
            and "token_type_ids" in input_names
            and "logits" in output_names
        )

    def answer(self, pil_image: Image.Image, question: str) -> np.ndarray:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        _, _, ph, pw = pixel_values.shape
        pixel_mask = np.ones((1, ph, pw), dtype=np.int64)

        enc = self._tokenizer.encode(question)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)
        token_type_ids = np.array([enc.type_ids], dtype=np.int64)

        out = self._session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
            },
        )
        return out[0][0]  # [num_labels]


# ---------------------------------------------------------------------------
# Dialect: transformers-vilt-no-token-type
# ---------------------------------------------------------------------------

class ViltNoTokenTypeAdapter(VQAAdapter):
    """ViLT variant that omits ``token_type_ids``.

    Inputs : ``input_ids``, ``attention_mask``,
             ``pixel_values``, ``pixel_mask``
    Output : ``logits [1, num_labels]``
    """

    DIALECT: ClassVar[str] = "transformers-vilt-no-token-type"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return (
            "input_ids" in input_names
            and "pixel_values" in input_names
            and "pixel_mask" in input_names
            and "token_type_ids" not in input_names
            and "logits" in output_names
        )

    def answer(self, pil_image: Image.Image, question: str) -> np.ndarray:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        _, _, ph, pw = pixel_values.shape
        pixel_mask = np.ones((1, ph, pw), dtype=np.int64)

        enc = self._tokenizer.encode(question)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attention_mask = np.array([enc.attention_mask], dtype=np.int64)

        out = self._session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_mask": pixel_mask,
            },
        )
        return out[0][0]  # [num_labels]


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[VQAAdapter]] = [
    ViltAdapter,
    ViltNoTokenTypeAdapter,
]


def resolve_vqa_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
) -> VQAAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg, tokenizer)
