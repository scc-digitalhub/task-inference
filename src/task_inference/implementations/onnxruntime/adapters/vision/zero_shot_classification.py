# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime zero-shot image-classification dialect adapters.

Supported dialects
------------------
transformers-clip
    OpenAI CLIP and compatible dual-encoder models exported as a single
    fused ONNX graph.  Inputs: ``input_ids [N, ctx]``,
    ``attention_mask [N, ctx]``, ``pixel_values [1, 3, H, W]``; output
    ``logits_per_image [1, N]`` (cosine-similarity logits between the
    image and each label embedding).  Context length defaults to 77
    (CLIP classic) but is read from the tokenizer at runtime.
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

class ZeroShotClassificationAdapter(OnnxDialectAdapter):
    """Abstract adapter for zero-shot image-classification tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any], tokenizer: Any) -> None:
        self._session = session
        self._pp_cfg = pp_cfg
        self._tokenizer = tokenizer

    @abstractmethod
    def classify_zero_shot(
        self,
        pil_image: Image.Image,
        candidate_labels: list[str],
    ) -> np.ndarray:
        """Return softmax probabilities ``[num_labels]`` over candidate labels."""


# ---------------------------------------------------------------------------
# Dialect: transformers-clip
# ---------------------------------------------------------------------------

class ClipAdapter(ZeroShotClassificationAdapter):
    """CLIP dialect adapter.

    Inputs : ``input_ids [N, ctx]``, ``attention_mask [N, ctx]``,
             ``pixel_values [1, 3, H, W]``
    Output : ``logits_per_image [1, N]``
    """

    DIALECT: ClassVar[str] = "transformers-clip"

    def __init__(self, session: Any, pp_cfg: dict[str, Any], tokenizer: Any) -> None:
        super().__init__(session, pp_cfg, tokenizer)
        # Determine context length from tokenizer or default to 77
        ctx = 77
        try:
            ctx = tokenizer.get_vocab_size() and 77  # keep 77 as  CLIP default
        except Exception:
            pass
        self._ctx = ctx
        tokenizer.enable_truncation(max_length=ctx)
        pad_id = tokenizer.token_to_id("<|endoftext|>") or 0
        tokenizer.enable_padding(
            pad_id=pad_id, pad_token="<|endoftext|>", length=ctx
        )

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return (
            "input_ids" in input_names
            and "attention_mask" in input_names
            and "pixel_values" in input_names
            and "logits_per_image" in output_names
        )

    def classify_zero_shot(
        self,
        pil_image: Image.Image,
        candidate_labels: list[str],
    ) -> np.ndarray:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)

        encs = self._tokenizer.encode_batch(candidate_labels)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        out = self._session.run(
            ["logits_per_image"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            },
        )
        logits = out[0][0]  # [num_labels]
        return _M._softmax(logits)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[ZeroShotClassificationAdapter]] = [
    ClipAdapter,
]


def resolve_zero_shot_classification_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
) -> ZeroShotClassificationAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg, tokenizer)
