# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime audio-classification dialect adapters.

Supported dialects
------------------
input-values-logits
    Standard raw-PCM audio classifier.  Input ``input_values [B, seq_len]``
    (z-normalised float32); output ``logits [B, num_labels]``.  Covers:
    wav2vec2, HuBERT, Wav2Vec2Conformer, AST (Audio Spectrogram Transformer),
    SSAST, and any other model that follows this contract.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np

from ..base import OnnxDialectAdapter, resolve_adapter, io_names_from_session


# ---------------------------------------------------------------------------
# Per-task abstract base
# ---------------------------------------------------------------------------

class AudioClassificationAdapter(OnnxDialectAdapter):
    """Abstract adapter for audio-classification tasks."""

    def __init__(self, session: Any, do_normalize: bool) -> None:
        self._session = session
        self._do_normalize = do_normalize

    @abstractmethod
    def classify_audio(self, audio: np.ndarray) -> np.ndarray:
        """Run inference on a 1-D float32 PCM array and return
        raw logits ``[num_labels]``."""


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _normalize_if_needed(audio: np.ndarray, do_normalize: bool) -> np.ndarray:
    if do_normalize:
        mean = float(audio.mean())
        std = float(audio.std()) + 1e-9
        return (audio - mean) / std
    return audio


# ---------------------------------------------------------------------------
# Dialect: input-values-logits
# ---------------------------------------------------------------------------

class InputValuesLogitsAdapter(AudioClassificationAdapter):
    """wav2vec2 / HuBERT / AST audio-classification dialect adapter.

    Expects:
    * Input  ``input_values [1, seq_len]`` (z-normalised float32 PCM)
    * Output ``logits [1, num_labels]``
    """

    DIALECT: ClassVar[str] = "input-values-logits"

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return "input_values" in input_names and "logits" in output_names

    def classify_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = _normalize_if_needed(audio, self._do_normalize)
        input_values = audio[np.newaxis, :]  # [1, seq_len]
        out = self._session.run(["logits"], {"input_values": input_values})
        return out[0][0]  # [num_labels]


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[AudioClassificationAdapter]] = [
    InputValuesLogitsAdapter,
]


def resolve_audio_classification_adapter(
    session: Any,
    do_normalize: bool,
    config: dict[str, Any],
) -> AudioClassificationAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, do_normalize)
