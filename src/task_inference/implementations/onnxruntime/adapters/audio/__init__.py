# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Audio-task ONNX dialect adapters."""

from .asr import (
    ASRAdapter,
    Wav2Vec2CTCAdapter,
    WhisperAdapter,
    resolve_asr_adapter,
)
from .classification import (
    AudioClassificationAdapter,
    InputValuesLogitsAdapter,
    resolve_audio_classification_adapter,
)

__all__ = [
    # asr
    "ASRAdapter",
    "Wav2Vec2CTCAdapter",
    "WhisperAdapter",
    "resolve_asr_adapter",
    # classification
    "AudioClassificationAdapter",
    "InputValuesLogitsAdapter",
    "resolve_audio_classification_adapter",
]
