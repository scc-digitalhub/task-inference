# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.tasks.audio - audio task definitions."""

from .audio_classification import (
    AudioClassificationInput,
    AudioClassificationOutput,
    AudioClassificationResult,
    AudioClassificationTask,
)
from .speech_recognition import (
    ASRChunk,
    ASRInput,
    ASROutput,
    ASRTask,
)

__all__ = [
    # audio classification
    "AudioClassificationInput",
    "AudioClassificationOutput",
    "AudioClassificationResult",
    "AudioClassificationTask",
    # ASR
    "ASRChunk",
    "ASRInput",
    "ASROutput",
    "ASRTask",
]
