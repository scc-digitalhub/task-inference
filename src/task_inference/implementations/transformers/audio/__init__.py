# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations.transformers.audio - Transformers audio tasks."""

from .audio_classification import TransformersAudioClassificationTask
from .speech_recognition import TransformersASRTask

__all__ = [
    "TransformersAudioClassificationTask",
    "TransformersASRTask",
]
