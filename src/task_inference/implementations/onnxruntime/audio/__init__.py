# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations.onnxruntime.audio"""

from .audio_classification import OnnxAudioClassificationTask
from .speech_recognition import OnnxASRTask

__all__ = ["OnnxAudioClassificationTask", "OnnxASRTask"]
