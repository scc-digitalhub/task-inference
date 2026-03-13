# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations.onnxruntime - ONNX Runtime backend."""

from .audio import (
    OnnxASRTask,
    OnnxAudioClassificationTask,
)
from .vision import (
    OnnxDepthEstimationTask,
    OnnxImageAnonymizationTask,
    OnnxImageClassificationTask,
    OnnxImageSegmentationTask,
    OnnxObjectDetectionTask,
    OnnxVQATask,
    OnnxZeroShotImageClassificationTask,
    OnnxZeroShotObjectDetectionTask,
)

__all__ = [
    # vision
    "OnnxDepthEstimationTask",
    "OnnxImageAnonymizationTask",
    "OnnxImageClassificationTask",
    "OnnxImageSegmentationTask",
    "OnnxObjectDetectionTask",
    "OnnxVQATask",
    "OnnxZeroShotImageClassificationTask",
    "OnnxZeroShotObjectDetectionTask",
    # audio
    "OnnxASRTask",
    "OnnxAudioClassificationTask",
]
