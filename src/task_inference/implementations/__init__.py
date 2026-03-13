# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations - backend implementations."""

from .onnxruntime import (
    OnnxASRTask,
    OnnxAudioClassificationTask,
    OnnxDepthEstimationTask,
    OnnxImageAnonymizationTask,
    OnnxImageClassificationTask,
    OnnxImageSegmentationTask,
    OnnxObjectDetectionTask,
    OnnxVQATask,
    OnnxZeroShotImageClassificationTask,
    OnnxZeroShotObjectDetectionTask,
)
from .transformers import (
    TransformersASRTask,
    TransformersAudioClassificationTask,
    TransformersDepthEstimationTask,
    TransformersImageAnonymizationTask,
    TransformersImageClassificationTask,
    TransformersImageSegmentationTask,
    TransformersMaskGenerationTask,
    TransformersObjectDetectionTask,
    TransformersVQATask,
    TransformersZeroShotImageClassificationTask,
    TransformersZeroShotObjectDetectionTask,
)

__all__ = [
    # transformers
    "TransformersASRTask",
    "TransformersAudioClassificationTask",
    "TransformersDepthEstimationTask",
    "TransformersImageAnonymizationTask",
    "TransformersImageClassificationTask",
    "TransformersImageSegmentationTask",
    "TransformersMaskGenerationTask",
    "TransformersObjectDetectionTask",
    "TransformersVQATask",
    "TransformersZeroShotImageClassificationTask",
    "TransformersZeroShotObjectDetectionTask",
    # onnxruntime
    "OnnxASRTask",
    "OnnxAudioClassificationTask",
    "OnnxDepthEstimationTask",
    "OnnxImageAnonymizationTask",
    "OnnxImageClassificationTask",
    "OnnxImageSegmentationTask",
    "OnnxObjectDetectionTask",
    "OnnxVQATask",
    "OnnxZeroShotImageClassificationTask",
    "OnnxZeroShotObjectDetectionTask",
]
