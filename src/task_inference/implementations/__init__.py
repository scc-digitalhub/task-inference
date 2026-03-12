# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations - backend implementations."""

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
]
