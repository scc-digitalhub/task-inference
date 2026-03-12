# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations.transformers - HuggingFace reference backend."""

from .audio import (
    TransformersASRTask,
    TransformersAudioClassificationTask,
)
from .vision import (
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
    # vision
    "TransformersDepthEstimationTask",
    "TransformersImageAnonymizationTask",
    "TransformersImageClassificationTask",
    "TransformersImageSegmentationTask",
    "TransformersMaskGenerationTask",
    "TransformersObjectDetectionTask",
    "TransformersVQATask",
    "TransformersZeroShotImageClassificationTask",
    "TransformersZeroShotObjectDetectionTask",
    # audio
    "TransformersASRTask",
    "TransformersAudioClassificationTask",
]
