# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations.transformers.vision - Transformers vision tasks."""

from .depth_estimation import TransformersDepthEstimationTask
from .image_anonymization import TransformersImageAnonymizationTask
from .image_classification import TransformersImageClassificationTask
from .image_segmentation import TransformersImageSegmentationTask
from .mask_generation import TransformersMaskGenerationTask
from .object_detection import TransformersObjectDetectionTask
from .visual_question_answering import TransformersVQATask
from .zero_shot_image_classification import TransformersZeroShotImageClassificationTask
from .zero_shot_object_detection import TransformersZeroShotObjectDetectionTask

__all__ = [
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
