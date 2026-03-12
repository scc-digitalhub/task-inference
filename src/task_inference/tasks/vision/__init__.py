# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.tasks.vision - vision task definitions."""

from .depth_estimation import (
    DepthEstimationInput,
    DepthEstimationOutput,
    DepthEstimationTask,
)
from .image_anonymization import (
    AnonymizationStrategy,
    ImageAnonymizationInput,
    ImageAnonymizationOutput,
    ImageAnonymizationTask,
)
from .image_classification import (
    ClassificationResult,
    ImageClassificationInput,
    ImageClassificationOutput,
    ImageClassificationTask,
)
from .image_segmentation import (
    ImageSegmentationInput,
    ImageSegmentationOutput,
    ImageSegmentationTask,
    SegmentResult,
)
from .mask_generation import (
    GeneratedMask,
    MaskGenerationInput,
    MaskGenerationOutput,
    MaskGenerationTask,
    Point,
)
from .object_detection import (
    BoundingBox,
    DetectedObject,
    ObjectDetectionInput,
    ObjectDetectionOutput,
    ObjectDetectionTask,
)
from .visual_question_answering import (
    VQAAnswer,
    VQAInput,
    VQAOutput,
    VQATask,
)
from .zero_shot_image_classification import (
    ZeroShotClassificationResult,
    ZeroShotImageClassificationInput,
    ZeroShotImageClassificationOutput,
    ZeroShotImageClassificationTask,
)
from .zero_shot_object_detection import (
    ZeroShotObjectDetectionInput,
    ZeroShotObjectDetectionOutput,
    ZeroShotObjectDetectionTask,
)

__all__ = [
    # depth estimation
    "DepthEstimationInput",
    "DepthEstimationOutput",
    "DepthEstimationTask",
    # image anonymization
    "AnonymizationStrategy",
    "ImageAnonymizationInput",
    "ImageAnonymizationOutput",
    "ImageAnonymizationTask",
    # image classification
    "ClassificationResult",
    "ImageClassificationInput",
    "ImageClassificationOutput",
    "ImageClassificationTask",
    # image segmentation
    "ImageSegmentationInput",
    "ImageSegmentationOutput",
    "ImageSegmentationTask",
    "SegmentResult",
    # mask generation
    "GeneratedMask",
    "MaskGenerationInput",
    "MaskGenerationOutput",
    "MaskGenerationTask",
    "Point",
    # object detection
    "BoundingBox",
    "DetectedObject",
    "ObjectDetectionInput",
    "ObjectDetectionOutput",
    "ObjectDetectionTask",
    # vqa
    "VQAAnswer",
    "VQAInput",
    "VQAOutput",
    "VQATask",
    # zero-shot image classification
    "ZeroShotClassificationResult",
    "ZeroShotImageClassificationInput",
    "ZeroShotImageClassificationOutput",
    "ZeroShotImageClassificationTask",
    # zero-shot object detection
    "ZeroShotObjectDetectionInput",
    "ZeroShotObjectDetectionOutput",
    "ZeroShotObjectDetectionTask",
]
