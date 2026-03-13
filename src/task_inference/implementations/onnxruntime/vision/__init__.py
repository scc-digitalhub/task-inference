# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.implementations.onnxruntime.vision"""

from .depth_estimation import OnnxDepthEstimationTask
from .image_anonymization import OnnxImageAnonymizationTask
from .image_classification import OnnxImageClassificationTask
from .image_segmentation import OnnxImageSegmentationTask
from .object_detection import OnnxObjectDetectionTask
from .visual_question_answering import OnnxVQATask
from .zero_shot_image_classification import OnnxZeroShotImageClassificationTask
from .zero_shot_object_detection import OnnxZeroShotObjectDetectionTask

__all__ = [
    "OnnxDepthEstimationTask",
    "OnnxImageAnonymizationTask",
    "OnnxImageClassificationTask",
    "OnnxImageSegmentationTask",
    "OnnxObjectDetectionTask",
    "OnnxVQATask",
    "OnnxZeroShotImageClassificationTask",
    "OnnxZeroShotObjectDetectionTask",
]
