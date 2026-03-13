# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Vision-task ONNX dialect adapters."""

from .classification import (
    ImageClassificationAdapter,
    PixelLogitsAdapter,
    resolve_classification_adapter,
)
from .depth import (
    DepthEstimationAdapter,
    PredictedDepthAdapter,
    resolve_depth_adapter,
)
from .detection import (
    DetrAdapter,
    ObjectDetectionAdapter,
    TorchVisionDetectionAdapter,
    YolosAdapter,
    YoloV8Adapter,
    resolve_detection_adapter,
)
from .segmentation import (
    ImageSegmentationAdapter,
    SemanticLogitsAdapter,
    resolve_segmentation_adapter,
)
from .vqa import (
    ViltAdapter,
    ViltNoTokenTypeAdapter,
    VQAAdapter,
    resolve_vqa_adapter,
)
from .zero_shot_classification import (
    ClipAdapter,
    ZeroShotClassificationAdapter,
    resolve_zero_shot_classification_adapter,
)
from .zero_shot_detection import (
    OwlVitAdapter,
    ZeroShotDetectionAdapter,
    resolve_zero_shot_detection_adapter,
)

__all__ = [
    # classification
    "ImageClassificationAdapter",
    "PixelLogitsAdapter",
    "resolve_classification_adapter",
    # detection
    "ObjectDetectionAdapter",
    "DetrAdapter",
    "YolosAdapter",
    "TorchVisionDetectionAdapter",
    "YoloV8Adapter",
    "resolve_detection_adapter",
    # depth
    "DepthEstimationAdapter",
    "PredictedDepthAdapter",
    "resolve_depth_adapter",
    # segmentation
    "ImageSegmentationAdapter",
    "SemanticLogitsAdapter",
    "resolve_segmentation_adapter",
    # vqa
    "VQAAdapter",
    "ViltAdapter",
    "ViltNoTokenTypeAdapter",
    "resolve_vqa_adapter",
    # zero-shot classification
    "ZeroShotClassificationAdapter",
    "ClipAdapter",
    "resolve_zero_shot_classification_adapter",
    # zero-shot detection
    "ZeroShotDetectionAdapter",
    "OwlVitAdapter",
    "resolve_zero_shot_detection_adapter",
]
