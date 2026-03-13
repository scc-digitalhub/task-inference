# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime dialect adapters.

Dialect adapters encapsulate the preprocessing, inference, and
postprocessing logic for a specific model I/O contract ("dialect").
Each task has its own abstract adapter base; concrete subclasses
implement a particular dialect and register themselves via the module-level
``_ADAPTERS`` list in their respective module.

The ``resolve_*_adapter()`` factory functions perform automatic dialect
detection by inspecting the ORT session's input/output tensor names and
(optionally) ``config.json``.

Usage
-----
Inside a task ``__init__``::

    from .adapters import resolve_detection_adapter

    session = self._create_session(onnx_path, providers)
    self._adapter = resolve_detection_adapter(session, pp_cfg, config)

Inside ``process()``::

    dets = self._adapter.detect(pil_image, inputs.threshold, self._id2label)

Extending
---------
To add a custom dialect:

1. Subclass the appropriate per-task abstract base
   (e.g. ``ObjectDetectionAdapter``).
2. Implement ``DIALECT``, ``accepts()``, and the task-specific run method.
3. Append the class to the module-level ``_ADAPTERS`` list **before** the
   catch-all entry (if present) in that module.
"""

from .base import OnnxDialectAdapter, io_names_from_session, resolve_adapter
from .audio import (
    ASRAdapter,
    AudioClassificationAdapter,
    InputValuesLogitsAdapter,
    Wav2Vec2CTCAdapter,
    WhisperAdapter,
    resolve_asr_adapter,
    resolve_audio_classification_adapter,
)
from .vision import (
    ClipAdapter,
    DetrAdapter,
    DepthEstimationAdapter,
    ImageClassificationAdapter,
    ImageSegmentationAdapter,
    ObjectDetectionAdapter,
    OwlVitAdapter,
    PixelLogitsAdapter,
    PredictedDepthAdapter,
    SemanticLogitsAdapter,
    TorchVisionDetectionAdapter,
    ViltAdapter,
    ViltNoTokenTypeAdapter,
    VQAAdapter,
    YolosAdapter,
    YoloV8Adapter,
    ZeroShotClassificationAdapter,
    ZeroShotDetectionAdapter,
    resolve_classification_adapter,
    resolve_depth_adapter,
    resolve_detection_adapter,
    resolve_segmentation_adapter,
    resolve_vqa_adapter,
    resolve_zero_shot_classification_adapter,
    resolve_zero_shot_detection_adapter,
)

__all__ = [
    # base
    "OnnxDialectAdapter",
    "resolve_adapter",
    "io_names_from_session",
    # vision — classification
    "ImageClassificationAdapter",
    "PixelLogitsAdapter",
    "resolve_classification_adapter",
    # vision — detection
    "ObjectDetectionAdapter",
    "DetrAdapter",
    "YolosAdapter",
    "TorchVisionDetectionAdapter",
    "YoloV8Adapter",
    "resolve_detection_adapter",
    # vision — depth
    "DepthEstimationAdapter",
    "PredictedDepthAdapter",
    "resolve_depth_adapter",
    # vision — segmentation
    "ImageSegmentationAdapter",
    "SemanticLogitsAdapter",
    "resolve_segmentation_adapter",
    # vision — vqa
    "VQAAdapter",
    "ViltAdapter",
    "ViltNoTokenTypeAdapter",
    "resolve_vqa_adapter",
    # vision — zero-shot classification
    "ZeroShotClassificationAdapter",
    "ClipAdapter",
    "resolve_zero_shot_classification_adapter",
    # vision — zero-shot detection
    "ZeroShotDetectionAdapter",
    "OwlVitAdapter",
    "resolve_zero_shot_detection_adapter",
    # audio — classification
    "AudioClassificationAdapter",
    "InputValuesLogitsAdapter",
    "resolve_audio_classification_adapter",
    # audio — asr
    "ASRAdapter",
    "Wav2Vec2CTCAdapter",
    "WhisperAdapter",
    "resolve_asr_adapter",
]
