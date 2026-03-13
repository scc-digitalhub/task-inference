# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Task factory - instantiate any supported backend/task combination by name."""
from __future__ import annotations

from typing import Any

from .tasks.base import BaseTask

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each entry maps  task_name  →  (import_path, class_name)
# The import is deferred so that heavy dependencies (torch, transformers, …)
# are only loaded when the relevant backend is actually requested.
_TRANSFORMERS_REGISTRY: dict[str, tuple[str, str]] = {
    # vision
    "image-classification": (
        "task_inference.implementations.transformers.vision.image_classification",
        "TransformersImageClassificationTask",
    ),
    "object-detection": (
        "task_inference.implementations.transformers.vision.object_detection",
        "TransformersObjectDetectionTask",
    ),
    "depth-estimation": (
        "task_inference.implementations.transformers.vision.depth_estimation",
        "TransformersDepthEstimationTask",
    ),
    "image-segmentation": (
        "task_inference.implementations.transformers.vision.image_segmentation",
        "TransformersImageSegmentationTask",
    ),
    "image-anonymization": (
        "task_inference.implementations.transformers.vision.image_anonymization",
        "TransformersImageAnonymizationTask",
    ),
    "mask-generation": (
        "task_inference.implementations.transformers.vision.mask_generation",
        "TransformersMaskGenerationTask",
    ),
    "visual-question-answering": (
        "task_inference.implementations.transformers.vision.visual_question_answering",
        "TransformersVQATask",
    ),
    "zero-shot-image-classification": (
        "task_inference.implementations.transformers.vision.zero_shot_image_classification",
        "TransformersZeroShotImageClassificationTask",
    ),
    "zero-shot-object-detection": (
        "task_inference.implementations.transformers.vision.zero_shot_object_detection",
        "TransformersZeroShotObjectDetectionTask",
    ),
    # audio
    "audio-classification": (
        "task_inference.implementations.transformers.audio.audio_classification",
        "TransformersAudioClassificationTask",
    ),
    "automatic-speech-recognition": (
        "task_inference.implementations.transformers.audio.speech_recognition",
        "TransformersASRTask",
    ),
}

_ONNXRUNTIME_REGISTRY: dict[str, tuple[str, str]] = {
    # vision
    "image-classification": (
        "task_inference.implementations.onnxruntime.vision.image_classification",
        "OnnxImageClassificationTask",
    ),
    "object-detection": (
        "task_inference.implementations.onnxruntime.vision.object_detection",
        "OnnxObjectDetectionTask",
    ),
    "depth-estimation": (
        "task_inference.implementations.onnxruntime.vision.depth_estimation",
        "OnnxDepthEstimationTask",
    ),
    "image-segmentation": (
        "task_inference.implementations.onnxruntime.vision.image_segmentation",
        "OnnxImageSegmentationTask",
    ),
    "image-anonymization": (
        "task_inference.implementations.onnxruntime.vision.image_anonymization",
        "OnnxImageAnonymizationTask",
    ),
    "visual-question-answering": (
        "task_inference.implementations.onnxruntime.vision.visual_question_answering",
        "OnnxVQATask",
    ),
    "zero-shot-image-classification": (
        "task_inference.implementations.onnxruntime.vision.zero_shot_image_classification",
        "OnnxZeroShotImageClassificationTask",
    ),
    "zero-shot-object-detection": (
        "task_inference.implementations.onnxruntime.vision.zero_shot_object_detection",
        "OnnxZeroShotObjectDetectionTask",
    ),
    # audio
    "audio-classification": (
        "task_inference.implementations.onnxruntime.audio.audio_classification",
        "OnnxAudioClassificationTask",
    ),
    "automatic-speech-recognition": (
        "task_inference.implementations.onnxruntime.audio.speech_recognition",
        "OnnxASRTask",
    ),
}

_BACKEND_REGISTRIES: dict[str, dict[str, tuple[str, str]]] = {
    "transformers": _TRANSFORMERS_REGISTRY,
    "onnxruntime": _ONNXRUNTIME_REGISTRY,
}


def create_task(
    backend: str,
    task_name: str,
    model_name: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> BaseTask:  # type: ignore[type-arg]
    """Instantiate a task implementation by backend and task name.

    Parameters
    ----------
    backend:
        Name of the inference backend.  Supported values: ``"transformers"``,
        ``"onnxruntime"``.
    task_name:
        OIP / HuggingFace pipeline task identifier, e.g.
        ``"image-classification"``, ``"automatic-speech-recognition"``.
        See :data:`supported_tasks` for the full list.
    model_name:
        HuggingFace model identifier or local path passed to the backend
        constructor, e.g. ``"google/vit-base-patch16-224"``.  When *None*
        the backend's built-in default model is used.
    model_params:
        Additional keyword arguments forwarded to the task constructor
        (e.g. ``{"device": "cuda", "chunk_length_s": 30}``).  When *None*
        an empty dict is used.

    Returns
    -------
    BaseTask
        A fully initialised task instance ready for inference.

    Raises
    ------
    ValueError
        If *backend* or *task_name* is not recognised.

    Examples
    --------
    >>> from task_inference.factory import create_task
    >>> task = create_task(
    ...     backend="transformers",
    ...     task_name="image-classification",
    ...     model_name="google/vit-base-patch16-224",
    ...     model_params={"device": "cpu"},
    ... )
    """
    if backend not in _BACKEND_REGISTRIES:
        supported = ", ".join(sorted(_BACKEND_REGISTRIES))
        raise ValueError(
            f"Unknown backend {backend!r}. Supported backends: {supported}"
        )

    registry = _BACKEND_REGISTRIES[backend]

    if task_name not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown task {task_name!r} for backend {backend!r}. "
            f"Supported tasks: {supported}"
        )

    module_path, class_name = registry[task_name]

    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    kwargs: dict[str, Any] = {}
    if model_name is not None:
        kwargs["model_name"] = model_name
    if model_params:
        kwargs.update(model_params)

    return cls(**kwargs)  # type: ignore[return-value]


def supported_tasks(backend: str | None = None) -> dict[str, list[str]]:
    """Return the tasks supported by each backend (or a single backend).

    Parameters
    ----------
    backend:
        When provided, return only the tasks for that backend.

    Returns
    -------
    dict
        Mapping of ``backend_name → [task_name, …]``.
    """
    if backend is not None:
        if backend not in _BACKEND_REGISTRIES:
            supported = ", ".join(sorted(_BACKEND_REGISTRIES))
            raise ValueError(
                f"Unknown backend {backend!r}. Supported backends: {supported}"
            )
        return {backend: sorted(_BACKEND_REGISTRIES[backend])}
    return {name: sorted(reg) for name, reg in _BACKEND_REGISTRIES.items()}
