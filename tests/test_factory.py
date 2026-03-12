# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the task factory (no model loading required)."""
from __future__ import annotations

import pytest

from task_inference.factory import create_task, supported_tasks
from task_inference.tasks.audio.audio_classification import AudioClassificationTask
from task_inference.tasks.audio.speech_recognition import ASRTask
from task_inference.tasks.vision.depth_estimation import DepthEstimationTask
from task_inference.tasks.vision.image_anonymization import ImageAnonymizationTask
from task_inference.tasks.vision.image_classification import ImageClassificationTask
from task_inference.tasks.vision.image_segmentation import ImageSegmentationTask
from task_inference.tasks.vision.mask_generation import MaskGenerationTask
from task_inference.tasks.vision.object_detection import ObjectDetectionTask
from task_inference.tasks.vision.visual_question_answering import VQATask


# ---------------------------------------------------------------------------
# Stub implementations (no real models needed)
# ---------------------------------------------------------------------------

class _FakeVisionTask(ImageClassificationTask):
    def __init__(self, model_name: str = "stub", **kwargs):
        self.model_name = model_name

    def process(self, inputs):  # type: ignore[override]
        raise NotImplementedError


# Patch the registry in create_task so it points to this light stub class.
# We test the factory routing logic, not the heavy Transformers backend.

import task_inference.factory as _factory_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _patch_registry(monkeypatch):
    """Replace every registry entry with a lightweight stub constructor."""

    class _Stub:
        """Generic task stub that records constructor kwargs."""

        def __init__(self, model_name: str = "stub", **kwargs):
            self.model_name = model_name
            self.init_kwargs = kwargs

        def process(self, inputs):  # pragma: no cover
            raise NotImplementedError

    # Build a patched registry: same keys, but all point to _Stub via a fake module.
    import types

    fake_mod = types.ModuleType("_stub_module")
    fake_mod._Stub = _Stub  # type: ignore[attr-defined]

    patched: dict[str, dict[str, tuple[str, str]]] = {
        backend: {task: ("_stub_module", "_Stub") for task in tasks}
        for backend, tasks in _factory_mod._BACKEND_REGISTRIES.items()
    }

    import sys
    sys.modules["_stub_module"] = fake_mod
    monkeypatch.setattr(_factory_mod, "_BACKEND_REGISTRIES", patched)
    yield
    sys.modules.pop("_stub_module", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_task_returns_instance():
    task = create_task("transformers", "image-classification", "some/model")
    assert task.model_name == "some/model"


def test_create_task_forwards_model_params():
    task = create_task(
        "transformers",
        "automatic-speech-recognition",
        "openai/whisper-base",
        {"device": "cuda", "chunk_length_s": 15},
    )
    assert task.init_kwargs == {"device": "cuda", "chunk_length_s": 15}  # type: ignore[attr-defined]


def test_create_task_model_params_none():
    task = create_task("transformers", "audio-classification", "my/model", None)
    assert task.init_kwargs == {}  # type: ignore[attr-defined]


def test_create_task_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        create_task("nonexistent", "image-classification", "m")


def test_create_task_unknown_task():
    with pytest.raises(ValueError, match="Unknown task"):
        create_task("transformers", "not-a-task", "m")


def test_supported_tasks_all():
    result = supported_tasks()
    assert "transformers" in result
    tasks = result["transformers"]
    assert "image-classification" in tasks
    assert "automatic-speech-recognition" in tasks


def test_supported_tasks_single_backend():
    result = supported_tasks("transformers")
    assert list(result.keys()) == ["transformers"]


def test_supported_tasks_unknown_backend():
    with pytest.raises(ValueError, match="Unknown backend"):
        supported_tasks("nonexistent")


@pytest.mark.parametrize(
    "task_name",
    [
        "image-classification",
        "object-detection",
        "depth-estimation",
        "image-segmentation",
        "image-anonymization",
        "mask-generation",
        "visual-question-answering",
        "zero-shot-image-classification",
        "zero-shot-object-detection",
        "audio-classification",
        "automatic-speech-recognition",
    ],
)
def test_all_task_names_resolvable(task_name: str):
    task = create_task("transformers", task_name, "stub/model")
    assert task.model_name == "stub/model"
