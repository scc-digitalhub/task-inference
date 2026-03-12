# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Transformers backend task implementations.

All HuggingFace ``pipeline`` calls are replaced with ``MagicMock`` objects so
no network access, GPU, or model download is required.  The whole module is
skipped automatically when ``transformers`` is not installed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ----------------------------------------------------------------------- #
# Guard -- skip entire module if transformers is not installed             #
# ----------------------------------------------------------------------- #
pytest.importorskip("transformers", reason="transformers not installed")

# Vision implementations
from task_inference.implementations.transformers.vision.depth_estimation import (  # noqa: E402
    TransformersDepthEstimationTask,
)
from task_inference.implementations.transformers.vision.image_anonymization import (  # noqa: E402
    TransformersImageAnonymizationTask,
)
from task_inference.implementations.transformers.vision.image_classification import (  # noqa: E402
    TransformersImageClassificationTask,
)
from task_inference.implementations.transformers.vision.image_segmentation import (  # noqa: E402
    TransformersImageSegmentationTask,
)
from task_inference.implementations.transformers.vision.mask_generation import (  # noqa: E402
    TransformersMaskGenerationTask,
)
from task_inference.implementations.transformers.vision.object_detection import (  # noqa: E402
    TransformersObjectDetectionTask,
)
from task_inference.implementations.transformers.vision.visual_question_answering import (  # noqa: E402
    TransformersVQATask,
)
from task_inference.implementations.transformers.vision.zero_shot_image_classification import (  # noqa: E402
    TransformersZeroShotImageClassificationTask,
)
from task_inference.implementations.transformers.vision.zero_shot_object_detection import (  # noqa: E402
    TransformersZeroShotObjectDetectionTask,
)

# Audio implementations
from task_inference.implementations.transformers.audio.audio_classification import (  # noqa: E402
    TransformersAudioClassificationTask,
)
from task_inference.implementations.transformers.audio.speech_recognition import (  # noqa: E402
    TransformersASRTask,
)

# Task input schemas
from task_inference.tasks.audio.audio_classification import AudioClassificationInput  # noqa: E402
from task_inference.tasks.audio.speech_recognition import ASRInput  # noqa: E402
from task_inference.tasks.vision.depth_estimation import DepthEstimationInput  # noqa: E402
from task_inference.tasks.vision.image_anonymization import (  # noqa: E402
    AnonymizationStrategy,
    ImageAnonymizationInput,
)
from task_inference.tasks.vision.image_classification import ImageClassificationInput  # noqa: E402
from task_inference.tasks.vision.image_segmentation import ImageSegmentationInput  # noqa: E402
from task_inference.tasks.vision.mask_generation import MaskGenerationInput, Point  # noqa: E402
from task_inference.tasks.vision.object_detection import ObjectDetectionInput  # noqa: E402
from task_inference.tasks.vision.visual_question_answering import VQAInput  # noqa: E402
from task_inference.tasks.vision.zero_shot_image_classification import ZeroShotImageClassificationInput  # noqa: E402
from task_inference.tasks.vision.zero_shot_object_detection import ZeroShotObjectDetectionInput  # noqa: E402

# ---------------------------------------------------------------------------
# Helper: build a task instance with mocked pipeline
# ---------------------------------------------------------------------------

_MOD_BASE = "task_inference.implementations.transformers"


def _make_task(cls, pipe_output, submodule: str, **init_kwargs):
    """Return a task instance backed by a MagicMock pipeline.

    Parameters
    ----------
    cls:
        Task class to instantiate.
    pipe_output:
        Value the mocked pipeline callable should return.
    submodule:
        Dotted submodule path relative to ``_MOD_BASE`` (e.g. ``vision.image_classification``).
    **init_kwargs:
        Extra keyword arguments forwarded to the constructor.
    """
    mock_pipe = MagicMock(return_value=pipe_output)
    mock_pipeline = MagicMock(return_value=mock_pipe)
    full_path = f"{_MOD_BASE}.{submodule}.pipeline"
    with patch(full_path, mock_pipeline):
        task = cls("stub/model", **init_kwargs)
    return task, mock_pipe


# ===========================================================================
# Vision tasks
# ===========================================================================


class TestImageClassification:
    def test_init_stores_model_name(self, sample_image_bytes):
        task, _ = _make_task(
            TransformersImageClassificationTask,
            [[{"label": "cat", "score": 0.9}]],
            "vision.image_classification",
        )
        assert task.model_name == "stub/model"

    def test_process_returns_correct_output(self, sample_image_bytes):
        # Pipeline returns list-of-lists (batch of images, each with a list of results)
        task, _ = _make_task(
            TransformersImageClassificationTask,
            [[{"label": "cat", "score": 0.9}, {"label": "dog", "score": 0.1}]],
            "vision.image_classification",
        )
        result = task.process(ImageClassificationInput(images=[sample_image_bytes], top_k=2))
        assert len(result.results) == 1          # 1 image
        assert len(result.results[0]) == 2       # 2 predictions
        assert result.results[0][0].label == "cat"
        assert abs(result.results[0][0].score - 0.9) < 1e-6


class TestObjectDetection:
    def test_process_returns_detections(self, sample_image_bytes):
        pipe_output = [
            [
                {"label": "person", "score": 0.95,
                 "box": {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200}},
            ]
        ]
        task, _ = _make_task(
            TransformersObjectDetectionTask,
            pipe_output,
            "vision.object_detection",
        )
        result = task.process(ObjectDetectionInput(images=[sample_image_bytes]))
        assert len(result.detections) == 1       # 1 image
        assert len(result.detections[0]) == 1    # 1 detection
        assert result.detections[0][0].label == "person"
        assert result.detections[0][0].box.xmin == 10.0
        assert result.detections[0][0].box.ymax == 200.0

    def test_process_empty_detections(self, sample_image_bytes):
        task, _ = _make_task(
            TransformersObjectDetectionTask,
            [[]],
            "vision.object_detection",
        )
        result = task.process(ObjectDetectionInput(images=[sample_image_bytes]))
        assert result.detections == [[]]


class TestDepthEstimation:
    def test_process_with_numpy_predicted_depth(self, sample_image_bytes):
        """Pipeline called once with all PIL images; result is a list of dicts."""
        depth_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pipe_output = [{"predicted_depth": depth_array, "depth": None}]
        task, _ = _make_task(
            TransformersDepthEstimationTask,
            pipe_output,
            "vision.depth_estimation",
        )
        result = task.process(DepthEstimationInput(images=[sample_image_bytes]))
        assert result.height == 2
        assert result.width == 2
        assert result.predicted_depth == [[1.0, 2.0, 3.0, 4.0]]
        assert result.depth is None

    def test_process_with_depth_visualisation(self, sample_image_bytes):
        depth_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        depth_vis = Image.new("L", (2, 2), color=128)
        pipe_output = [{"predicted_depth": depth_array, "depth": depth_vis}]
        task, _ = _make_task(
            TransformersDepthEstimationTask,
            pipe_output,
            "vision.depth_estimation",
        )
        result = task.process(DepthEstimationInput(images=[sample_image_bytes]))
        assert result.depth is not None
        assert isinstance(result.depth[0], bytes)

    def test_process_fallback_to_depth_field(self, sample_image_bytes):
        """When predicted_depth is None the implementation reads the depth PIL image."""
        depth_vis = Image.fromarray(
            np.array([[10, 20], [30, 40]], dtype=np.uint8), mode="L"
        )
        pipe_output = [{"predicted_depth": None, "depth": depth_vis}]
        task, _ = _make_task(
            TransformersDepthEstimationTask,
            pipe_output,
            "vision.depth_estimation",
        )
        result = task.process(DepthEstimationInput(images=[sample_image_bytes]))
        assert result.height == 2
        assert result.width == 2
        assert result.depth is not None

    def test_process_batch_single_pipeline_call(self, sample_image_bytes):
        """Pipeline must be called exactly once regardless of batch size."""
        depth_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        pipe_output = [
            {"predicted_depth": depth_array, "depth": None},
            {"predicted_depth": depth_array, "depth": None},
        ]
        task, mock_pipe = _make_task(
            TransformersDepthEstimationTask,
            pipe_output,
            "vision.depth_estimation",
        )
        result = task.process(
            DepthEstimationInput(images=[sample_image_bytes, sample_image_bytes])
        )
        mock_pipe.assert_called_once()
        assert len(result.predicted_depth) == 2


class TestImageSegmentation:
    def _make_mask(self) -> Image.Image:
        return Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255, mode="L")

    def test_process_returns_segments(self, sample_image_bytes):
        # Batch pipeline returns list-of-lists (one per image)
        pipe_output = [[{"label": "sky", "score": 0.9, "mask": self._make_mask()}]]
        task, _ = _make_task(
            TransformersImageSegmentationTask,
            pipe_output,
            "vision.image_segmentation",
        )
        result = task.process(ImageSegmentationInput(images=[sample_image_bytes]))
        assert len(result.segments) == 1         # 1 image
        assert len(result.segments[0]) == 1      # 1 segment
        assert result.segments[0][0].label == "sky"
        assert isinstance(result.segments[0][0].mask, bytes)

    def test_process_with_subtask(self, sample_image_bytes):
        pipe_output = [[{"label": "car", "score": 0.8, "mask": self._make_mask()}]]
        task, mock_pipe = _make_task(
            TransformersImageSegmentationTask,
            pipe_output,
            "vision.image_segmentation",
        )
        inputs = ImageSegmentationInput(images=[sample_image_bytes], subtask="instance")
        task.process(inputs)
        _, call_kwargs = mock_pipe.call_args
        assert call_kwargs.get("subtask") == "instance"


class TestImageAnonymization:
    _BOX = {"xmin": 5, "ymin": 5, "xmax": 45, "ymax": 45}

    def _make_task_with_detections(self, detections):
        # Batch pipeline returns a list-of-lists (one per image)
        return _make_task(
            TransformersImageAnonymizationTask,
            [detections],
            "vision.image_anonymization",
        )

    def test_blur_strategy(self, sample_image_bytes):
        task, _ = self._make_task_with_detections(
            [{"label": "person", "score": 0.9, "box": self._BOX}]
        )
        result = task.process(ImageAnonymizationInput(images=[sample_image_bytes]))
        assert result.num_regions_anonymized == [1]
        assert isinstance(result.images[0], bytes)

    def test_pixelate_strategy(self, sample_image_bytes):
        task, _ = self._make_task_with_detections(
            [{"label": "person", "score": 0.9, "box": self._BOX}]
        )
        inputs = ImageAnonymizationInput(
            images=[sample_image_bytes], strategy=AnonymizationStrategy.PIXELATE
        )
        result = task.process(inputs)
        assert result.num_regions_anonymized == [1]

    def test_black_box_strategy(self, sample_image_bytes):
        task, _ = self._make_task_with_detections(
            [{"label": "face", "score": 0.85, "box": self._BOX}]
        )
        inputs = ImageAnonymizationInput(
            images=[sample_image_bytes],
            strategy=AnonymizationStrategy.BLACK_BOX,
            classes=["face"],
        )
        result = task.process(inputs)
        assert result.num_regions_anonymized == [1]

    def test_classes_filter(self, sample_image_bytes):
        """Detections whose label is not in classes are skipped."""
        task, _ = self._make_task_with_detections(
            [{"label": "car", "score": 0.9, "box": self._BOX}]
        )
        inputs = ImageAnonymizationInput(
            images=[sample_image_bytes], classes=["person"]
        )
        result = task.process(inputs)
        assert result.num_regions_anonymized == [0]

    def test_no_classes_anon_all(self, sample_image_bytes):
        """When classes is None all detections are anonymized."""
        task, _ = self._make_task_with_detections(
            [
                {"label": "car", "score": 0.9, "box": self._BOX},
                {"label": "person", "score": 0.8, "box": {"xmin": 10, "ymin": 10, "xmax": 20, "ymax": 20}},
            ]
        )
        result = task.process(ImageAnonymizationInput(images=[sample_image_bytes]))
        assert result.num_regions_anonymized == [2]


class TestMaskGeneration:
    def test_process_numpy_mask(self, sample_image_bytes):
        mask_np = np.ones((4, 4), dtype=np.float32)
        pipe_output = {"masks": [mask_np], "scores": [0.95]}
        task, _ = _make_task(
            TransformersMaskGenerationTask,
            pipe_output,
            "vision.mask_generation",
            points_per_batch=32,
        )
        result = task.process(MaskGenerationInput(images=[sample_image_bytes]))
        assert len(result.masks) == 1           # 1 image
        assert len(result.masks[0]) == 1        # 1 mask
        assert abs(result.masks[0][0].score - 0.95) < 1e-6
        assert isinstance(result.masks[0][0].mask, bytes)

    def test_process_pil_mask(self, sample_image_bytes):
        pil_mask = Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255, mode="L")
        pipe_output = {"masks": [pil_mask], "scores": [0.8]}
        task, _ = _make_task(
            TransformersMaskGenerationTask,
            pipe_output,
            "vision.mask_generation",
        )
        result = task.process(MaskGenerationInput(images=[sample_image_bytes]))
        assert len(result.masks) == 1
        assert len(result.masks[0]) == 1

    def test_process_other_mask_type(self, sample_image_bytes):
        """Masks that are neither ndarray nor PIL.Image go through the else branch."""
        other_mask = [[True, False], [False, True]]
        pipe_output = {"masks": [other_mask], "scores": [0.7]}
        task, _ = _make_task(
            TransformersMaskGenerationTask,
            pipe_output,
            "vision.mask_generation",
        )
        result = task.process(MaskGenerationInput(images=[sample_image_bytes]))
        assert len(result.masks) == 1
        assert len(result.masks[0]) == 1

    def test_process_empty_masks(self, sample_image_bytes):
        task, _ = _make_task(
            TransformersMaskGenerationTask,
            {"masks": [], "scores": []},
            "vision.mask_generation",
        )
        result = task.process(MaskGenerationInput(images=[sample_image_bytes]))
        assert result.masks == [[]]

    def test_process_batch_loops_per_image(self, sample_image_bytes):
        """SAM is called once per image; batch of 2 images = 2 pipe calls."""
        pil_mask = Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255, mode="L")
        pipe_output = {"masks": [pil_mask], "scores": [0.9]}
        task, mock_pipe = _make_task(
            TransformersMaskGenerationTask,
            pipe_output,
            "vision.mask_generation",
        )
        result = task.process(
            MaskGenerationInput(images=[sample_image_bytes, sample_image_bytes])
        )
        assert mock_pipe.call_count == 2
        assert len(result.masks) == 2


class TestVQA:
    def test_process_returns_answers(self, sample_image_bytes):
        pipe_output = [{"answer": "yes", "score": 0.8}]
        task, _ = _make_task(
            TransformersVQATask,
            pipe_output,
            "vision.visual_question_answering",
        )
        result = task.process(VQAInput(images=[sample_image_bytes], question="Is there a cat?"))
        assert len(result.answers) == 1          # 1 image
        assert len(result.answers[0]) == 1       # 1 answer candidate
        assert result.answers[0][0].answer == "yes"
        assert abs(result.answers[0][0].score - 0.8) < 1e-6

    def test_process_multiple_answers(self, sample_image_bytes):
        pipe_output = [
            {"answer": "yes", "score": 0.7},
            {"answer": "no", "score": 0.3},
        ]
        task, _ = _make_task(
            TransformersVQATask,
            pipe_output,
            "vision.visual_question_answering",
        )
        result = task.process(VQAInput(images=[sample_image_bytes], question="red?", top_k=2))
        assert len(result.answers) == 1
        assert len(result.answers[0]) == 2

    def test_process_batch_loops_per_image(self, sample_image_bytes):
        """VQA pipeline is called once per image; batch of 2 = 2 pipe calls."""
        pipe_output = [{"answer": "yes", "score": 0.8}]
        task, mock_pipe = _make_task(
            TransformersVQATask,
            pipe_output,
            "vision.visual_question_answering",
        )
        result = task.process(
            VQAInput(images=[sample_image_bytes, sample_image_bytes], question="red?")
        )
        assert mock_pipe.call_count == 2
        assert len(result.answers) == 2


class TestZeroShotImageClassification:
    def test_process_returns_results(self, sample_image_bytes):
        pipe_output = [{"label": "cat", "score": 0.9}, {"label": "dog", "score": 0.1}]
        task, _ = _make_task(
            TransformersZeroShotImageClassificationTask,
            pipe_output,
            "vision.zero_shot_image_classification",
        )
        result = task.process(
            ZeroShotImageClassificationInput(
                images=[sample_image_bytes], candidate_labels=["cat", "dog"]
            )
        )
        assert len(result.results) == 1
        assert result.results[0][0].label == "cat"
        assert abs(result.results[0][0].score - 0.9) < 1e-6

    def test_process_batch_loops_per_image(self, sample_image_bytes):
        """zero-shot-image-classification pipeline is called once per image."""
        pipe_output = [{"label": "cat", "score": 0.85}]
        task, mock_pipe = _make_task(
            TransformersZeroShotImageClassificationTask,
            pipe_output,
            "vision.zero_shot_image_classification",
        )
        result = task.process(
            ZeroShotImageClassificationInput(
                images=[sample_image_bytes, sample_image_bytes], candidate_labels=["cat"]
            )
        )
        assert mock_pipe.call_count == 2
        assert len(result.results) == 2


class TestZeroShotObjectDetection:
    def test_process_returns_detections(self, sample_image_bytes):
        pipe_output = [
            {"label": "cat", "score": 0.8, "box": {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200}}
        ]
        task, _ = _make_task(
            TransformersZeroShotObjectDetectionTask,
            pipe_output,
            "vision.zero_shot_object_detection",
        )
        result = task.process(
            ZeroShotObjectDetectionInput(
                images=[sample_image_bytes], candidate_labels=["cat", "dog"]
            )
        )
        assert len(result.detections) == 1
        assert len(result.detections[0]) == 1
        det = result.detections[0][0]
        assert det.label == "cat"
        assert abs(det.score - 0.8) < 1e-6
        assert det.box.xmin == 10.0
        assert det.box.ymax == 200.0

    def test_process_empty_detections(self, sample_image_bytes):
        task, _ = _make_task(
            TransformersZeroShotObjectDetectionTask,
            [],
            "vision.zero_shot_object_detection",
        )
        result = task.process(
            ZeroShotObjectDetectionInput(
                images=[sample_image_bytes], candidate_labels=["cat"]
            )
        )
        assert result.detections == [[]]

    def test_process_batch_loops_per_image(self, sample_image_bytes):
        """zero-shot-object-detection pipeline is called once per image."""
        pipe_output = [{"label": "dog", "score": 0.7, "box": {"xmin": 5, "ymin": 5, "xmax": 50, "ymax": 50}}]
        task, mock_pipe = _make_task(
            TransformersZeroShotObjectDetectionTask,
            pipe_output,
            "vision.zero_shot_object_detection",
        )
        result = task.process(
            ZeroShotObjectDetectionInput(
                images=[sample_image_bytes, sample_image_bytes], candidate_labels=["dog"]
            )
        )
        assert mock_pipe.call_count == 2
        assert len(result.detections) == 2


# ===========================================================================
# Audio tasks
# ===========================================================================


class TestAudioClassification:
    def test_process_returns_results(self, sample_audio_bytes):
        pipe_output = [[{"label": "speech", "score": 0.9}]]
        task, _ = _make_task(
            TransformersAudioClassificationTask,
            pipe_output,
            "audio.audio_classification",
        )
        result = task.process(AudioClassificationInput(audio=[sample_audio_bytes], sample_rate=16000, top_k=1))
        assert len(result.results) == 1
        assert result.results[0][0].label == "speech"
        assert abs(result.results[0][0].score - 0.9) < 1e-6

    def test_process_with_sample_rate(self, sample_audio_bytes):
        task, _ = _make_task(
            TransformersAudioClassificationTask,
            [[{"label": "music", "score": 0.6}]],

            "audio.audio_classification",
        )
        result = task.process(
            AudioClassificationInput(audio=[sample_audio_bytes], sample_rate=16000, top_k=1)
        )
        assert result.results[0][0].label == "music"


class TestASR:
    def test_process_basic_transcription(self, sample_audio_bytes):
        pipe_output = [{"text": "hello world", "chunks": []}]
        task, _ = _make_task(
            TransformersASRTask,
            pipe_output,
            "audio.speech_recognition",
            chunk_length_s=15,
        )
        result = task.process(ASRInput(audio=[sample_audio_bytes], sample_rate=16000))
        assert result.texts == ["hello world"]
        assert result.chunks is None

    def test_process_with_timestamps(self, sample_audio_bytes):
        pipe_output = [
            {
                "text": "hello world",
                "chunks": [
                    {"text": "hello", "timestamp": [0.0, 0.4]},
                    {"text": "world", "timestamp": [0.5, 1.0]},
                ],
            }
        ]
        task, _ = _make_task(
            TransformersASRTask,
            pipe_output,
            "audio.speech_recognition",
        )
        result = task.process(
            ASRInput(audio=[sample_audio_bytes], sample_rate=16000, return_timestamps=True)
        )
        assert result.texts == ["hello world"]
        assert len(result.chunks) == 1
        assert len(result.chunks[0]) == 2
        assert result.chunks[0][0].timestamp_start == 0.0
        assert result.chunks[0][1].timestamp_end == 1.0

    def test_process_with_language_hint(self, sample_audio_bytes):
        pipe_output = [{"text": "bonjour", "chunks": []}]
        task, mock_pipe = _make_task(
            TransformersASRTask,
            pipe_output,
            "audio.speech_recognition",
        )
        result = task.process(
            ASRInput(audio=[sample_audio_bytes], sample_rate=16000, language="fr", return_timestamps=False)
        )
        assert result.texts == ["bonjour"]
        _, call_kwargs = mock_pipe.call_args
        assert call_kwargs.get("generate_kwargs") == {"language": "fr"}

    def test_process_timestamp_end_none(self, sample_audio_bytes):
        """timestamp_end should be None when the second element of the tuple is None."""
        pipe_output = [
            {
                "text": "hi",
                "chunks": [{"text": "hi", "timestamp": [0.0, None]}],
            }
        ]
        task, _ = _make_task(
            TransformersASRTask,
            pipe_output,
            "audio.speech_recognition",
        )
        result = task.process(ASRInput(audio=[sample_audio_bytes], sample_rate=16000, return_timestamps=True))
        assert result.chunks[0][0].timestamp_end is None

    def test_chunk_length_zero_disables_chunking(self):
        """chunk_length_s=0 should pass None to the pipeline constructor."""
        mock_pipe = MagicMock(return_value={"text": "", "chunks": []})
        mock_pipeline = MagicMock(return_value=mock_pipe)
        full_path = f"{_MOD_BASE}.audio.speech_recognition.pipeline"
        with patch(full_path, mock_pipeline):
            TransformersASRTask("stub/model", chunk_length_s=0)
        _, call_kwargs = mock_pipeline.call_args
        assert call_kwargs.get("chunk_length_s") is None

