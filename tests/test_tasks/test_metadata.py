# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for get_metadata() on task classes and related protocol models."""
from __future__ import annotations

import pytest

from task_inference.protocol.v2 import (
    Datatype,
    MetadataTensor,
    ModelMetadataResponse,
    ParameterMetadata,
)
from task_inference.tasks.audio.audio_classification import (
    AudioClassificationInput,
    AudioClassificationOutput,
    AudioClassificationTask,
)
from task_inference.tasks.audio.speech_recognition import ASRInput, ASROutput, ASRTask
from task_inference.tasks.vision.depth_estimation import (
    DepthEstimationInput,
    DepthEstimationOutput,
    DepthEstimationTask,
)
from task_inference.tasks.vision.image_anonymization import (
    ImageAnonymizationInput,
    ImageAnonymizationOutput,
    ImageAnonymizationTask,
)
from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
    ImageClassificationTask,
)
from task_inference.tasks.vision.image_segmentation import (
    ImageSegmentationInput,
    ImageSegmentationOutput,
    ImageSegmentationTask,
)
from task_inference.tasks.vision.mask_generation import (
    MaskGenerationInput,
    MaskGenerationOutput,
    MaskGenerationTask,
)
from task_inference.tasks.vision.object_detection import (
    ObjectDetectionInput,
    ObjectDetectionOutput,
    ObjectDetectionTask,
)
from task_inference.tasks.vision.visual_question_answering import (
    VQAInput,
    VQAOutput,
    VQATask,
)
from task_inference.tasks.vision.zero_shot_image_classification import (
    ZeroShotImageClassificationInput,
    ZeroShotImageClassificationOutput,
    ZeroShotImageClassificationTask,
)
from task_inference.tasks.vision.zero_shot_object_detection import (
    ZeroShotObjectDetectionInput,
    ZeroShotObjectDetectionOutput,
    ZeroShotObjectDetectionTask,
)

# ---------------------------------------------------------------------------
# Minimal concrete stubs (process() is not under test here)
# ---------------------------------------------------------------------------


class _AudioClass(AudioClassificationTask):
    def process(self, inputs: AudioClassificationInput) -> AudioClassificationOutput:
        raise NotImplementedError


class _ASR(ASRTask):
    def process(self, inputs: ASRInput) -> ASROutput:
        raise NotImplementedError


class _Depth(DepthEstimationTask):
    def process(self, inputs: DepthEstimationInput) -> DepthEstimationOutput:
        raise NotImplementedError


class _ImageAnon(ImageAnonymizationTask):
    def process(self, inputs: ImageAnonymizationInput) -> ImageAnonymizationOutput:
        raise NotImplementedError


class _ImageClass(ImageClassificationTask):
    def process(self, inputs: ImageClassificationInput) -> ImageClassificationOutput:
        raise NotImplementedError


class _Segmentation(ImageSegmentationTask):
    def process(self, inputs: ImageSegmentationInput) -> ImageSegmentationOutput:
        raise NotImplementedError


class _MaskGen(MaskGenerationTask):
    def process(self, inputs: MaskGenerationInput) -> MaskGenerationOutput:
        raise NotImplementedError


class _ObjDet(ObjectDetectionTask):
    def process(self, inputs: ObjectDetectionInput) -> ObjectDetectionOutput:
        raise NotImplementedError


class _VQA(VQATask):
    def process(self, inputs: VQAInput) -> VQAOutput:
        raise NotImplementedError


class _ZSIC(ZeroShotImageClassificationTask):
    def process(self, inputs: ZeroShotImageClassificationInput) -> ZeroShotImageClassificationOutput:
        raise NotImplementedError


class _ZSOD(ZeroShotObjectDetectionTask):
    def process(self, inputs: ZeroShotObjectDetectionInput) -> ZeroShotObjectDetectionOutput:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# MetadataTensor model
# ---------------------------------------------------------------------------


def test_metadata_tensor_bytes():
    t = MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1])
    assert t.name == "image"
    assert t.datatype == Datatype.BYTES
    assert t.shape == [-1]


def test_metadata_tensor_fp32_multidim():
    t = MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1])
    assert t.datatype == Datatype.FP32
    assert t.shape == [-1, -1]


def test_metadata_tensor_int32():
    t = MetadataTensor(name="count", datatype=Datatype.INT32, shape=[-1])
    assert t.datatype == Datatype.INT32


# ---------------------------------------------------------------------------
# ParameterMetadata model
# ---------------------------------------------------------------------------


def test_parameter_metadata_required():
    p = ParameterMetadata(name="sample_rate", datatype="int64", required=True)
    assert p.name == "sample_rate"
    assert p.datatype == "int64"
    assert p.required is True
    assert p.default is None
    assert p.description is None


def test_parameter_metadata_optional_with_default():
    p = ParameterMetadata(
        name="top_k",
        datatype="int64",
        required=False,
        default=5,
        description="Number of top predictions",
    )
    assert p.required is False
    assert p.default == 5
    assert p.description == "Number of top predictions"


def test_parameter_metadata_string_type():
    p = ParameterMetadata(name="language", datatype="string", required=False)
    assert p.datatype == "string"
    assert p.default is None


def test_parameter_metadata_bool_type():
    p = ParameterMetadata(name="return_timestamps", datatype="bool", default=False)
    assert p.datatype == "bool"
    assert p.default is False


# ---------------------------------------------------------------------------
# ModelMetadataResponse model
# ---------------------------------------------------------------------------


def test_model_metadata_response_minimal():
    resp = ModelMetadataResponse(
        name="my-model",
        inputs=[MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1])],
        outputs=[MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1])],
    )
    assert resp.name == "my-model"
    assert resp.platform == ""
    assert resp.versions is None
    assert resp.parameters is None


def test_model_metadata_response_all_fields():
    resp = ModelMetadataResponse(
        name="my-model",
        versions=["1.0", "2.0"],
        platform="transformers",
        inputs=[MetadataTensor(name="image", datatype=Datatype.BYTES, shape=[-1])],
        outputs=[MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1])],
        parameters=[ParameterMetadata(name="top_k", datatype="int64", default=5)],
    )
    assert resp.versions == ["1.0", "2.0"]
    assert resp.platform == "transformers"
    assert len(resp.parameters) == 1
    assert resp.parameters[0].name == "top_k"


def test_model_metadata_response_json_roundtrip():
    resp = ModelMetadataResponse(
        name="audio-classification",
        inputs=[MetadataTensor(name="audio", datatype=Datatype.BYTES, shape=[-1])],
        outputs=[MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1])],
        parameters=[ParameterMetadata(name="top_k", datatype="int64", default=5)],
    )
    restored = ModelMetadataResponse.model_validate_json(resp.model_dump_json())
    assert restored.name == resp.name
    assert restored.inputs[0].name == "audio"
    assert restored.parameters[0].default == 5


# ---------------------------------------------------------------------------
# get_metadata() — AudioClassificationTask
# ---------------------------------------------------------------------------


def test_audio_classification_metadata_name_default():
    meta = _AudioClass().get_metadata()
    assert isinstance(meta, ModelMetadataResponse)
    assert meta.name == "audio-classification"


def test_audio_classification_metadata_custom_name():
    meta = _AudioClass().get_metadata(model_name="my-audio-model")
    assert meta.name == "my-audio-model"


def test_audio_classification_metadata_platform_versions():
    meta = _AudioClass().get_metadata(platform="pt", versions=["1.0", "2.0"])
    assert meta.platform == "pt"
    assert meta.versions == ["1.0", "2.0"]


def test_audio_classification_metadata_inputs():
    meta = _AudioClass().get_metadata()
    assert len(meta.inputs) == 1
    inp = meta.inputs[0]
    assert inp.name == "audio"
    assert inp.datatype == Datatype.BYTES
    assert inp.shape == [-1]


def test_audio_classification_metadata_outputs():
    meta = _AudioClass().get_metadata()
    out_names = [o.name for o in meta.outputs]
    assert "label" in out_names
    assert "score" in out_names
    score = next(o for o in meta.outputs if o.name == "score")
    assert score.datatype == Datatype.FP32


def test_audio_classification_metadata_parameters():
    meta = _AudioClass().get_metadata()
    assert meta.parameters is not None
    param_names = [p.name for p in meta.parameters]
    assert "sample_rate" in param_names
    assert "top_k" in param_names

    sample_rate = next(p for p in meta.parameters if p.name == "sample_rate")
    assert sample_rate.required is True

    top_k = next(p for p in meta.parameters if p.name == "top_k")
    assert top_k.required is False
    assert top_k.default == 5


# ---------------------------------------------------------------------------
# get_metadata() — ASRTask
# ---------------------------------------------------------------------------


def test_asr_metadata_name():
    meta = _ASR().get_metadata()
    assert meta.name == "automatic-speech-recognition"


def test_asr_metadata_inputs():
    meta = _ASR().get_metadata()
    assert len(meta.inputs) == 1
    assert meta.inputs[0].name == "audio"


def test_asr_metadata_outputs():
    meta = _ASR().get_metadata()
    out_names = [o.name for o in meta.outputs]
    assert "text" in out_names
    assert "chunks_texts" in out_names
    assert "chunks_ts_start" in out_names
    assert "chunks_ts_end" in out_names


def test_asr_metadata_parameters():
    meta = _ASR().get_metadata()
    assert meta.parameters is not None
    param_names = [p.name for p in meta.parameters]
    assert "sample_rate" in param_names
    assert "return_timestamps" in param_names
    assert "language" in param_names

    return_ts = next(p for p in meta.parameters if p.name == "return_timestamps")
    assert return_ts.default is False


# ---------------------------------------------------------------------------
# get_metadata() — tasks without parameters (no METADATA_PARAMETERS set)
# ---------------------------------------------------------------------------


def test_depth_estimation_metadata_no_parameters():
    meta = _Depth().get_metadata()
    assert meta.name == "depth-estimation"
    assert meta.parameters is None


def test_depth_estimation_metadata_outputs():
    meta = _Depth().get_metadata()
    out_names = [o.name for o in meta.outputs]
    assert "predicted_depth" in out_names
    assert "depth" in out_names

    pd = next(o for o in meta.outputs if o.name == "predicted_depth")
    assert pd.datatype == Datatype.FP32
    assert len(pd.shape) == 3


def test_zero_shot_image_classification_metadata_no_parameters():
    meta = _ZSIC().get_metadata()
    assert meta.name == "zero-shot-image-classification"
    assert meta.parameters is None


def test_zero_shot_image_classification_metadata_inputs():
    meta = _ZSIC().get_metadata()
    in_names = [i.name for i in meta.inputs]
    assert "image" in in_names
    assert "candidate_labels" in in_names


# ---------------------------------------------------------------------------
# get_metadata() — ImageClassificationTask
# ---------------------------------------------------------------------------


def test_image_classification_metadata():
    meta = _ImageClass().get_metadata()
    assert meta.name == "image-classification"
    assert len(meta.inputs) == 1
    assert meta.inputs[0].name == "image"
    out_names = [o.name for o in meta.outputs]
    assert "label" in out_names and "score" in out_names
    assert meta.parameters is not None
    assert any(p.name == "top_k" for p in meta.parameters)


# ---------------------------------------------------------------------------
# get_metadata() — ObjectDetectionTask
# ---------------------------------------------------------------------------


def test_object_detection_metadata():
    meta = _ObjDet().get_metadata()
    assert meta.name == "object-detection"
    out_names = [o.name for o in meta.outputs]
    assert "box" in out_names
    box = next(o for o in meta.outputs if o.name == "box")
    assert box.datatype == Datatype.FP32
    assert box.shape == [-1, -1, 4]

    assert meta.parameters is not None
    threshold = next(p for p in meta.parameters if p.name == "threshold")
    assert threshold.datatype == "fp64"
    assert threshold.default == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_metadata() — ImageSegmentationTask
# ---------------------------------------------------------------------------


def test_image_segmentation_metadata():
    meta = _Segmentation().get_metadata()
    assert meta.name == "image-segmentation"
    out_names = [o.name for o in meta.outputs]
    assert "mask" in out_names
    assert meta.parameters is not None
    param_names = [p.name for p in meta.parameters]
    assert "threshold" in param_names
    assert "mask_threshold" in param_names
    assert "overlap_mask_area_threshold" in param_names
    assert "subtask" in param_names


# ---------------------------------------------------------------------------
# get_metadata() — MaskGenerationTask
# ---------------------------------------------------------------------------


def test_mask_generation_metadata():
    meta = _MaskGen().get_metadata()
    assert meta.name == "mask-generation"
    out_names = [o.name for o in meta.outputs]
    assert "mask" in out_names
    assert "score" in out_names
    assert meta.parameters is not None
    param_names = [p.name for p in meta.parameters]
    assert "mask_threshold" in param_names
    assert "pred_iou_thresh" in param_names
    assert "crops_n_layers" in param_names


# ---------------------------------------------------------------------------
# get_metadata() — VQATask
# ---------------------------------------------------------------------------


def test_vqa_metadata():
    meta = _VQA().get_metadata()
    assert meta.name == "visual-question-answering"
    in_names = [i.name for i in meta.inputs]
    assert "image" in in_names
    assert "question" in in_names
    assert meta.parameters is not None
    assert any(p.name == "top_k" for p in meta.parameters)


# ---------------------------------------------------------------------------
# get_metadata() — ZeroShotObjectDetectionTask
# ---------------------------------------------------------------------------


def test_zero_shot_object_detection_metadata():
    meta = _ZSOD().get_metadata()
    assert meta.name == "zero-shot-object-detection"
    in_names = [i.name for i in meta.inputs]
    assert "image" in in_names
    assert "candidate_labels" in in_names
    out_names = [o.name for o in meta.outputs]
    assert "box" in out_names
    assert meta.parameters is not None
    threshold = next(p for p in meta.parameters if p.name == "threshold")
    assert threshold.default == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# get_metadata() — ImageAnonymizationTask
# ---------------------------------------------------------------------------


def test_image_anonymization_metadata():
    meta = _ImageAnon().get_metadata()
    assert meta.name == "image-anonymization"
    assert len(meta.inputs) == 1
    assert meta.inputs[0].name == "image"
    out_names = [o.name for o in meta.outputs]
    assert "anonymized_image" in out_names
    assert "num_regions_anonymized" in out_names

    num_regions = next(o for o in meta.outputs if o.name == "num_regions_anonymized")
    assert num_regions.datatype == Datatype.INT32

    assert meta.parameters is not None
    param_names = [p.name for p in meta.parameters]
    assert "strategy" in param_names
    assert "blur_radius" in param_names
    assert "threshold" in param_names
    assert "classes" in param_names

    strategy = next(p for p in meta.parameters if p.name == "strategy")
    assert strategy.default == "blur"
