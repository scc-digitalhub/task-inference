# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for vision task schema and conversion methods (no model required)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from task_inference.protocol.v2 import Datatype
from task_inference.tasks.vision.depth_estimation import (
    DepthEstimationInput,
    DepthEstimationOutput,
    DepthEstimationTask,
)
from task_inference.tasks.vision.image_anonymization import (
    AnonymizationStrategy,
    ImageAnonymizationInput,
    ImageAnonymizationOutput,
    ImageAnonymizationTask,
)
from task_inference.tasks.vision.image_classification import (
    ClassificationResult,
    ImageClassificationInput,
    ImageClassificationOutput,
    ImageClassificationTask,
)
from task_inference.tasks.vision.image_segmentation import (
    ImageSegmentationInput,
    ImageSegmentationOutput,
    ImageSegmentationTask,
    SegmentResult,
)
from task_inference.tasks.vision.mask_generation import (
    GeneratedMask,
    MaskGenerationInput,
    MaskGenerationOutput,
    MaskGenerationTask,
    Point,
)
from task_inference.tasks.vision.object_detection import (
    BoundingBox,
    DetectedObject,
    ObjectDetectionInput,
    ObjectDetectionOutput,
    ObjectDetectionTask,
)
from task_inference.tasks.vision.visual_question_answering import (
    VQAAnswer,
    VQAInput,
    VQAOutput,
    VQATask,
)
from task_inference.tasks.vision.zero_shot_image_classification import (
    ZeroShotClassificationResult,
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
# Helpers
# ---------------------------------------------------------------------------


def _make_mask_bytes() -> bytes:
    """Return a minimal 4x4 grayscale PNG suitable as a segmentation mask."""
    buf = io.BytesIO()
    Image.new("L", (4, 4), color=128).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Concrete stub tasks for testing (no actual models needed)
# ---------------------------------------------------------------------------


class _StubClassificationTask(ImageClassificationTask):
    def process(self, inputs: ImageClassificationInput) -> ImageClassificationOutput:
        return ImageClassificationOutput(
            results=[
                [
                    ClassificationResult(label="cat", score=0.9),
                    ClassificationResult(label="dog", score=0.1),
                ]
                for _ in inputs.images
            ]
        )


class _StubObjectDetectionTask(ObjectDetectionTask):
    def process(self, inputs: ObjectDetectionInput) -> ObjectDetectionOutput:
        return ObjectDetectionOutput(
            detections=[
                [
                    DetectedObject(
                        label="person",
                        score=0.95,
                        box=BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=200.0),
                    )
                ]
                for _ in inputs.images
            ]
        )


class _StubDepthTask(DepthEstimationTask):
    def process(self, inputs: DepthEstimationInput) -> DepthEstimationOutput:
        return DepthEstimationOutput(
            predicted_depth=[[1.0, 2.0, 3.0, 4.0]] * len(inputs.images),
            width=2,
            height=2,
        )


class _StubVQATask(VQATask):
    def process(self, inputs: VQAInput) -> VQAOutput:
        return VQAOutput(
            answers=[[VQAAnswer(answer="yes", score=0.8)] for _ in inputs.images]
        )


class _StubImageAnonymizationTask(ImageAnonymizationTask):
    def process(self, inputs: ImageAnonymizationInput) -> ImageAnonymizationOutput:
        return ImageAnonymizationOutput(
            images=inputs.images,
            num_regions_anonymized=[2] * len(inputs.images),
        )


class _StubImageSegmentationTask(ImageSegmentationTask):
    def process(self, inputs: ImageSegmentationInput) -> ImageSegmentationOutput:
        mask = _make_mask_bytes()
        return ImageSegmentationOutput(
            segments=[[SegmentResult(label="sky", score=0.9, mask=mask)] for _ in inputs.images]
        )
class _StubMaskGenerationTask(MaskGenerationTask):
    def process(self, inputs: MaskGenerationInput) -> MaskGenerationOutput:
        mask = _make_mask_bytes()
        return MaskGenerationOutput(
            masks=[[GeneratedMask(mask=mask, score=0.95)] for _ in inputs.images]
        )


class _StubZeroShotImageClassificationTask(ZeroShotImageClassificationTask):
    def process(self, inputs: ZeroShotImageClassificationInput) -> ZeroShotImageClassificationOutput:
        return ZeroShotImageClassificationOutput(
            results=[
                [ZeroShotClassificationResult(label=lbl, score=0.9 / (i + 1)) for i, lbl in enumerate(inputs.candidate_labels)]
                for _ in inputs.images
            ]
        )


class _StubZeroShotObjectDetectionTask(ZeroShotObjectDetectionTask):
    def process(self, inputs: ZeroShotObjectDetectionInput) -> ZeroShotObjectDetectionOutput:
        return ZeroShotObjectDetectionOutput(
            detections=[
                [
                    DetectedObject(
                        label=inputs.candidate_labels[0],
                        score=0.8,
                        box=BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=200.0),
                    )
                ]
                for _ in inputs.images
            ]
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_image_classification_build_request(sample_image_bytes):
    req = ImageClassificationInput(images=[sample_image_bytes], top_k=3).to_inference_request()
    assert len(req.inputs) == 1
    assert req.inputs[0].name == "image"
    assert req.inputs[0].datatype == Datatype.BYTES
    # encode_image returns raw bytes; data contains the original bytes directly
    assert req.inputs[0].data[0] == sample_image_bytes
    assert req.parameters == {"top_k": 3}


def test_image_classification_full_pipeline(sample_image_bytes):
    task = _StubClassificationTask()
    resp = task(ImageClassificationInput(images=[sample_image_bytes]).to_inference_request())
    result = ImageClassificationOutput.from_inference_response(resp)
    assert len(result.results) == 1         # 1 image
    assert len(result.results[0]) == 2      # 2 results for that image
    assert result.results[0][0].label == "cat"
    assert abs(result.results[0][0].score - 0.9) < 1e-6


def test_image_classification_run_helper(sample_image_bytes):
    task = _StubClassificationTask()
    resp = task.run(images=[sample_image_bytes], top_k=2)
    result = ImageClassificationOutput.from_inference_response(resp)
    assert len(result.results) == 1
    assert len(result.results[0]) == 2


def test_object_detection_build_request(sample_image_bytes):
    req = ObjectDetectionInput(images=[sample_image_bytes], threshold=0.6).to_inference_request()
    assert req.parameters == {"threshold": 0.6}


def test_object_detection_parse_response(sample_image_bytes):
    task = _StubObjectDetectionTask()
    resp = task(ObjectDetectionInput(images=[sample_image_bytes]).to_inference_request())
    result = ObjectDetectionOutput.from_inference_response(resp)
    assert len(result.detections) == 1       # 1 image
    assert len(result.detections[0]) == 1    # 1 detection
    det = result.detections[0][0]
    assert det.label == "person"
    assert det.box.xmin == 10.0
    assert det.box.ymax == 200.0


def test_depth_estimation_parse_response(sample_image_bytes):
    task = _StubDepthTask()
    resp = task(DepthEstimationInput(images=[sample_image_bytes]).to_inference_request())
    result = DepthEstimationOutput.from_inference_response(resp)
    assert result.height == 2
    assert result.width == 2
    assert result.predicted_depth == [[1.0, 2.0, 3.0, 4.0]]


def test_vqa_build_request(sample_image_bytes):
    req = VQAInput(images=[sample_image_bytes], question="Is this a cat?").to_inference_request()
    names = {t.name for t in req.inputs}
    assert "image" in names
    assert "question" in names
    q_tensor = next(t for t in req.inputs if t.name == "question")
    assert q_tensor.data == ["Is this a cat?"]


def test_vqa_parse_response(sample_image_bytes):
    task = _StubVQATask()
    resp = task(VQAInput(images=[sample_image_bytes], question="Is it red?").to_inference_request())
    result = VQAOutput.from_inference_response(resp)
    assert result.answers[0][0].answer == "yes"
    assert abs(result.answers[0][0].score - 0.8) < 1e-6


# ===========================================================================
# OIP round-trip tests: from_inference_request / to/from_inference_response
# ===========================================================================


# ---------------------------------------------------------------------------
# ImageClassification
# ---------------------------------------------------------------------------


def test_image_classification_from_inference_request(sample_image_bytes):
    original = ImageClassificationInput(images=[sample_image_bytes], top_k=3)
    req = original.to_inference_request()
    recovered = ImageClassificationInput.from_inference_request(req)
    assert recovered.images == [sample_image_bytes]
    assert recovered.top_k == 3


def test_image_classification_from_inference_request_default_top_k(sample_image_bytes):
    """from_inference_request uses top_k=5 when the parameter is absent."""
    req = ImageClassificationInput(images=[sample_image_bytes]).to_inference_request()
    # Remove top_k from parameters to test default handling
    req.parameters = {}
    recovered = ImageClassificationInput.from_inference_request(req)
    assert recovered.top_k == 5


def test_image_classification_to_inference_response(sample_image_bytes):
    output = ImageClassificationOutput(
        results=[[ClassificationResult(label="cat", score=0.9)]]
    )
    resp = output.to_inference_response("my-model")
    assert resp.model_name == "my-model"
    labels_tensor = resp.get_output("label")
    scores_tensor = resp.get_output("score")
    assert labels_tensor.data == [["cat"]]
    assert abs(scores_tensor.data[0][0] - 0.9) < 1e-6


def test_image_classification_from_inference_response(sample_image_bytes):
    output = ImageClassificationOutput(
        results=[
            [
                ClassificationResult(label="cat", score=0.9),
                ClassificationResult(label="dog", score=0.1),
            ]
        ]
    )
    resp = output.to_inference_response("m")
    recovered = ImageClassificationOutput.from_inference_response(resp)
    assert len(recovered.results) == 1
    assert len(recovered.results[0]) == 2
    assert recovered.results[0][0].label == "cat"
    assert abs(recovered.results[0][1].score - 0.1) < 1e-6


# ---------------------------------------------------------------------------
# ObjectDetection
# ---------------------------------------------------------------------------


def test_object_detection_from_inference_request(sample_image_bytes):
    original = ObjectDetectionInput(images=[sample_image_bytes], threshold=0.7)
    recovered = ObjectDetectionInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.images == [sample_image_bytes]
    assert abs(recovered.threshold - 0.7) < 1e-6


def test_object_detection_to_inference_response(sample_image_bytes):
    output = ObjectDetectionOutput(
        detections=[
            [
                DetectedObject(
                    label="car", score=0.8,
                    box=BoundingBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0),
                )
            ]
        ]
    )
    resp = output.to_inference_response("m")
    labels = resp.get_output("label").data
    boxes = resp.get_output("box").data
    # Per-image nested: labels[0] is the first image's labels
    assert labels[0] == ["car"]
    assert boxes[0] == [1.0, 2.0, 3.0, 4.0]


def test_object_detection_from_inference_response(sample_image_bytes):
    output = ObjectDetectionOutput(
        detections=[
            [
                DetectedObject(
                    label="person", score=0.95,
                    box=BoundingBox(xmin=10.0, ymin=20.0, xmax=100.0, ymax=200.0),
                )
            ]
        ]
    )
    recovered = ObjectDetectionOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert len(recovered.detections) == 1       # 1 image
    assert len(recovered.detections[0]) == 1    # 1 detection
    det = recovered.detections[0][0]
    assert det.label == "person"
    assert det.box.xmin == 10.0
    assert det.box.ymax == 200.0


# ---------------------------------------------------------------------------
# DepthEstimation
# ---------------------------------------------------------------------------


def test_depth_estimation_from_inference_request(sample_image_bytes):
    original = DepthEstimationInput(images=[sample_image_bytes])
    recovered = DepthEstimationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.images == [sample_image_bytes]


def test_depth_estimation_to_inference_response_no_depth_image(sample_image_bytes):
    output = DepthEstimationOutput(
        predicted_depth=[[1.0, 2.0, 3.0, 4.0]], width=2, height=2
    )
    resp = output.to_inference_response("m")
    assert resp.get_output("predicted_depth").data == [[1.0, 2.0, 3.0, 4.0]]
    assert resp.get_output("predicted_depth").shape == [1, 2, 2]
    output_names = {o.name for o in resp.outputs}
    assert "depth" not in output_names


def test_depth_estimation_to_inference_response_with_depth_image(sample_image_bytes):
    output = DepthEstimationOutput(
        predicted_depth=[[1.0, 2.0, 3.0, 4.0]], width=2, height=2,
        depth=[sample_image_bytes],
    )
    resp = output.to_inference_response("m")
    depth_tensor = resp.get_output("depth")
    assert depth_tensor is not None
    assert len(depth_tensor.data) == 1


def test_depth_estimation_from_inference_response(sample_image_bytes):
    output = DepthEstimationOutput(
        predicted_depth=[[1.0, 2.0, 3.0, 4.0]], width=2, height=2
    )
    recovered = DepthEstimationOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert recovered.predicted_depth == [[1.0, 2.0, 3.0, 4.0]]
    assert recovered.height == 2
    assert recovered.width == 2
    assert recovered.depth is None


def test_depth_estimation_from_inference_response_with_depth_image(sample_image_bytes):
    output = DepthEstimationOutput(
        predicted_depth=[[1.0, 2.0, 3.0, 4.0]], width=2, height=2,
        depth=[sample_image_bytes],
    )
    recovered = DepthEstimationOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert recovered.depth == [sample_image_bytes]


# ---------------------------------------------------------------------------
# VQA
# ---------------------------------------------------------------------------


def test_vqa_from_inference_request(sample_image_bytes):
    original = VQAInput(images=[sample_image_bytes], question="What colour?", top_k=3)
    recovered = VQAInput.from_inference_request(original.to_inference_request())
    assert recovered.images == [sample_image_bytes]
    assert recovered.question == "What colour?"
    assert recovered.top_k == 3


def test_vqa_to_inference_response(sample_image_bytes):
    output = VQAOutput(answers=[[VQAAnswer(answer="red", score=0.7)]])
    resp = output.to_inference_response("m")
    assert resp.get_output("answer").data == [["red"]]
    assert abs(resp.get_output("score").data[0][0] - 0.7) < 1e-6


def test_vqa_from_inference_response(sample_image_bytes):
    output = VQAOutput(answers=[[VQAAnswer(answer="yes", score=0.9)]])
    recovered = VQAOutput.from_inference_response(output.to_inference_response("m"))
    assert recovered.answers[0][0].answer == "yes"
    assert abs(recovered.answers[0][0].score - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# ImageAnonymization
# ---------------------------------------------------------------------------


def test_image_anonymization_pipeline(sample_image_bytes):
    task = _StubImageAnonymizationTask()
    resp = task(ImageAnonymizationInput(images=[sample_image_bytes]).to_inference_request())
    result = ImageAnonymizationOutput.from_inference_response(resp)
    assert result.num_regions_anonymized == [2]
    assert isinstance(result.images[0], bytes)


def test_image_anonymization_to_inference_request_default(sample_image_bytes):
    req = ImageAnonymizationInput(images=[sample_image_bytes]).to_inference_request()
    assert req.parameters["strategy"] == "blur"
    assert req.parameters["blur_radius"] == 51
    assert req.parameters["threshold"] == 0.5
    assert "classes" not in req.parameters


def test_image_anonymization_to_inference_request_with_classes(sample_image_bytes):
    inp = ImageAnonymizationInput(
        images=[sample_image_bytes], classes=["person", "face"]
    )
    req = inp.to_inference_request()
    assert req.parameters["classes"] == ["person", "face"]


def test_image_anonymization_from_inference_request(sample_image_bytes):
    original = ImageAnonymizationInput(
        images=[sample_image_bytes],
        strategy=AnonymizationStrategy.PIXELATE,
        blur_radius=31,
        threshold=0.7,
        classes=["face"],
    )
    recovered = ImageAnonymizationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.images == [sample_image_bytes]
    assert recovered.strategy == AnonymizationStrategy.PIXELATE
    assert recovered.blur_radius == 31
    assert abs(recovered.threshold - 0.7) < 1e-6
    assert recovered.classes == ["face"]


def test_image_anonymization_from_inference_request_no_classes(sample_image_bytes):
    original = ImageAnonymizationInput(images=[sample_image_bytes])
    recovered = ImageAnonymizationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.classes is None


def test_image_anonymization_to_inference_response(sample_image_bytes):
    output = ImageAnonymizationOutput(images=[sample_image_bytes], num_regions_anonymized=[3])
    resp = output.to_inference_response("m")
    assert resp.get_output("num_regions_anonymized").data == [3]


def test_image_anonymization_from_inference_response(sample_image_bytes):
    output = ImageAnonymizationOutput(images=[sample_image_bytes], num_regions_anonymized=[5])
    recovered = ImageAnonymizationOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert recovered.num_regions_anonymized == [5]
    assert recovered.images == [sample_image_bytes]


# ---------------------------------------------------------------------------
# ImageSegmentation
# ---------------------------------------------------------------------------


def test_image_segmentation_pipeline(sample_image_bytes):
    task = _StubImageSegmentationTask()
    resp = task(ImageSegmentationInput(images=[sample_image_bytes]).to_inference_request())
    result = ImageSegmentationOutput.from_inference_response(resp)
    assert len(result.segments) == 1         # 1 image
    assert len(result.segments[0]) == 1      # 1 segment
    assert result.segments[0][0].label == "sky"


def test_image_segmentation_to_inference_request_no_subtask(sample_image_bytes):
    req = ImageSegmentationInput(images=[sample_image_bytes]).to_inference_request()
    assert abs(req.parameters["threshold"] - 0.9) < 1e-6
    assert "subtask" not in req.parameters


def test_image_segmentation_to_inference_request_with_subtask(sample_image_bytes):
    req = ImageSegmentationInput(
        images=[sample_image_bytes], subtask="panoptic"
    ).to_inference_request()
    assert req.parameters["subtask"] == "panoptic"


def test_image_segmentation_from_inference_request(sample_image_bytes):
    original = ImageSegmentationInput(
        images=[sample_image_bytes], threshold=0.8, subtask="semantic"
    )
    recovered = ImageSegmentationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.images == [sample_image_bytes]
    assert abs(recovered.threshold - 0.8) < 1e-6
    assert recovered.subtask == "semantic"


def test_image_segmentation_from_inference_request_no_subtask(sample_image_bytes):
    original = ImageSegmentationInput(images=[sample_image_bytes])
    recovered = ImageSegmentationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.subtask is None


def test_image_segmentation_to_inference_response(sample_image_bytes):
    mask = _make_mask_bytes()
    output = ImageSegmentationOutput(
        segments=[[SegmentResult(label="road", score=0.85, mask=mask)]]
    )
    resp = output.to_inference_response("m")
    # Per-image nested: labels[0] is first image
    assert resp.get_output("label").data[0] == ["road"]
    assert abs(resp.get_output("score").data[0][0] - 0.85) < 1e-6
    assert len(resp.get_output("mask").data[0]) == 1


def test_image_segmentation_from_inference_response(sample_image_bytes):
    mask = _make_mask_bytes()
    output = ImageSegmentationOutput(
        segments=[[SegmentResult(label="sky", score=0.9, mask=mask)]]
    )
    recovered = ImageSegmentationOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert len(recovered.segments) == 1
    assert len(recovered.segments[0]) == 1
    assert recovered.segments[0][0].label == "sky"
    assert recovered.segments[0][0].mask == mask


# ---------------------------------------------------------------------------
# MaskGeneration
# ---------------------------------------------------------------------------


def test_mask_generation_pipeline(sample_image_bytes):
    task = _StubMaskGenerationTask()
    resp = task(MaskGenerationInput(images=[sample_image_bytes]).to_inference_request())
    result = MaskGenerationOutput.from_inference_response(resp)
    assert len(result.masks) == 1           # 1 image
    assert len(result.masks[0]) == 1        # 1 mask
    assert abs(result.masks[0][0].score - 0.95) < 1e-6


def test_mask_generation_to_inference_request_no_points(sample_image_bytes):
    req = MaskGenerationInput(images=[sample_image_bytes]).to_inference_request()
    names = {t.name for t in req.inputs}
    assert "image" in names
    assert "points" not in names
    assert "point_labels" not in names

def test_mask_generation_from_inference_request_no_points(sample_image_bytes):
    original = MaskGenerationInput(images=[sample_image_bytes])
    recovered = MaskGenerationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.images == [sample_image_bytes]


def test_mask_generation_to_inference_response(sample_image_bytes):
    mask = _make_mask_bytes()
    output = MaskGenerationOutput(masks=[[GeneratedMask(mask=mask, score=0.9)]])
    resp = output.to_inference_response("m")
    assert len(resp.get_output("mask").data[0]) == 1
    assert abs(resp.get_output("score").data[0][0] - 0.9) < 1e-6


def test_mask_generation_from_inference_response(sample_image_bytes):
    mask = _make_mask_bytes()
    output = MaskGenerationOutput(masks=[[GeneratedMask(mask=mask, score=0.88)]])
    recovered = MaskGenerationOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert len(recovered.masks) == 1
    assert len(recovered.masks[0]) == 1
    assert recovered.masks[0][0].mask == mask
    assert abs(recovered.masks[0][0].score - 0.88) < 1e-6


# ===========================================================================
# ZeroShotImageClassification
# ===========================================================================


def test_zero_shot_image_classification_build_request(sample_image_bytes):
    inp = ZeroShotImageClassificationInput(
        images=[sample_image_bytes], candidate_labels=["cat", "dog"]
    )
    req = inp.to_inference_request()
    names = {t.name for t in req.inputs}
    assert "image" in names
    assert "candidate_labels" in names
    labels_tensor = next(t for t in req.inputs if t.name == "candidate_labels")
    assert labels_tensor.data == ["cat", "dog"]
    assert labels_tensor.shape == [2]


def test_zero_shot_image_classification_from_inference_request(sample_image_bytes):
    original = ZeroShotImageClassificationInput(
        images=[sample_image_bytes], candidate_labels=["cat", "dog", "bird"]
    )
    recovered = ZeroShotImageClassificationInput.from_inference_request(original.to_inference_request())
    assert recovered.images == [sample_image_bytes]
    assert recovered.candidate_labels == ["cat", "dog", "bird"]


def test_zero_shot_image_classification_full_pipeline(sample_image_bytes):
    task = _StubZeroShotImageClassificationTask()
    resp = task(
        ZeroShotImageClassificationInput(
            images=[sample_image_bytes], candidate_labels=["cat", "dog"]
        ).to_inference_request()
    )
    result = ZeroShotImageClassificationOutput.from_inference_response(resp)
    assert len(result.results) == 1
    assert len(result.results[0]) == 2
    assert result.results[0][0].label == "cat"


def test_zero_shot_image_classification_to_inference_response(sample_image_bytes):
    output = ZeroShotImageClassificationOutput(
        results=[[ZeroShotClassificationResult(label="cat", score=0.9)]]
    )
    resp = output.to_inference_response("m")
    assert resp.get_output("label").data == [["cat"]]
    assert abs(resp.get_output("score").data[0][0] - 0.9) < 1e-6


def test_zero_shot_image_classification_from_inference_response():
    output = ZeroShotImageClassificationOutput(
        results=[[ZeroShotClassificationResult(label="cat", score=0.85)]]
    )
    recovered = ZeroShotImageClassificationOutput.from_inference_response(output.to_inference_response("m"))
    assert len(recovered.results) == 1
    assert recovered.results[0][0].label == "cat"
    assert abs(recovered.results[0][0].score - 0.85) < 1e-6


# ===========================================================================
# ZeroShotObjectDetection
# ===========================================================================


def test_zero_shot_object_detection_build_request(sample_image_bytes):
    inp = ZeroShotObjectDetectionInput(
        images=[sample_image_bytes], candidate_labels=["cat", "dog"], threshold=0.2
    )
    req = inp.to_inference_request()
    names = {t.name for t in req.inputs}
    assert "image" in names
    assert "candidate_labels" in names
    labels_tensor = next(t for t in req.inputs if t.name == "candidate_labels")
    assert labels_tensor.data == ["cat", "dog"]
    assert req.parameters == {"threshold": 0.2}


def test_zero_shot_object_detection_from_inference_request(sample_image_bytes):
    original = ZeroShotObjectDetectionInput(
        images=[sample_image_bytes], candidate_labels=["person", "car"], threshold=0.3
    )
    recovered = ZeroShotObjectDetectionInput.from_inference_request(original.to_inference_request())
    assert recovered.images == [sample_image_bytes]
    assert recovered.candidate_labels == ["person", "car"]
    assert abs(recovered.threshold - 0.3) < 1e-6


def test_zero_shot_object_detection_full_pipeline(sample_image_bytes):
    task = _StubZeroShotObjectDetectionTask()
    resp = task(
        ZeroShotObjectDetectionInput(
            images=[sample_image_bytes], candidate_labels=["cat"]
        ).to_inference_request()
    )
    result = ZeroShotObjectDetectionOutput.from_inference_response(resp)
    assert len(result.detections) == 1
    assert len(result.detections[0]) == 1
    det = result.detections[0][0]
    assert det.label == "cat"
    assert det.box.xmin == 10.0
    assert det.box.ymax == 200.0


def test_zero_shot_object_detection_to_inference_response(sample_image_bytes):
    output = ZeroShotObjectDetectionOutput(
        detections=[[DetectedObject(label="dog", score=0.75, box=BoundingBox(xmin=5.0, ymin=10.0, xmax=50.0, ymax=100.0))]]
    )
    resp = output.to_inference_response("m")
    assert resp.get_output("label").data == [["dog"]]
    assert abs(resp.get_output("score").data[0][0] - 0.75) < 1e-6


def test_zero_shot_object_detection_from_inference_response():
    output = ZeroShotObjectDetectionOutput(
        detections=[[DetectedObject(label="cat", score=0.9, box=BoundingBox(xmin=1.0, ymin=2.0, xmax=3.0, ymax=4.0))]]
    )
    recovered = ZeroShotObjectDetectionOutput.from_inference_response(output.to_inference_response("m"))
    assert len(recovered.detections) == 1
    assert recovered.detections[0][0].label == "cat"
    assert recovered.detections[0][0].box.xmin == 1.0

