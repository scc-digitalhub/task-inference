# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for audio task schema and conversion methods (no model required)."""
from __future__ import annotations

from task_inference.protocol.v2 import Datatype
from task_inference.tasks.audio.audio_classification import (
    AudioClassificationInput,
    AudioClassificationOutput,
    AudioClassificationResult,
    AudioClassificationTask,
)
from task_inference.tasks.audio.speech_recognition import ASRChunk, ASRInput, ASROutput, ASRTask

# ---------------------------------------------------------------------------
# Stub tasks
# ---------------------------------------------------------------------------


class _StubASRTask(ASRTask):
    def process(self, inputs: ASRInput) -> ASROutput:
        return ASROutput(texts=["hello world"])


class _StubASRWithTimestampsTask(ASRTask):
    def process(self, inputs: ASRInput) -> ASROutput:
        return ASROutput(
            texts=["hello world"],
            chunks=[[
                ASRChunk(text="hello", timestamp_start=0.0, timestamp_end=0.4),
                ASRChunk(text="world", timestamp_start=0.5, timestamp_end=1.0),
            ]],
        )


class _StubAudioClassTask(AudioClassificationTask):
    def process(self, inputs: AudioClassificationInput) -> AudioClassificationOutput:
        return AudioClassificationOutput(
            results=[
                [
                    AudioClassificationResult(label="speech", score=0.8),
                    AudioClassificationResult(label="music", score=0.2),
                ]
            ]
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_asr_build_request(sample_audio_bytes):
    req = ASRInput(audio=[sample_audio_bytes], sample_rate=16000).to_inference_request()
    assert req.inputs[0].name == "audio"
    assert req.inputs[0].datatype == Datatype.BYTES
    assert req.inputs[0].data[0] == sample_audio_bytes


def test_asr_full_pipeline(sample_audio_bytes):
    task = _StubASRTask()
    resp = task(ASRInput(audio=[sample_audio_bytes], sample_rate=16000).to_inference_request())
    result = ASROutput.from_inference_response(resp)
    assert result.texts == ["hello world"]
    assert result.chunks is None


def test_asr_timestamps(sample_audio_bytes):
    task = _StubASRWithTimestampsTask()
    resp = task(ASRInput(audio=[sample_audio_bytes], sample_rate=16000, return_timestamps=True).to_inference_request())
    result = ASROutput.from_inference_response(resp)
    assert len(result.chunks) == 1
    assert len(result.chunks[0]) == 2
    assert result.chunks[0][0].text == "hello"
    assert result.chunks[0][0].timestamp_start == 0.0


def test_audio_classification_pipeline(sample_audio_bytes):
    task = _StubAudioClassTask()
    resp = task(AudioClassificationInput(audio=[sample_audio_bytes], sample_rate=16000, top_k=2).to_inference_request())
    result = AudioClassificationOutput.from_inference_response(resp)
    assert result.results[0][0].label == "speech"
    assert abs(result.results[0][0].score - 0.8) < 1e-6


# ===========================================================================
# OIP round-trip tests: from_inference_request / to/from_inference_response
# ===========================================================================


# ---------------------------------------------------------------------------
# AudioClassification
# ---------------------------------------------------------------------------


def test_audio_classification_to_inference_request_with_sample_rate(sample_audio_bytes):
    inp = AudioClassificationInput(audio=[sample_audio_bytes], sample_rate=16000, top_k=3)
    req = inp.to_inference_request()
    assert req.parameters["sample_rate"] == 16000
    assert req.parameters["top_k"] == 3


def test_audio_classification_from_inference_request(sample_audio_bytes):
    original = AudioClassificationInput(
        audio=[sample_audio_bytes], sample_rate=16000, top_k=3
    )
    recovered = AudioClassificationInput.from_inference_request(
        original.to_inference_request()
    )
    assert recovered.audio == [sample_audio_bytes]
    assert recovered.sample_rate == 16000
    assert recovered.top_k == 3


def test_audio_classification_to_inference_response(sample_audio_bytes):
    output = AudioClassificationOutput(
        results=[
            [
                AudioClassificationResult(label="speech", score=0.8),
                AudioClassificationResult(label="music", score=0.2),
            ]
        ]
    )
    resp = output.to_inference_response("m")
    assert resp.get_output("label").data == [["speech", "music"]]
    assert abs(resp.get_output("score").data[0][0] - 0.8) < 1e-6


def test_audio_classification_from_inference_response(sample_audio_bytes):
    output = AudioClassificationOutput(
        results=[[AudioClassificationResult(label="noise", score=0.6)]]
    )
    recovered = AudioClassificationOutput.from_inference_response(
        output.to_inference_response("m")
    )
    assert len(recovered.results) == 1
    assert len(recovered.results[0]) == 1
    assert recovered.results[0][0].label == "noise"
    assert abs(recovered.results[0][0].score - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------


def test_asr_to_inference_request_all_params(sample_audio_bytes):
    inp = ASRInput(
        audio=[sample_audio_bytes],
        sample_rate=16000,
        language="en",
        return_timestamps=True,
    )
    req = inp.to_inference_request()
    assert req.parameters["sample_rate"] == 16000
    assert req.parameters["language"] == "en"
    assert req.parameters["return_timestamps"] is True


def test_asr_to_inference_request_minimal_params(sample_audio_bytes):
    req = ASRInput(audio=[sample_audio_bytes], sample_rate=16000).to_inference_request()
    assert req.parameters["sample_rate"] == 16000
    assert "language" not in req.parameters
    assert req.parameters["return_timestamps"] is False


def test_asr_from_inference_request(sample_audio_bytes):
    original = ASRInput(
        audio=[sample_audio_bytes],
        sample_rate=8000,
        language="fr",
        return_timestamps=True,
    )
    recovered = ASRInput.from_inference_request(original.to_inference_request())
    assert recovered.audio == [sample_audio_bytes]
    assert recovered.sample_rate == 8000
    assert recovered.language == "fr"
    assert recovered.return_timestamps is True


def test_asr_to_inference_response_with_chunks(sample_audio_bytes):
    output = ASROutput(
        texts=["hello world"],
        chunks=[[
            ASRChunk(text="hello", timestamp_start=0.0, timestamp_end=0.4),
            ASRChunk(text="world", timestamp_start=0.5, timestamp_end=1.0),
        ]],
    )
    resp = output.to_inference_response("m")
    assert resp.get_output("text").data == ["hello world"]
    assert resp.get_output("chunks_texts").data == [["hello", "world"]]
    assert resp.get_output("chunks_ts_start").data == [[0.0, 0.5]]
    assert resp.get_output("chunks_ts_end").data == [[0.4, 1.0]]


def test_asr_from_inference_response_with_chunks(sample_audio_bytes):
    output = ASROutput(
        texts=["hi there"],
        chunks=[[ASRChunk(text="hi", timestamp_start=0.0, timestamp_end=0.3)]],
    )
    recovered = ASROutput.from_inference_response(output.to_inference_response("m"))
    assert recovered.texts == ["hi there"]
    assert len(recovered.chunks) == 1
    assert len(recovered.chunks[0]) == 1
    assert recovered.chunks[0][0].text == "hi"
    assert recovered.chunks[0][0].timestamp_start == 0.0


def test_asr_from_inference_response_no_chunks(sample_audio_bytes):
    output = ASROutput(texts=["hello"])
    recovered = ASROutput.from_inference_response(output.to_inference_response("m"))
    assert recovered.texts == ["hello"]
    assert recovered.chunks is None

