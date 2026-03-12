# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Automatic Speech Recognition (ASR) task - OIP v2 schema and conversion."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_audio, encode_audio
from ..base import BaseTask

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class ASRInput(BaseModel):
    """Inputs for an Automatic Speech Recognition request."""

    audio: list[bytes] = Field(..., description="Raw PCM float32 audio samples, little-endian")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    language: str | None = Field(
        None, description="BCP-47 language code hint (e.g. 'en', 'fr')"
    )
    return_timestamps: bool = Field(
        False, description="Whether to return word/chunk-level timestamps"
    )
    
    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        params: dict = {"return_timestamps": self.return_timestamps, "sample_rate": self.sample_rate}
        if self.language is not None:
            params["language"] = self.language
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name=_IN_AUDIO,
                    shape=[len(self.audio)],
                    datatype=Datatype.BYTES,
                    data=[encode_audio(a) for a in self.audio],
                )
            ],
            parameters=params,
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "ASRInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        audio_b64_list = request.inputs[0].data or []
        params = request.parameters or {}
        return cls(
            audio=[decode_audio(a) for a in audio_b64_list],
            sample_rate=int(params["sample_rate"]),
            language=params.get("language"),
            return_timestamps=bool(params.get("return_timestamps", False)),
        )


class ASRChunk(BaseModel):
    """A single timestamped transcript chunk."""

    text: str
    timestamp_start: float | None = None
    timestamp_end: float | None = None


class ASROutput(BaseModel):
    """Outputs of an ASR request."""

    texts: list[str] = Field(..., description="Full transcription text")
    chunks: list[list[ASRChunk]] | None = Field(
        None, description="Per-chunk transcription with timestamps (when requested)"
    )

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        outputs = [
            ResponseOutput(name=_OUT_TEXT, shape=[len(self.texts)], datatype=Datatype.BYTES, data=self.texts)
        ]
        if self.chunks is not None:
            chunk_texts = [[c.text for c in chunk] for chunk in self.chunks]
            ts_starts = [[c.timestamp_start or 0.0 for c in chunk] for chunk in self.chunks]
            ts_ends = [[c.timestamp_end or 0.0 for c in chunk] for chunk in self.chunks]
            n = len(self.texts)
            m = min(len(chunk) for chunk in self.chunks) if self.chunks else 0
            outputs += [
                ResponseOutput(name=_OUT_CHUNK_TEXTS, shape=[n, m], datatype=Datatype.BYTES, data=chunk_texts),
                ResponseOutput(name=_OUT_CHUNK_TS_START, shape=[n, m], datatype=Datatype.FP32, data=ts_starts),
                ResponseOutput(name=_OUT_CHUNK_TS_END, shape=[n, m], datatype=Datatype.FP32, data=ts_ends),
            ]
        return InferenceResponse(model_name=model_name, outputs=outputs)

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "ASROutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        texts = response.get_output(_OUT_TEXT).data or [""]
        chunks: list[list[ASRChunk]] | None = None
        try:
            chunk_texts = response.get_output(_OUT_CHUNK_TEXTS).data or []
            ts_starts = response.get_output(_OUT_CHUNK_TS_START).data or []
            ts_ends = response.get_output(_OUT_CHUNK_TS_END).data or []
            chunks = [
                [ASRChunk(text=t, timestamp_start=s, timestamp_end=e) for t, s, e in zip(ct, ts, te, strict=False)]
                for ct, ts, te in zip(chunk_texts, ts_starts, ts_ends, strict=False)
            ]
        except KeyError:
            pass
        return cls(texts=texts, chunks=chunks or None)


# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_AUDIO = "audio"
_OUT_TEXT = "text"
_OUT_CHUNK_TEXTS = "chunks_texts"          # STRING [C]
_OUT_CHUNK_TS_START = "chunks_ts_start"  # FP32   [C]
_OUT_CHUNK_TS_END = "chunks_ts_end"      # FP32   [C]


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class ASRTask(BaseTask[ASRInput, ASROutput]):
    """Task definition for Automatic Speech Recognition.

    OIP v2 request tensors
    ----------------------
    * ``audio``   - ``BYTES [1]``  audio file bytes
    * parameters: ``sample_rate``, ``language``, ``return_timestamps``

    OIP v2 response tensors
    -----------------------
    * ``text``            - ``BYTES [-1]``   full transcription
    * ``chunks_texts``     - ``BYTES [-1, -1]``   per-chunk text (when timestamps)
    * ``chunks_ts_start``  - ``FP32   [-1, -1]``   chunk start times in seconds
    * ``chunks_ts_end``    - ``FP32   [-1, -1]``   chunk end times in seconds
    """

    TASK_NAME = "automatic-speech-recognition"
    INPUT_SCHEMA = ASRInput
    OUTPUT_SCHEMA = ASROutput

    METADATA_INPUTS = [
        MetadataTensor(name="audio", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="text", datatype=Datatype.BYTES, shape=[-1]),
        MetadataTensor(name="chunks_texts", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="chunks_ts_start", datatype=Datatype.FP32, shape=[-1, -1]),
        MetadataTensor(name="chunks_ts_end", datatype=Datatype.FP32, shape=[-1, -1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="sample_rate", datatype="int64", required=True, description="Sample rate of the input audio in Hz"),
        ParameterMetadata(name="return_timestamps", datatype="bool", required=False, default=False, description="Whether to return word/chunk-level timestamps"),
        ParameterMetadata(name="language", datatype="string", required=False, default=None, description="BCP-47 language code hint (e.g. 'en', 'fr')"),
    ]
