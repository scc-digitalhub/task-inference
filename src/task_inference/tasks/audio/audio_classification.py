# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Audio classification task - OIP v2 schema and conversion methods."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ...protocol.v2 import Datatype, InferenceRequest, InferenceResponse, MetadataTensor, ParameterMetadata, RequestInput, ResponseOutput
from ...utils import decode_audio, encode_audio
from ..base import BaseTask

# ---------------------------------------------------------------------------
# Python-level input / output schemas
# ---------------------------------------------------------------------------


class AudioClassificationInput(BaseModel):
    """Inputs for an audio classification request."""

    audio: list[bytes] = Field(..., description="Raw PCM float32 audio samples, little-endian")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    top_k: int = Field(5, ge=1, description="Number of top predictions to return")

    def to_inference_request(self) -> InferenceRequest:
        """Serialise to an OIP v2 :class:`InferenceRequest`."""
        return InferenceRequest(
            inputs=[
                RequestInput(
                    name=_IN_AUDIO,
                    shape=[len(self.audio)],
                    datatype=Datatype.BYTES,
                    data=[encode_audio(a) for a in self.audio],
                )
            ],
            parameters={"top_k": self.top_k, "sample_rate": self.sample_rate},
        )

    @classmethod
    def from_inference_request(cls, request: InferenceRequest) -> "AudioClassificationInput":
        """Construct from an OIP v2 :class:`InferenceRequest`."""
        audio_b64_list = request.inputs[0].data or []
        params = request.parameters or {}
        return cls(
            audio=[decode_audio(a) for a in audio_b64_list],
            sample_rate=int(params["sample_rate"]),
            top_k=int(params.get("top_k", 5)),
        )


class AudioClassificationOutput(BaseModel):
    """Outputs of an audio classification request."""

    results: list[list[AudioClassificationResult]]

    def to_inference_response(self, model_name: str) -> InferenceResponse:
        """Convert to an OIP v2 :class:`InferenceResponse`."""
        labels = [[d.label for d in dets] for dets in self.results]
        scores = [[d.score for d in dets] for dets in self.results]

        n = len(labels)
        m = min(len(result) for result in self.results) if self.results else 0
        return InferenceResponse(
            model_name=model_name,
            outputs=[
                ResponseOutput(name=_OUT_LABELS, shape=[n, m], datatype=Datatype.BYTES, data=labels),
                ResponseOutput(name=_OUT_SCORES, shape=[n, m], datatype=Datatype.FP32, data=scores),
            ],
        )

    @classmethod
    def from_inference_response(cls, response: InferenceResponse) -> "AudioClassificationOutput":
        """Construct from an OIP v2 :class:`InferenceResponse`."""
        labels = response.get_output(_OUT_LABELS).data or []
        scores = response.get_output(_OUT_SCORES).data or []
        res = []
        for lbls, scs in zip(labels, scores, strict=False):
            ac_res = []
            for i, (lbl, sc) in enumerate(zip(lbls, scs, strict=False)):
                ac_res.append(
                    AudioClassificationResult( label=lbl, score=sc)
                )
            res.append(ac_res)
        return cls(results=res)

class AudioClassificationResult(BaseModel):
    """A single label-score pair."""

    label: str
    score: float


# ---------------------------------------------------------------------------
# OIP v2 tensor names
# ---------------------------------------------------------------------------

_IN_AUDIO = "audio"
_OUT_LABELS = "label"
_OUT_SCORES = "score"


# ---------------------------------------------------------------------------
# Abstract task definition
# ---------------------------------------------------------------------------


class AudioClassificationTask(BaseTask[AudioClassificationInput, AudioClassificationOutput]):
    """Task definition for audio classification.

    OIP v2 request tensors
    ----------------------
    * ``audio``   - ``BYTES [1]``   audio file bytes
    * parameters: ``sample_rate``, ``top_k``

    OIP v2 response tensors
    -----------------------
    * ``label``  - ``STRING [N]``  predicted class labels
    * ``score``  - ``FP32   [N]``  corresponding confidence scores
    """

    TASK_NAME = "audio-classification"
    INPUT_SCHEMA = AudioClassificationInput
    OUTPUT_SCHEMA = AudioClassificationOutput

    METADATA_INPUTS = [
        MetadataTensor(name="audio", datatype=Datatype.BYTES, shape=[-1]),
    ]
    METADATA_OUTPUTS = [
        MetadataTensor(name="label", datatype=Datatype.BYTES, shape=[-1, -1]),
        MetadataTensor(name="score", datatype=Datatype.FP32, shape=[-1, -1]),
    ]
    METADATA_PARAMETERS = [
        ParameterMetadata(name="sample_rate", datatype="int64", required=True, description="Sample rate of the input audio in Hz"),
        ParameterMetadata(name="top_k", datatype="int64", required=False, default=5, description="Number of top predictions to return per audio clip"),
    ]
