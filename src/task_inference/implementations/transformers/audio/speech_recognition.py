# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - Automatic Speech Recognition."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.audio.speech_recognition import ASRChunk, ASRInput, ASROutput, ASRTask
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "openai/whisper-base"


class TransformersASRTask(TransformersTaskMixin, ASRTask):
    """ASR using a HuggingFace ``automatic-speech-recognition`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``automatic-speech-recognition`` compatible model on the Hub.
        Defaults to ``openai/whisper-base``.
    device:
        Inference device.
    chunk_length_s:
        Audio chunk length in seconds for long-form transcription.
        Set to ``0`` to disable chunking (short clips only).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | int = "cpu",
        chunk_length_s: int = 30,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._chunk_length_s = chunk_length_s
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
            chunk_length_s=chunk_length_s if chunk_length_s > 0 else None,
        )

    def process(self, inputs: ASRInput) -> ASROutput:
        audio_dict = [self._raw_pcm_to_array(a, inputs.sample_rate) for a in inputs.audio]

        kwargs: dict = {}
        if inputs.return_timestamps:
            kwargs["return_timestamps"] = True
        if inputs.language:
            kwargs["generate_kwargs"] = {"language": inputs.language}

        result_list = self._pipe(audio_dict, **kwargs)

        full_text_list: list[str] = [result.get("text", "") for result in result_list]
        chunks_list: list[list[ASRChunk]] | None = None

        if inputs.return_timestamps:
            raw_chunks_list = [result.get("chunks", []) for result in result_list]
            chunks_list = [
                [
                    ASRChunk(
                        text=c.get("text", ""),
                        timestamp_start=float(c["timestamp"][0]) if c.get("timestamp") else None,
                        timestamp_end=float(c["timestamp"][1]) if (c.get("timestamp") and c["timestamp"][1] is not None) else None,
                    )
                    for c in raw_chunks
                ]
                for raw_chunks in raw_chunks_list
            ]


        return ASROutput(texts=full_text_list, chunks=chunks_list)
