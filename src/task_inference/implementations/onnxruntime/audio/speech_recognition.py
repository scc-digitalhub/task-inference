# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - Automatic Speech Recognition (pure ORT, no transformers).

Dialect is auto-detected from ``config.json`` and the ONNX file layout.
Supported dialects:

* **whisper-encoder-decoder** — encoder-decoder Whisper model
  (``encoder_model.onnx`` + ``decoder_model.onnx`` / ``decoder_model_merged.onnx``)
* **wav2vec2-ctc** — CTC models: wav2vec2, HuBERT, Wav2Vec2Conformer;
  raw PCM input, greedy argmax CTC decode
"""
from __future__ import annotations

from ....tasks.audio.speech_recognition import ASRChunk, ASRInput, ASROutput, ASRTask
from ..base import OnnxRuntimeTaskMixin
from ..adapters.audio.asr import resolve_asr_adapter


class OnnxASRTask(OnnxRuntimeTaskMixin, ASRTask):
    """Automatic Speech Recognition using a local ONNX model.

    Dialect is auto-detected from ``config.json`` → ``model_type`` and the
    presence of ``encoder_model.onnx``:

    * **whisper-encoder-decoder** — log-mel spectrogram input (via ``librosa``),
      greedy encoder-decoder decode, BPE text decoding (``tokenizers`` package).
    * **wav2vec2-ctc** — normalised PCM input, greedy CTC decode, vocab
      decoded via ``tokenizer.json`` / ``vocab.json``.

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device.
    chunk_length_s:
        Ignored (kept for API compatibility with the transformers backend).

    Example
    -------
    ::

        optimum-cli export onnx --model openai/whisper-base ./onnx/whisper-base/
        task = OnnxASRTask(model_name="./onnx/whisper-base/")
    """

    def __init__(
        self,
        model_name: str,
        device: str | int = "cpu",
        chunk_length_s: int = 30,  # kept for API compatibility
    ) -> None:
        self.model_name = model_name
        self.device = device
        providers = self._device_to_providers(device)
        cfg = self._load_config(model_name)
        self._adapter = resolve_asr_adapter(model_name, providers, cfg)

    def process(self, inputs: ASRInput) -> ASROutput:
        texts: list[str] = []
        chunks_list: list[list[ASRChunk]] = []

        for raw_pcm in inputs.audio:
            audio = self._raw_pcm_to_numpy(raw_pcm)
            text, chunks = self._adapter.transcribe(
                audio,
                language=inputs.language,
                return_timestamps=inputs.return_timestamps,
            )
            texts.append(text)
            chunks_list.append(chunks or [])

        return ASROutput(
            texts=texts,
            chunks=chunks_list if inputs.return_timestamps else None,
        )

