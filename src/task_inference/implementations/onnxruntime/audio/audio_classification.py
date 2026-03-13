# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - audio classification (pure ORT, no transformers)."""
from __future__ import annotations

import numpy as np

from ....tasks.audio.audio_classification import (
    AudioClassificationInput,
    AudioClassificationOutput,
    AudioClassificationResult,
    AudioClassificationTask,
)
from ..base import OnnxRuntimeTaskMixin
from ..adapters.audio.classification import resolve_audio_classification_adapter


class OnnxAudioClassificationTask(OnnxRuntimeTaskMixin, AudioClassificationTask):
    """Audio classification using a local ONNX model.

    Dialect is auto-detected from the model's ONNX input/output tensor names.
    Supported dialects:

    * **input-values-logits** — ``input_values`` → ``logits``
      (wav2vec2, HuBERT, AST, …)

    The model directory must contain:

    * ``model.onnx`` — the exported ONNX model
    * ``config.json`` — includes ``id2label`` mapping
    * ``preprocessor_config.json`` — includes ``do_normalize`` flag

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device (``'cpu'``, ``'cuda'``, ``'mps'``, or int CUDA
        device index).

    Example
    -------
    ::

        optimum-cli export onnx --model superb/wav2vec2-base-superb-ks ./onnx/wav2vec2-ks/
        task = OnnxAudioClassificationTask(model_name="./onnx/wav2vec2-ks/")
    """

    def __init__(self, model_name: str, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        cfg = self._load_config(model_name)
        self._id2label: dict[int, str] = {
            int(k): v for k, v in cfg.get("id2label", {}).items()
        }
        pp_cfg = self._load_preprocessor_config(model_name)
        do_normalize: bool = bool(pp_cfg.get("do_normalize", True))
        providers = self._device_to_providers(device)
        session = self._create_session(self._find_onnx_file(model_name), providers)
        self._adapter = resolve_audio_classification_adapter(session, do_normalize, cfg)

    def process(self, inputs: AudioClassificationInput) -> AudioClassificationOutput:
        results = []
        for raw_pcm in inputs.audio:
            audio = self._raw_pcm_to_numpy(raw_pcm)
            logits = self._adapter.classify_audio(audio)
            probs = self._softmax(logits)
            top_k = min(inputs.top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]
            results.append(
                [
                    AudioClassificationResult(
                        label=self._id2label.get(int(i), str(int(i))),
                        score=float(probs[i]),
                    )
                    for i in top_indices
                ]
            )
        return AudioClassificationOutput(results=results)
