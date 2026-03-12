# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - audio classification."""
from __future__ import annotations

from transformers import pipeline

from ....tasks.audio.audio_classification import (
    AudioClassificationInput,
    AudioClassificationOutput,
    AudioClassificationResult,
    AudioClassificationTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "superb/wav2vec2-base-superb-ks"


class TransformersAudioClassificationTask(
    TransformersTaskMixin, AudioClassificationTask
):
    """Audio classification using a HuggingFace ``audio-classification`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``audio-classification`` compatible model on the Hub.
        Defaults to ``superb/wav2vec2-base-superb-ks`` (keyword spotting).
        For environmental sound classification try e.g.
        ``MIT/ast-finetuned-audioset-10-10-0.4593``.
    device:
        Inference device.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | int = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline(
            "audio-classification",
            model=model_name,
            device=device,
        )

    def process(self, inputs: AudioClassificationInput) -> AudioClassificationOutput:
        audio_dict = [self._raw_pcm_to_array(a, inputs.sample_rate) for a in inputs.audio]
        results_list = self._pipe(audio_dict, top_k=inputs.top_k)
        return AudioClassificationOutput(
            results=[
                [AudioClassificationResult(label=r["label"], score=float(r["score"]))
                for r in results]
                for results in results_list
            ]
        ) 
