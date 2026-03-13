# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - zero-shot image classification (pure ORT).

Compatible with CLIP models exported via ``optimum-cli export onnx``.
Requires the ``tokenizers`` package and a ``tokenizer.json`` in the model
directory.
"""
from __future__ import annotations

import pathlib

import numpy as np

from ....tasks.vision.zero_shot_image_classification import (
    ZeroShotClassificationResult,
    ZeroShotImageClassificationInput,
    ZeroShotImageClassificationOutput,
    ZeroShotImageClassificationTask,
)
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.zero_shot_classification import resolve_zero_shot_classification_adapter


class OnnxZeroShotImageClassificationTask(
    OnnxRuntimeTaskMixin, ZeroShotImageClassificationTask
):
    """Zero-shot image classification using a local ONNX model.

    Compatible with CLIP models exported via ``optimum-cli export onnx``.
    The combined ONNX model accepts:

    * ``input_ids`` — ``[num_labels, seq_len]``
    * ``attention_mask`` — ``[num_labels, seq_len]``
    * ``pixel_values`` — ``[1, 3, H, W]``

    and outputs ``logits_per_image`` of shape ``[1, num_labels]``.

    The model directory must contain:

    * ``model.onnx``
    * ``preprocessor_config.json``
    * ``tokenizer.json``

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device.

    Example
    -------
    ::

        optimum-cli export onnx --model openai/clip-vit-base-patch32 ./onnx/clip/
        task = OnnxZeroShotImageClassificationTask(model_name="./onnx/clip/")
    """

    def __init__(self, model_name: str, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        cfg = self._load_config(model_name)
        pp_cfg = self._load_preprocessor_config(model_name)
        providers = self._device_to_providers(device)
        session = self._create_session(self._find_onnx_file(model_name), providers)

        from tokenizers import Tokenizer  # noqa: PLC0415
        tokenizer_path = pathlib.Path(model_name) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._adapter = resolve_zero_shot_classification_adapter(session, pp_cfg, cfg, tokenizer)

    def process(
        self, inputs: ZeroShotImageClassificationInput
    ) -> ZeroShotImageClassificationOutput:
        results_per_image = []
        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            probs = self._adapter.classify_zero_shot(pil_image, inputs.candidate_labels)
            order = np.argsort(probs)[::-1]
            results_per_image.append(
                [
                    ZeroShotClassificationResult(
                        label=inputs.candidate_labels[int(i)],
                        score=float(probs[i]),
                    )
                    for i in order
                ]
            )
        return ZeroShotImageClassificationOutput(results=results_per_image)
