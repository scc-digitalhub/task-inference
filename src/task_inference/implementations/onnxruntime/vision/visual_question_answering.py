# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - Visual Question Answering (pure ORT, no transformers).

Compatible with ViLT-style VQA models exported via ``optimum-cli export onnx``.
Requires the ``tokenizers`` package (``pip install tokenizers``) and a
``tokenizer.json`` file in the model directory.
"""
from __future__ import annotations

import pathlib

import numpy as np

from ....tasks.vision.visual_question_answering import VQAAnswer, VQAInput, VQAOutput, VQATask
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.vqa import resolve_vqa_adapter


class OnnxVQATask(OnnxRuntimeTaskMixin, VQATask):
    """Visual Question Answering using a local ONNX model.

    Compatible with ViLT-style models exported via ``optimum-cli export onnx``.
    The model accepts five inputs:

    * ``input_ids`` — tokenized question ``[1, seq_len]``
    * ``attention_mask`` — ``[1, seq_len]``
    * ``token_type_ids`` — ``[1, seq_len]``
    * ``pixel_values`` — ``[1, 3, H, W]``
    * ``pixel_mask`` — ``[1, H, W]``

    and outputs ``logits`` of shape ``[1, num_labels]``.

    The model directory must contain:

    * ``model.onnx``
    * ``config.json`` (``id2label``)
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

        optimum-cli export onnx --model dandelin/vilt-b32-finetuned-vqa ./onnx/vilt-vqa/
        task = OnnxVQATask(model_name="./onnx/vilt-vqa/")
    """

    def __init__(self, model_name: str, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        cfg = self._load_config(model_name)
        self._id2label: dict[int, str] = {
            int(k): v for k, v in cfg.get("id2label", {}).items()
        }
        pp_cfg = self._load_preprocessor_config(model_name)
        providers = self._device_to_providers(device)
        session = self._create_session(self._find_onnx_file(model_name), providers)

        from tokenizers import Tokenizer  # noqa: PLC0415
        tokenizer_path = pathlib.Path(model_name) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._adapter = resolve_vqa_adapter(session, pp_cfg, cfg, tokenizer)

    def process(self, inputs: VQAInput) -> VQAOutput:
        answers_per_image = []
        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            logits = self._adapter.answer(pil_image, inputs.question)
            probs = self._softmax(logits)
            top_k = min(inputs.top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]
            answers_per_image.append(
                [
                    VQAAnswer(
                        answer=self._id2label.get(int(i), str(int(i))),
                        score=float(probs[i]),
                    )
                    for i in top_indices
                ]
            )
        return VQAOutput(answers=answers_per_image)
