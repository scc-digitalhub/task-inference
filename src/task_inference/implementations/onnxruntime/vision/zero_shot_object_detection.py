# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - zero-shot object detection (pure ORT).

Compatible with OWL-ViT models exported via ``optimum-cli export onnx``.
Requires the ``tokenizers`` package and a ``tokenizer.json`` in the model
directory.
"""
from __future__ import annotations

import pathlib

from ....tasks.vision.zero_shot_object_detection import (
    ZeroShotObjectDetectionInput,
    ZeroShotObjectDetectionOutput,
    ZeroShotObjectDetectionTask,
)
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.zero_shot_detection import resolve_zero_shot_detection_adapter


class OnnxZeroShotObjectDetectionTask(
    OnnxRuntimeTaskMixin, ZeroShotObjectDetectionTask
):
    """Zero-shot object detection using a local ONNX model.

    Dialect is auto-detected from the model's ONNX input/output tensor names.
    Supported dialects:

    * **transformers-owlvit** — ``input_ids`` + ``attention_mask`` +
      ``pixel_values`` → ``logits [B,P,N]`` + ``pred_boxes [B,P,4]``

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

        optimum-cli export onnx --model google/owlvit-base-patch32 ./onnx/owlvit/
        task = OnnxZeroShotObjectDetectionTask(model_name="./onnx/owlvit/")
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
        self._adapter = resolve_zero_shot_detection_adapter(session, pp_cfg, cfg, tokenizer)

    def process(
        self, inputs: ZeroShotObjectDetectionInput
    ) -> ZeroShotObjectDetectionOutput:
        all_detections = []
        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            dets = self._adapter.detect_zero_shot(
                pil_image, inputs.candidate_labels, inputs.threshold
            )
            all_detections.append(dets)
        return ZeroShotObjectDetectionOutput(detections=all_detections)
