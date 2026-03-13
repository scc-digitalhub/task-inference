# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - image classification (pure ORT, no transformers)."""
from __future__ import annotations

import numpy as np

from ....tasks.vision.image_classification import (
    ClassificationResult,
    ImageClassificationInput,
    ImageClassificationOutput,
    ImageClassificationTask,
)
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.classification import resolve_classification_adapter


class OnnxImageClassificationTask(OnnxRuntimeTaskMixin, ImageClassificationTask):
    """Image classification using a local ONNX model.

    Dialect is auto-detected from the model's ONNX input/output tensor names.
    Supported dialects:

    * **pixel-logits** — ``pixel_values`` → ``logits`` (ViT, ResNet, EfficientNet, …)

    The model directory must contain:

    * ``model.onnx`` — the exported ONNX model
    * ``config.json`` — includes ``id2label`` mapping
    * ``preprocessor_config.json`` — image preprocessing parameters

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device (``'cpu'``, ``'cuda'``, ``'mps'``, or int).

    Example
    -------
    ::

        optimum-cli export onnx --model google/vit-base-patch16-224 ./onnx/vit/
        task = OnnxImageClassificationTask(model_name="./onnx/vit/")
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
        self._adapter = resolve_classification_adapter(session, pp_cfg, cfg)

    def process(self, inputs: ImageClassificationInput) -> ImageClassificationOutput:
        results = []
        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            logits = self._adapter.classify(pil_image)
            probs = self._softmax(logits)
            top_k = min(inputs.top_k, len(probs))
            top_indices = np.argsort(probs)[::-1][:top_k]
            results.append(
                [
                    ClassificationResult(
                        label=self._id2label.get(int(i), str(int(i))),
                        score=float(probs[i]),
                    )
                    for i in top_indices
                ]
            )
        return ImageClassificationOutput(results=results)
