# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime zero-shot object-detection dialect adapters.

Supported dialects
------------------
transformers-owlvit
    OWL-ViT (Contrastive Vision-Transformer for open-vocabulary detection).
    Inputs: ``input_ids [N, 16]``, ``attention_mask [N, 16]``,
    ``pixel_values [1, 3, H, W]``; outputs ``logits [1, P, N]`` (sigmoid
    scores per patch per label) and ``pred_boxes [1, P, 4]`` in normalised
    ``(cx, cy, w, h)`` format.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from ..base import OnnxDialectAdapter, resolve_adapter, io_names_from_session
from ...base import OnnxRuntimeTaskMixin as _M
from .....tasks.vision.object_detection import BoundingBox, DetectedObject


# ---------------------------------------------------------------------------
# Per-task abstract base
# ---------------------------------------------------------------------------

class ZeroShotDetectionAdapter(OnnxDialectAdapter):
    """Abstract adapter for zero-shot object-detection tasks."""

    def __init__(self, session: Any, pp_cfg: dict[str, Any], tokenizer: Any) -> None:
        self._session = session
        self._pp_cfg = pp_cfg
        self._tokenizer = tokenizer

    @abstractmethod
    def detect_zero_shot(
        self,
        pil_image: Image.Image,
        candidate_labels: list[str],
        threshold: float,
    ) -> list[DetectedObject]:
        """Run inference and return detections above *threshold*."""


# ---------------------------------------------------------------------------
# Dialect: transformers-owlvit
# ---------------------------------------------------------------------------

class OwlVitAdapter(ZeroShotDetectionAdapter):
    """OWL-ViT dialect adapter.

    Inputs : ``input_ids [N, 16]``, ``attention_mask [N, 16]``,
             ``pixel_values [1, 3, H, W]``
    Outputs: ``logits [1, P, N]`` (sigmoid), ``pred_boxes [1, P, 4]``
             (normalised ``cx, cy, w, h``)
    """

    DIALECT: ClassVar[str] = "transformers-owlvit"
    _CTX = 16  # OWL-ViT text context length

    def __init__(self, session: Any, pp_cfg: dict[str, Any], tokenizer: Any) -> None:
        super().__init__(session, pp_cfg, tokenizer)
        tokenizer.enable_truncation(max_length=self._CTX)
        pad_id = tokenizer.token_to_id("<|endoftext|>") or 0
        tokenizer.enable_padding(
            pad_id=pad_id, pad_token="<|endoftext|>", length=self._CTX
        )

    @classmethod
    def accepts(cls, input_names, output_names, config) -> bool:
        return (
            "input_ids" in input_names
            and "attention_mask" in input_names
            and "pixel_values" in input_names
            and "logits" in output_names
            and "pred_boxes" in output_names
        )

    def detect_zero_shot(
        self,
        pil_image: Image.Image,
        candidate_labels: list[str],
        threshold: float,
    ) -> list[DetectedObject]:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        orig_w, orig_h = pil_image.size

        encs = self._tokenizer.encode_batch(candidate_labels)
        input_ids = np.array([e.ids for e in encs], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        out = self._session.run(
            ["logits", "pred_boxes"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            },
        )
        scores = _M._sigmoid(out[0][0])  # [P, N]
        pred_boxes = out[1][0]            # [P, 4]

        best_label_ids = np.argmax(scores, axis=-1)                           # [P]
        best_scores = scores[np.arange(len(scores)), best_label_ids]          # [P]

        cx = pred_boxes[:, 0]; cy = pred_boxes[:, 1]
        w  = pred_boxes[:, 2]; h  = pred_boxes[:, 3]

        dets = []
        for i in range(len(best_scores)):
            score = float(best_scores[i])
            if score < threshold:
                continue
            label_idx = int(best_label_ids[i])
            dets.append(
                DetectedObject(
                    label=candidate_labels[label_idx],
                    score=score,
                    box=BoundingBox(
                        xmin=float(np.clip((cx[i] - w[i] / 2) * orig_w, 0, orig_w)),
                        ymin=float(np.clip((cy[i] - h[i] / 2) * orig_h, 0, orig_h)),
                        xmax=float(np.clip((cx[i] + w[i] / 2) * orig_w, 0, orig_w)),
                        ymax=float(np.clip((cy[i] + h[i] / 2) * orig_h, 0, orig_h)),
                    ),
                )
            )
        return dets


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_ADAPTERS: list[type[ZeroShotDetectionAdapter]] = [
    OwlVitAdapter,
]


def resolve_zero_shot_detection_adapter(
    session: Any,
    pp_cfg: dict[str, Any],
    config: dict[str, Any],
    tokenizer: Any,
) -> ZeroShotDetectionAdapter:
    """Detect the dialect from *session* and return an instantiated adapter."""
    input_names, output_names = io_names_from_session(session)
    adapter_cls = resolve_adapter(_ADAPTERS, input_names, output_names, config)
    return adapter_cls(session, pp_cfg, tokenizer)
