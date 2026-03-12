# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Shared base class for all HuggingFace Transformers task implementations."""
from __future__ import annotations

import io
from typing import Any

from PIL import Image

from ...utils import bytes_to_pil


class TransformersTaskMixin:
    """Mixin providing common utilities for Transformers-backed task implementations.

    Attributes
    ----------
    model_name : str
        HuggingFace model identifier used when loading the pipeline.
    device : str | int
        Device string (``'cpu'``, ``'cuda'``, ``'mps'``) or device index.
    """

    model_name: str
    device: str | int

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    
    @staticmethod
    def _raw_to_pil(data: bytes) -> Image.Image:
        return bytes_to_pil(data)

    @staticmethod
    def _pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_pcm_to_array(raw: bytes, sample_rate: int) -> dict[str, Any]:
        """Convert raw float32 PCM bytes to a dict understood by transformers pipelines.

        Parameters
        ----------
        raw:
            Little-endian float32 PCM samples.
        sample_rate:
            Sample rate in Hz.

        Returns
        -------
        dict
            ``{"array": np.ndarray, "sampling_rate": int}``
        """
        import numpy as np

        data = np.frombuffer(raw, dtype=np.float32)
        return {"array": data, "sampling_rate": sample_rate}
