# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for TransformersTaskMixin image/audio static helpers.

No model loading or pipeline calls are required because all tested methods are
pure-Python or Pillow-only wrappers.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image

pytest.importorskip("transformers")  # transitively required by implementations __init__.py

from task_inference.implementations.transformers.base import TransformersTaskMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil(w: int = 8, h: int = 8, color: tuple = (255, 0, 0)) -> Image.Image:
    return Image.new("RGB", (w, h), color=color)


# ---------------------------------------------------------------------------
# Image helpers - require only Pillow
# ---------------------------------------------------------------------------


def test_raw_to_pil(sample_image_bytes):
    img = TransformersTaskMixin._raw_to_pil(sample_image_bytes)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)

def test_pil_to_bytes_returns_bytes():
    raw = TransformersTaskMixin._pil_to_bytes(_make_pil())
    assert isinstance(raw, bytes)
    assert len(raw) > 0


def test_pil_to_bytes_roundtrip():
    img = _make_pil(16, 16, (0, 128, 255))
    raw = TransformersTaskMixin._pil_to_bytes(img, fmt="PNG")
    img2 = Image.open(io.BytesIO(raw))
    assert img2.size == (16, 16)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402 (required dep, always available)


def test_raw_pcm_to_array_structure(sample_audio_bytes):
    result = TransformersTaskMixin._raw_pcm_to_array(sample_audio_bytes, sample_rate=16000)
    assert "array" in result
    assert "sampling_rate" in result
    assert isinstance(result["array"], np.ndarray)
    assert result["array"].ndim == 1
    assert result["array"].dtype == np.float32
    assert result["sampling_rate"] == 16000


def test_raw_pcm_to_array_custom_sample_rate(sample_audio_bytes):
    result = TransformersTaskMixin._raw_pcm_to_array(sample_audio_bytes, sample_rate=8000)
    assert result["sampling_rate"] == 8000
