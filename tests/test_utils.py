# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared encode/decode + image/audio helper utilities (utils.py)."""
from __future__ import annotations

import io

import pytest
from PIL import Image

from task_inference.utils import (
    bytes_to_pil,
    decode_audio,
    decode_image,
    encode_audio,
    encode_image,
    load_audio_bytes,
    load_image_bytes,
    pil_to_bytes,
)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def test_load_image_bytes_from_bytes(sample_image_bytes):
    """load_image_bytes on a bytes input returns the same object."""
    assert load_image_bytes(sample_image_bytes) is sample_image_bytes


def test_load_image_bytes_from_path(sample_image_bytes, tmp_path):
    """load_image_bytes reads the file from a Path object."""
    p = tmp_path / "img.png"
    p.write_bytes(sample_image_bytes)
    assert load_image_bytes(p) == sample_image_bytes


def test_load_image_bytes_from_str_path(sample_image_bytes, tmp_path):
    """load_image_bytes reads the file from a plain string path."""
    p = tmp_path / "img.png"
    p.write_bytes(sample_image_bytes)
    assert load_image_bytes(str(p)) == sample_image_bytes


def test_encode_image_from_bytes(sample_image_bytes):
    """encode_image returns the raw image bytes."""
    b64 = encode_image(sample_image_bytes)
    assert isinstance(b64, bytes)
    assert len(b64) > 0


def test_decode_image_roundtrip(sample_image_bytes):
    """encode → decode roundtrip recovers the original bytes."""
    assert decode_image(encode_image(sample_image_bytes)) == sample_image_bytes


def test_encode_image_from_path(sample_image_bytes, tmp_path):
    """encode_image accepts a Path and produces the same b64 as bytes."""
    p = tmp_path / "img.png"
    p.write_bytes(sample_image_bytes)
    assert encode_image(p) == encode_image(sample_image_bytes)


def test_encode_image_from_str_path(sample_image_bytes, tmp_path):
    """encode_image accepts a str path and produces the same b64 as bytes."""
    p = tmp_path / "img.png"
    p.write_bytes(sample_image_bytes)
    assert encode_image(str(p)) == encode_image(sample_image_bytes)


def test_bytes_to_pil_returns_pil_image(sample_image_bytes):
    img = bytes_to_pil(sample_image_bytes)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)


def test_pil_to_bytes_png(sample_image_bytes):
    img = bytes_to_pil(sample_image_bytes)
    raw = pil_to_bytes(img, fmt="PNG")
    assert isinstance(raw, bytes)
    assert bytes_to_pil(raw).size == img.size


def test_pil_to_bytes_jpeg(sample_image_bytes):
    img = bytes_to_pil(sample_image_bytes).convert("RGB")
    raw = pil_to_bytes(img, fmt="JPEG")
    assert isinstance(raw, bytes)
    assert bytes_to_pil(raw).size == img.size


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def test_load_audio_bytes_from_bytes(sample_audio_bytes):
    """load_audio_bytes on a bytes input returns the same object."""
    assert load_audio_bytes(sample_audio_bytes) is sample_audio_bytes


def test_load_audio_bytes_from_path(sample_audio_bytes, tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(sample_audio_bytes)
    assert load_audio_bytes(p) == sample_audio_bytes


def test_load_audio_bytes_from_str_path(sample_audio_bytes, tmp_path):
    p = tmp_path / "audio.wav"
    p.write_bytes(sample_audio_bytes)
    assert load_audio_bytes(str(p)) == sample_audio_bytes


def test_encode_audio_from_bytes(sample_audio_bytes):
    b64 = encode_audio(sample_audio_bytes)
    assert isinstance(b64, bytes)
    assert len(b64) > 0


def test_decode_audio_roundtrip(sample_audio_bytes):
    assert decode_audio(encode_audio(sample_audio_bytes)) == sample_audio_bytes
