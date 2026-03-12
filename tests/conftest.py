# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Return a 64×64 red PNG image as bytes."""
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_audio_bytes() -> bytes:
    """Return 1 second of silence as raw float32 PCM at 16 kHz (little-endian)."""
    return np.zeros(16000, dtype=np.float32).tobytes()


@pytest.fixture()
def sample_audio_sample_rate() -> int:
    """Sample rate matching ``sample_audio_bytes``."""
    return 16000
