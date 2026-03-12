# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Shared encode/decode helpers used by task schemas and implementations."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

ImageLike = bytes | Path | str
"""Accepted image sources: raw bytes, a filesystem path, or a URL string."""


def load_image_bytes(source: ImageLike) -> bytes:
    """Return raw image bytes regardless of the *source* type.

    * ``bytes`` - returned as-is.
    * :class:`~pathlib.Path` / filesystem string - the file is read.
    """
    if isinstance(source, bytes):
        return source
    path = Path(source)
    return path.read_bytes()


def encode_image(source: ImageLike) -> str:
    """Return a bytes suitable for an OIP v2 ``BYTES`` tensor element."""
    return load_image_bytes(source) 


def decode_image(s: str) -> bytes:
    """Decode OIP v2 ``BYTES`` tensor element back to raw image bytes."""
    return s


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Open image bytes as a :class:`~PIL.Image.Image`."""
    return Image.open(io.BytesIO(image_bytes))


def pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialise a :class:`~PIL.Image.Image` to bytes in the given *fmt*."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

AudioLike = bytes | Path | str
"""Accepted audio sources: raw bytes or a filesystem path."""


def load_audio_bytes(source: AudioLike) -> bytes:
    """Return raw audio bytes regardless of the *source* type."""
    if isinstance(source, bytes):
        return source
    return Path(source).read_bytes()


def encode_audio(source: AudioLike) -> str:
    """Return a string suitable for an OIP v2 ``BYTES`` tensor element."""
    return load_audio_bytes(source) 


def decode_audio(b64: str) -> bytes:
    """Decode a OIP v2 ``BYTES`` tensor element back to raw audio bytes."""
    return b64 
