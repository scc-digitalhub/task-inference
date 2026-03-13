# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Shared base mixin for all pure ONNX Runtime task implementations.

This module has **no dependency** on ``transformers`` or ``optimum``.

Dependencies
------------
- ``onnxruntime`` (required)
- ``numpy`` (required, already a core dep)
- ``Pillow`` (required, already a core dep)
- ``tokenizers`` (optional, needed for text-input tasks: VQA, CLIP, OWL-ViT, Whisper ASR)
- ``librosa`` (optional, needed only for Whisper ASR mel-spectrogram)

Models must be pre-exported as local ONNX directories, e.g. via
``optimum-cli export onnx --model <hub-id> ./local-dir/``.
"""
from __future__ import annotations

import io
import json
import pathlib
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# ORT execution-provider helpers
# ---------------------------------------------------------------------------

def _device_to_providers(device: str | int) -> list[str]:
    """Map a device specifier to an ORT execution-providers list."""
    if isinstance(device, int):
        return [("CUDAExecutionProvider", {"device_id": device}), "CPUExecutionProvider"]
    d = str(device).lower()
    if d == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if d in ("mps", "coreml"):
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Base mixin
# ---------------------------------------------------------------------------

class OnnxRuntimeTaskMixin:
    """Mixin for pure ONNX Runtime task implementations.

    Models must be available as local directories containing at minimum:

    * An ONNX model file (``model.onnx`` or similar)
    * ``config.json`` — model configuration including ``id2label``
    * ``preprocessor_config.json`` — image/audio preprocessing parameters
    * ``tokenizer.json`` or ``vocab.json`` — for text-input tasks

    No ``transformers`` or ``optimum`` dependency is required at inference time.
    Export models once with::

        optimum-cli export onnx --model <hub-id> ./my_model_dir/

    then instantiate the task with the local path::

        task = OnnxImageClassificationTask(model_name="./my_model_dir/")
    """

    model_name: str   # local directory path
    device: str | int

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _device_to_providers(device: str | int) -> list[str]:
        return _device_to_providers(device)

    # ------------------------------------------------------------------
    # ONNX file discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_onnx_file(model_dir: str, preferred: str = "model.onnx") -> str:
        """Locate the main ONNX model file inside *model_dir*."""
        p = pathlib.Path(model_dir)
        for name in [preferred, "model_optimized.onnx"]:
            candidate = p / name
            if candidate.exists():
                return str(candidate)
        found = sorted(p.glob("*.onnx"))
        if found:
            return str(found[0])
        raise FileNotFoundError(f"No ONNX model file found in {model_dir!r}")

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json(model_dir: str, *filenames: str) -> dict[str, Any]:
        """Return the first of *filenames* that exists in *model_dir*."""
        p = pathlib.Path(model_dir)
        for name in filenames:
            path = p / name
            if path.exists():
                return json.loads(path.read_text())
        return {}

    @classmethod
    def _load_config(cls, model_dir: str) -> dict[str, Any]:
        return cls._load_json(model_dir, "config.json")

    @classmethod
    def _load_preprocessor_config(cls, model_dir: str) -> dict[str, Any]:
        return cls._load_json(
            model_dir,
            "preprocessor_config.json",
            "feature_extractor_config.json",
        )

    # ------------------------------------------------------------------
    # Session creation
    # ------------------------------------------------------------------

    @staticmethod
    def _create_session(onnx_path: str, providers: list[str]) -> Any:
        """Create an ``onnxruntime.InferenceSession`` from *onnx_path*."""
        import onnxruntime as ort  # noqa: PLC0415
        return ort.InferenceSession(onnx_path, providers=providers)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_to_pil(data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    @staticmethod
    def _pil_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        return buf.getvalue()

    @staticmethod
    def _resize_image(pil_image: Image.Image, size_cfg: dict[str, Any] | int) -> Image.Image:
        """Resize *pil_image* according to a ``preprocessor_config.json`` size spec.

        Supports ``{"height": H, "width": W}``, ``{"shortest_edge": N}``, and
        a plain integer (square resize).
        """
        if isinstance(size_cfg, int):
            return pil_image.resize((size_cfg, size_cfg), Image.BILINEAR)
        if "height" in size_cfg and "width" in size_cfg:
            return pil_image.resize((int(size_cfg["width"]), int(size_cfg["height"])), Image.BILINEAR)
        if "shortest_edge" in size_cfg:
            target = int(size_cfg["shortest_edge"])
            orig_w, orig_h = pil_image.size
            scale = target / min(orig_w, orig_h)
            return pil_image.resize((int(orig_w * scale), int(orig_h * scale)), Image.BILINEAR)
        return pil_image.resize((224, 224), Image.BILINEAR)

    @staticmethod
    def _center_crop(pil_image: Image.Image, crop_size: dict[str, Any] | int) -> Image.Image:
        """Centre-crop *pil_image* to *crop_size*.

        Supports ``{"height": H, "width": W}`` and a plain integer (square crop).
        """
        if isinstance(crop_size, int):
            crop_h = crop_w = crop_size
        else:
            crop_h = int(crop_size.get("height", crop_size.get("shortest_edge", 224)))
            crop_w = int(crop_size.get("width", crop_size.get("shortest_edge", 224)))
        img_w, img_h = pil_image.size
        left = max(0, (img_w - crop_w) // 2)
        top  = max(0, (img_h - crop_h) // 2)
        return pil_image.crop((left, top, left + crop_w, top + crop_h))

    @classmethod
    def _preprocess_image_from_config(
        cls,
        pil_image: Image.Image,
        pp_cfg: dict[str, Any],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Pre-process a PIL image using ``preprocessor_config.json`` parameters.

        Returns
        -------
        pixel_values : np.ndarray
            Shape ``[1, 3, H, W]``, dtype float32.
        (orig_w, orig_h) : tuple[int, int]
            Original image dimensions before resizing.
        """
        orig_w, orig_h = pil_image.size

        if pp_cfg.get("do_resize", True):
            pil_image = cls._resize_image(pil_image, pp_cfg.get("size", 224))

        if pp_cfg.get("do_center_crop", False):
            pil_image = cls._center_crop(pil_image, pp_cfg.get("crop_size", pp_cfg.get("size", 224)))

        arr = np.array(pil_image, dtype=np.float32)

        if pp_cfg.get("do_rescale", True):
            arr *= float(pp_cfg.get("rescale_factor", 1.0 / 255.0))

        if pp_cfg.get("do_normalize", True):
            mean = np.array(pp_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
            std = np.array(pp_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
            arr = (arr - mean) / std

        # HWC → CHW, add batch dim
        arr = arr.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]
        return arr, (orig_w, orig_h)

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_pcm_to_numpy(raw: bytes) -> np.ndarray:
        """Convert raw float32 little-endian PCM bytes to a 1-D float32 array."""
        return np.frombuffer(raw, dtype=np.float32).copy()

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))
