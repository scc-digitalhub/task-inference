# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - depth estimation (pure ORT, no transformers)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from ....tasks.vision.depth_estimation import DepthEstimationInput, DepthEstimationOutput, DepthEstimationTask
from ..base import OnnxRuntimeTaskMixin
from ..adapters.vision.depth import resolve_depth_adapter


class OnnxDepthEstimationTask(OnnxRuntimeTaskMixin, DepthEstimationTask):
    """Depth estimation using a local ONNX model.

    Dialect is auto-detected from the model's ONNX input/output tensor names.
    Supported dialects:

    * **predicted-depth** — ``pixel_values`` → ``predicted_depth [B, H', W']``
      (DPT-large, DPT-hybrid, GLPN, …)

    Parameters
    ----------
    model_name:
        **Local directory** path to the exported ONNX model.
    device:
        ORT execution device.

    Example
    -------
    ::

        optimum-cli export onnx --model Intel/dpt-large ./onnx/dpt-large/
        task = OnnxDepthEstimationTask(model_name="./onnx/dpt-large/")
    """

    def __init__(self, model_name: str, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        cfg = self._load_config(model_name)
        pp_cfg = self._load_preprocessor_config(model_name)
        providers = self._device_to_providers(device)
        session = self._create_session(self._find_onnx_file(model_name), providers)
        self._adapter = resolve_depth_adapter(session, pp_cfg, cfg)

    @staticmethod
    def _depth_to_vis(depth_np: np.ndarray) -> bytes:
        """Convert a float32 depth map to a grayscale PNG visualisation."""
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max > d_min:
            norm = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(depth_np, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(norm, mode="L").save(buf, format="PNG")
        return buf.getvalue()

    def process(self, inputs: DepthEstimationInput) -> DepthEstimationOutput:
        all_predicted_depth: list[list[float]] = []
        depth_vis_list: list[bytes] = []
        h = w = 0

        for img_bytes in inputs.images:
            pil_image = self._raw_to_pil(img_bytes)
            depth_resized = self._adapter.estimate(pil_image)
            h, w = depth_resized.shape
            all_predicted_depth.append(depth_resized.flatten().tolist())
            depth_vis_list.append(self._depth_to_vis(depth_resized))

        return DepthEstimationOutput(
            predicted_depth=all_predicted_depth,
            width=w,
            height=h,
            depth=depth_vis_list,
        )
