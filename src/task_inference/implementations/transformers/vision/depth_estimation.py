# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - depth estimation."""
from __future__ import annotations

import numpy as np
from transformers import pipeline

from ....tasks.vision.depth_estimation import DepthEstimationInput, DepthEstimationOutput, DepthEstimationTask
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "Intel/dpt-large"


class TransformersDepthEstimationTask(TransformersTaskMixin, DepthEstimationTask):
    """Depth estimation using a HuggingFace ``depth-estimation`` pipeline.

    Parameters
    ----------
    model_name:
        Any ``depth-estimation`` compatible model on the Hub.
        Defaults to ``Intel/dpt-large``.
    device:
        Inference device.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | int = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline(
            "depth-estimation",
            model=model_name,
            device=device,
        )

    def process(self, inputs: DepthEstimationInput) -> DepthEstimationOutput:
        pil_images = [self._raw_to_pil(img) for img in inputs.images]
        results = self._pipe(pil_images)

        all_predicted_depth: list[list[float]] = []
        depth_vis_list: list[bytes] = []
        h = w = 0

        for result in results:
            depth_tensor = result.get("predicted_depth")
            if depth_tensor is not None:
                depth_np: np.ndarray = np.array(depth_tensor)
            else:
                depth_np = np.array(result["depth"], dtype=np.float32)

            h, w = int(depth_np.shape[0]), int(depth_np.shape[1])
            all_predicted_depth.append(depth_np.flatten().tolist())

            depth_vis = result.get("depth")  # PIL Image or None
            if depth_vis is not None:
                depth_vis_list.append(self._pil_to_bytes(depth_vis))

        return DepthEstimationOutput(
            predicted_depth=all_predicted_depth,
            width=w,
            height=h,
            depth=depth_vis_list if depth_vis_list else None,
        )
