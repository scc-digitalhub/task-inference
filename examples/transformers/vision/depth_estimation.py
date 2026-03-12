# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: monocular depth estimation with the Transformers backend.

Model: Intel/dpt-large (default)

Usage
-----
    python examples/transformers/vision/depth_estimation.py [OUT_DIR]

Outputs
-------
- Console: depth map statistics (min / max / mean) for each input image.
- File:    If OUT_DIR is provided, the PNG depth visualisation is saved to
           ``<OUT_DIR>/depth_vis_<N>.png`` for each image.
"""
from __future__ import annotations

import sys
from pathlib import Path

from task_inference import create_task
from task_inference.protocol.v2 import InferenceRequest, RequestInput
from task_inference.tasks.vision.depth_estimation import DepthEstimationInput


def load_image(path: str) -> bytes:
    if path.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path) as response:
            return response.read()
    return Path(path).read_bytes()


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    image_bytes = load_image('./examples/depth-estimation.jpg')

    print("Loading model…")
    task = create_task("transformers", "depth-estimation")

    # The task accepts a batch of images
    inp = InferenceRequest(
        inputs=[
            RequestInput(
                name="image",
                shape=[1, len(image_bytes)],
                datatype="BYTES",
                data=[image_bytes],
            ),
        ]
    )

    print("Running depth estimation…")
    resp = task(inp)
    predicted_depth = resp.get_output("predicted_depth")
    depth_image = resp.get_output("depth")

    import statistics

    for idx, depth_flat in enumerate(predicted_depth.data or []):
        mn = min(depth_flat)
        mx = max(depth_flat)
        mean = statistics.mean(depth_flat)
        print(
            f"\nImage {idx}: {predicted_depth.shape[1]}x{predicted_depth.shape[2]} px"
            f"  |  depth min={mn:.3f}  max={mx:.3f}  mean={mean:.3f}"
        )

    if out_dir and depth_image:
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, vis_bytes in enumerate(depth_image.data or []):
            out_path = out_dir / f"depth_vis_{idx}.png"
            out_path.write_bytes(vis_bytes)
            print(f"Depth visualisation saved to {out_path}")


if __name__ == "__main__":
    main()
