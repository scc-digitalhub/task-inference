# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: monocular depth estimation with the ONNX Runtime backend.

Default model: Intel/dpt-large

Usage
-----
    python examples/onnxruntime/vision/depth_estimation.py [OUT_DIR]

    # Use a pre-exported local ONNX directory
    python examples/onnxruntime/vision/depth_estimation.py [OUT_DIR] --model ./onnx/dpt/

Outputs
-------
- Console: depth map statistics (min / max / mean) for each input image.
- File:    If OUT_DIR is provided, the PNG depth visualisation is saved to
           ``<OUT_DIR>/depth_vis_<N>.png`` for each image.

Notes
-----
To pre-export manually::

    optimum-cli export onnx --model Intel/dpt-large ./onnx/dpt-large/
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

from task_inference import create_task
from task_inference.protocol.v2 import InferenceRequest, RequestInput


def load_image(path: str) -> bytes:
    if path.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path) as response:
            return response.read()
    return Path(path).read_bytes()


def main() -> None:
    args = sys.argv[1:]
    out_dir: Path | None = None
    model_name: str | None = "./onnx/dpt-large/"

    positional = [a for a in args if not a.startswith("--")]
    if positional:
        out_dir = Path(positional[0])

    if "--model" in args:
        idx = args.index("--model") + 1
        if idx < len(args):
            model_name = args[idx]

    image_bytes = load_image("./examples/depth-estimation.jpg")

    print("Loading ONNX model…")
    task = create_task("onnxruntime", "depth-estimation", model_name=model_name)

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
        ]
    )
    print("Running depth estimation…")
    resp = task(inp)

    predicted_depth = resp.get_output("predicted_depth")
    depth_image = resp.get_output("depth")

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
