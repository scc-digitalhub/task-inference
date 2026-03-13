# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: image classification with the ONNX Runtime backend.

Default model: google/vit-base-patch16-224

Usage
-----
    python examples/onnxruntime/vision/image_classification.py

    # Use a pre-exported local ONNX directory
    python examples/onnxruntime/vision/image_classification.py --model ./onnx/vit/

Notes
-----
To pre-export manually::

    optimum-cli export onnx --model google/vit-base-patch16-224 ./onnx/vit/
"""
from __future__ import annotations

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
    model_name: str | None = "./onnx/vit/"
    if "--model" in args:
        idx = args.index("--model") + 1
        if idx < len(args):
            model_name = args[idx]

    image_bytes = load_image("./examples/image-classification.jpg")

    print("Loading ONNX model…")
    task = create_task("onnxruntime", "image-classification", model_name=model_name)

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
        ],
        parameters={"top_k": 5},
    )
    print("Running image classification (top_k=5)…")
    resp = task(inp)

    labels = (resp.get_output("label").data or [[]])[0]
    scores = (resp.get_output("score").data or [[]])[0]

    print("\nTop predictions:")
    for i, (label, score) in enumerate(zip(labels, scores), 1):
        print(f"  {i}. {label:40s}  {score:.4f}")


if __name__ == "__main__":
    main()
