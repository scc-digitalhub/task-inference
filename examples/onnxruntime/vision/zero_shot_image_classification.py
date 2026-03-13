# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: zero-shot image classification with the ONNX Runtime backend.

Default model: openai/clip-vit-base-patch32

Usage
-----
    python examples/onnxruntime/vision/zero_shot_image_classification.py

    # Use a pre-exported local ONNX directory
    python examples/onnxruntime/vision/zero_shot_image_classification.py --model ./onnx/clip/

Outputs
-------
- Console: matched labels with confidence scores.

Notes
-----
To pre-export manually::

    optimum-cli export onnx --model openai/clip-vit-base-patch32 ./onnx/clip/
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
    model_name: str | None = "./onnx/clip/"
    if "--model" in args:
        idx = args.index("--model") + 1
        if idx < len(args):
            model_name = args[idx]

    candidate_labels = ["cat", "dog", "bird", "car", "person"]

    image_bytes = load_image("./examples/zero-shot-image-classification.jpg")

    print("Loading ONNX model…")
    task = create_task("onnxruntime", "zero-shot-image-classification", model_name=model_name)

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
            RequestInput(
                name="candidate_labels",
                shape=[len(candidate_labels)],
                datatype="BYTES",
                data=candidate_labels,
            ),
        ],
    )
    print(f"Candidate labels: {candidate_labels}")
    print("Running zero-shot image classification…")
    resp = task(inp)

    labels = (resp.get_output("label").data or [[]])[0]
    scores = (resp.get_output("score").data or [[]])[0]

    print("\nResults (sorted by score):")
    for i, (label, score) in enumerate(zip(labels, scores), 1):
        print(f"  {i}. {label:30s}  {score:.4f}")


if __name__ == "__main__":
    main()
