# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: image classification with the Transformers backend.

Model: google/vit-base-patch16-224 (default)

Usage
-----
    python examples/transformers/vision/image_classification.py
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
    image_bytes = load_image('./examples/image-classification.jpg')

    print("Loading model…")
    task = create_task("transformers", "image-classification")

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
