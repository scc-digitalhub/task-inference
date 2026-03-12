# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: object detection with the Transformers backend.

Model: facebook/detr-resnet-50 (default)

Usage
-----
    python examples/transformers/vision/object_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from task_inference import create_task
from task_inference.protocol.v2 import InferenceRequest, RequestInput

import timm

def load_image(path: str) -> bytes:
    if path.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path) as response:
            return response.read()
    return Path(path).read_bytes()


def main() -> None:
    image_bytes = load_image('./examples/object-detection.jpg')

    print("Loading model…")
    task = create_task("transformers", "object-detection")

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
        ],
        parameters={"threshold": 0.5},
    )
    print("Running object detection (threshold=0.5)…")
    resp = task(inp)

    labels = resp.get_output("label").data or []
    scores = resp.get_output("score").data or []
    boxes_flat = resp.get_output("box").data or []

    if not labels:
        print("\nNo objects detected above threshold.")
        return

    print(f"\nDetected {len(labels[0])} object(s):")
    for i, (label, score) in enumerate(zip(labels[0], scores[0])):
        xmin, ymin, xmax, ymax = boxes_flat[0][i * 4 : i * 4 + 4]
        print(
            f"  {label:30s}  score={score:.3f}"
            f"  box=({xmin:.0f}, {ymin:.0f}, {xmax:.0f}, {ymax:.0f})"
        )


if __name__ == "__main__":
    main()
