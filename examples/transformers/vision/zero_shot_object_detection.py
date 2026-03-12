# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: zero-shot object detection with the Transformers backend.

Model: google/owlvit-base-patch32 (default)

Usage
-----
    python examples/transformers/vision/zero_shot_object_detection.py

Outputs
-------
- Console: detected objects with labels, confidence scores, and bounding boxes.
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
    candidate_labels = ["cat", "dog", "person", "car", "bicycle", "bird", "parrot"]

    image_bytes = load_image('./examples/zero-shot-object-detection.jpg')

    print("Loading model…")
    task = create_task("transformers", "zero-shot-object-detection")

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
            RequestInput(name="candidate_labels", shape=[len(candidate_labels)], datatype="BYTES", data=candidate_labels),
        ],
        parameters={"threshold": 0.1},
    )
    print(f"Candidate labels: {candidate_labels}")
    print("Running zero-shot object detection (threshold=0.1)…")
    resp = task(inp)

    labels = resp.get_output("label").data or []
    scores = resp.get_output("score").data or []
    boxes_flat = resp.get_output("box").data or []

    if not labels or not labels[0]:
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
