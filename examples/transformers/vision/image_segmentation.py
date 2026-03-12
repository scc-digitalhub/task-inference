# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: image segmentation with the Transformers backend.

Model: facebook/mask2former-swin-large-coco-panoptic (default)

Usage
-----
    python examples/transformers/vision/image_segmentation.py [SUBTASK]

SUBTASK: optional - one of ``panoptic``, ``semantic``, ``instance``.
         Omit to let the model use its default.

Outputs
-------
- Console: segment labels and confidence scores.
"""
from __future__ import annotations

import sys
from pathlib import Path

from task_inference import create_task
from task_inference.protocol.v2 import InferenceRequest, RequestInput

import scipy

def load_image(path: str) -> bytes:
    if path.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path) as response:
            return response.read()
    return Path(path).read_bytes()


def main() -> None:
    subtask = 'panoptic'

    image_bytes = load_image('./examples/image-segmentation.jpg')

    print("Loading model…")
    task = create_task("transformers", "image-segmentation")

    params: dict = {"threshold": 0.9, "mask_threshold": 0.5, "overlap_mask_area_threshold": 0.5}
    if subtask is not None:
        params["subtask"] = subtask
    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
        ],
        parameters=params,
    )
    subtitle = f" (subtask={subtask})" if subtask else ""
    print(f"Running image segmentation{subtitle}…")
    resp = task(inp)

    labels = resp.get_output("label").data or []
    scores = resp.get_output("score").data or []
    masks = resp.get_output("mask").data or []

    if not labels:
        print("\nNo segments returned.")
        return

    for label, score, mask in zip(labels, scores, masks):
        print(f"\nFound {len(label)} segment(s):")
        for idx in range(len(label)):  
            mask_kb = len(mask[idx]) / 1024
            print(f"  {label[idx]:30s}  score={score[idx]:.3f}  mask={mask_kb:.1f} KB")


if __name__ == "__main__":
    main()
