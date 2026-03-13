# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: image segmentation with the ONNX Runtime backend.

Default model: nvidia/segformer-b0-finetuned-ade-512-512 (semantic, 150 ADE20K classes)

Note: The ONNX Runtime backend uses ORTModelForSemanticSegmentation, which
supports encoder-based semantic segmentation models.  The ``subtask``
parameter (panoptic / instance) is forwarded to the pipeline but has no
effect for semantic-only models.  Use the ``transformers`` backend for
panoptic and instance segmentation.

Usage
-----
    python examples/onnxruntime/vision/image_segmentation.py

    # Use a pre-exported local ONNX directory
    python examples/onnxruntime/vision/image_segmentation.py --model ./onnx/segformer/

Outputs
-------
- Console: segment labels and confidence scores.

Notes
-----
To pre-export manually::

    optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 ./onnx/segformer/
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
    model_name: str | None = "./onnx/segformer/"
    if "--model" in args:
        idx = args.index("--model") + 1
        if idx < len(args):
            model_name = args[idx]

    image_bytes = load_image("./examples/image-segmentation.jpg")

    print("Loading ONNX model…")
    task = create_task("onnxruntime", "image-segmentation", model_name=model_name)

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
        ],
        parameters={"threshold": 0.9, "mask_threshold": 0.5, "overlap_mask_area_threshold": 0.5},
    )
    print("Running image segmentation (semantic)…")
    resp = task(inp)

    labels = resp.get_output("label").data or []
    scores = resp.get_output("score").data or []
    masks = resp.get_output("mask").data or []

    if not labels or not labels[0]:
        print("\nNo segments returned.")
        return

    print(f"\nFound {len(labels[0])} segment(s):")
    for label, score, mask in zip(labels[0], scores[0], masks[0]):
        mask_kb = len(mask) / 1024
        print(f"  {label:30s}  score={score:.3f}  mask={mask_kb:.1f} KB")


if __name__ == "__main__":
    main()
