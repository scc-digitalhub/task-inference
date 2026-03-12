# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: mask generation (SAM-style) with the Transformers backend.

Model: facebook/sam-vit-base (default)

Usage
-----
    # Automatic mask generation (segment everything)
    python examples/transformers/vision/mask_generation.py [OUT_DIR]

Outputs
-------
- Console: number of masks generated and per-mask scores.
- File:    If OUT_DIR is provided, each mask PNG is saved as mask_<N>.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

from task_inference import create_task
from task_inference.protocol.v2 import InferenceRequest, RequestInput

import torchvision

def load_image(path: str) -> bytes:
    if path.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(path) as response:
            return response.read()
    return Path(path).read_bytes()


def main() -> None:
    image_bytes = load_image('./examples/mask-generation.jpg')
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    print("Loading model…")
    task = create_task("transformers", "mask-generation")

    request_inputs = [
        RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
    ]
    print("Using automatic mode (segment everything).")

    inp = InferenceRequest(
        inputs=request_inputs,
        parameters={"pred_iou_thresh": 0.88, "stability_score_thresh": 0.95},
    )
    print("Running mask generation…")
    resp = task(inp)

    masks = resp.get_output("mask").data or []
    scores = resp.get_output("score").data or []

    if not masks:
        print("\nNo masks generated.")
        return

    print(f"\nGenerated {len(masks[0])} mask(s):")
    for i, (mask, score) in enumerate(zip(masks[0], scores[0]), 1):
        print(f"  Mask {i:3d}  score={score:.4f}  size={len(mask) / 1024:.1f} KB")

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, mask in enumerate(masks[0], 1):
            out_path = out_dir / f"mask_{i:03d}.png"
            out_path.write_bytes(mask)
        print(f"\nSaved {len(masks[0])} mask(s) to {out_dir}/")


if __name__ == "__main__":
    main()
