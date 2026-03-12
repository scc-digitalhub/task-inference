# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: image anonymization with the Transformers backend.

Model: hustvl/yolos-tiny (default)

Usage
-----
    python examples/transformers/vision/image_anonymization.py [OUTPUT_PATH [STRATEGY]]
    
STRATEGY: ``blur`` (default), ``pixelate``, or ``black_box``.

Outputs
-------
- File: the anonymized image saved to OUTPUT_PATH (default: ./output/anonymized.png).
- Console: number of regions anonymized.
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
    output_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./output")) / "anonymized.png"
    strategy_name = sys.argv[2] if len(sys.argv) > 2 else "blur"

    image_bytes = load_image('./examples/image-anonymization.jpg')

    print("Loading model…")
    task = create_task("transformers", "image-anonymization")

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
        ],
        parameters={
            "strategy": strategy_name,
            "blur_radius": 51,
            "threshold": 0.5,
            "classes": ["person", "face", "cat", "dog"],
        },
    )
    print(f"Running image anonymization (strategy={strategy_name})…")
    resp = task(inp)

    anon_images = resp.get_output("anonymized_image").data or []
    counts = resp.get_output("num_regions_anonymized").data or []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(anon_images[0])
    print(f"\nAnonymized {counts[0]} region(s).")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
