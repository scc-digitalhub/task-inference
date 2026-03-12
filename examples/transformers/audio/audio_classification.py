# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: audio classification with the Transformers backend.

Model: superb/wav2vec2-base-superb-ks (keyword spotting, default)
       MIT/ast-finetuned-audioset-10-10-0.4593 for environmental sounds

Usage
-----
    python examples/transformers/audio/audio_classification.py 


Outputs
-------
- Console: top predicted audio classes with confidence scores.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from task_inference import create_task
from task_inference.protocol.v2 import InferenceRequest, RequestInput

SAMPLE_RATE = 16000


def convert_to_raw_pcm(path: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert any audio file or URL to raw float32 PCM via ffmpeg."""
    if path.startswith(("http://", "https://")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            urllib.request.urlretrieve(path, tmp.name)
            input_path = tmp.name
    else:
        input_path = path

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    audio_bytes = convert_to_raw_pcm("./examples/audio-classification.flac")

    print("Loading model…")
    task = create_task("transformers", "audio-classification")

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="audio", shape=[1], datatype="BYTES", data=[audio_bytes]),
        ],
        parameters={"top_k": 5, "sample_rate": SAMPLE_RATE},
    )
    print("Running audio classification (top_k=5)…")
    resp = task(inp)

    labels = resp.get_output("label").data or []
    scores = resp.get_output("score").data or []

    print("\nTop predictions:")
    for i, (label, score) in enumerate(zip(labels[0], scores[0]), 1):
        print(f"  {i}. {label:40s}  {score:.4f}")


if __name__ == "__main__":
    main()
