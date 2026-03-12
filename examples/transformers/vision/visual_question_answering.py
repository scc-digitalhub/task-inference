# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: visual question answering with the Transformers backend.

Model: dandelin/vilt-b32-finetuned-vqa (default)

Usage
-----
    python examples/transformers/vision/visual_question_answering.py

Outputs
-------
- Console: top answers with confidence scores.
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
    question =  "What is in the image?"

    image_bytes = load_image('./examples/visual-question-answering.jpg')

    print("Loading model…")
    task = create_task("transformers", "visual-question-answering")

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
            RequestInput(name="question", shape=[1], datatype="BYTES", data=[question]),
        ],
        parameters={"top_k": 1},
    )
    print(f'Question: "{question}"')
    print("Running visual question answering…")
    resp = task(inp)

    answers = resp.get_output("answer").data or []
    scores = resp.get_output("score").data or []

    print("\nTop answers:")
    for i, (answer, score) in enumerate(zip(answers[0], scores[0]), 1):
        print(f"  {i}. {answer:30s}  score={score:.4f}")


if __name__ == "__main__":
    main()
