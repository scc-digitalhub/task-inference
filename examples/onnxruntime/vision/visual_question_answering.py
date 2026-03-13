# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: visual question answering with the ONNX Runtime backend.

Default model: dandelin/vilt-b32-finetuned-vqa

Usage
-----
    python examples/onnxruntime/vision/visual_question_answering.py

    # Custom question
    python examples/onnxruntime/vision/visual_question_answering.py "How many people?"

    # Use a pre-exported local ONNX directory
    python examples/onnxruntime/vision/visual_question_answering.py "What color?" --model ./onnx/vilt/

Outputs
-------
- Console: top answers with confidence scores.

Notes
-----
To pre-export manually::

    optimum-cli export onnx --model dandelin/vilt-b32-finetuned-vqa ./onnx/vilt-vqa/
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
    positional = [a for a in args if not a.startswith("--")]
    question = positional[0] if positional else "What is in the image?"
    model_name: str | None = './onnx/vilt-vqa/'

    if "--model" in args:
        idx = args.index("--model") + 1
        if idx < len(args):
            model_name = args[idx]

    image_bytes = load_image("./examples/visual-question-answering.jpg")

    print("Loading ONNX model…")
    if not model_name:
        sys.exit("Error: --model is required. Provide the path to a local ONNX model directory.")
    task = create_task("onnxruntime", "visual-question-answering", model_name=model_name)

    inp = InferenceRequest(
        inputs=[
            RequestInput(name="image", shape=[1], datatype="BYTES", data=[image_bytes]),
            RequestInput(name="question", shape=[1], datatype="BYTES", data=[question]),
        ],
        parameters={"top_k": 3},
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
