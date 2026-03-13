# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: automatic speech recognition with the ONNX Runtime backend.

Default model: openai/whisper-base

Usage
-----
    # Basic transcription
    python examples/onnxruntime/audio/speech_recognition.py

    # Transcription with word-level timestamps
    python examples/onnxruntime/audio/speech_recognition.py --timestamps

    # Force a specific language (ISO-639-1 code)
    python examples/onnxruntime/audio/speech_recognition.py --language fr

    # Use a pre-exported local ONNX directory
    python examples/onnxruntime/audio/speech_recognition.py --model ./onnx/whisper-base/

Outputs
-------
- Console: full transcript and, if --timestamps is set, each chunk with timing.

Notes
-----
On first run the model is automatically exported to ONNX and cached.
To pre-export manually::

    optimum-cli export onnx --model openai/whisper-base ./onnx/whisper-base/
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import urllib.request

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
    args = sys.argv[1:]
    use_timestamps = "--timestamps" in args
    language: str | None = None
    model_name: str | None = './onnx/whisper-base/'

    if "--language" in args:
        lang_idx = args.index("--language") + 1
        if lang_idx < len(args):
            language = args[lang_idx]

    if "--model" in args:
        model_idx = args.index("--model") + 1
        if model_idx < len(args):
            model_name = args[model_idx]

    audio_bytes = convert_to_raw_pcm("./examples/speech-recognition.flac")

    print("Loading ONNX model…")
    task = create_task("onnxruntime", "automatic-speech-recognition", model_name=model_name)

    params: dict = {"return_timestamps": use_timestamps, "sample_rate": SAMPLE_RATE}
    if language:
        params["language"] = language
    inp = InferenceRequest(
        inputs=[
            RequestInput(name="audio", shape=[1], datatype="BYTES", data=[audio_bytes]),
        ],
        parameters=params,
    )
    print("Running ASR…")
    resp = task(inp)

    texts = resp.get_output("text").data or [""]
    print(f"\nTranscription:\n  {texts[0]}")

    if use_timestamps:
        try:
            chunk_texts = resp.get_output("chunks_texts").data or []
            ts_starts = resp.get_output("chunks_ts_start").data or []
            ts_ends = resp.get_output("chunks_ts_end").data or []
            print("\nChunks:")
            for text, start, end in zip(chunk_texts[0], ts_starts[0], ts_ends[0]):
                print(f"  [{start:6.2f}s → {end:6.2f}s]  {text}")
        except KeyError:
            print("  (no timestamp chunks returned)")


if __name__ == "__main__":
    main()
