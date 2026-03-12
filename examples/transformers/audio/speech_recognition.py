# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Example: automatic speech recognition with the Transformers backend.

Model: openai/whisper-base (default)

Usage
-----
    # Basic transcription
    python examples/transformers/audio/speech_recognition.py 

    # Transcription with word-level timestamps
    python examples/transformers/audio/speech_recognition.py --timestamps

    # Force a specific language (ISO-639-1 code)
    python examples/transformers/audio/speech_recognition.py --language fr

Outputs
-------
- Console: full transcript and, if --timestamps is set, each chunk with timing.
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
    if "--language" in args:
        lang_idx = args.index("--language") + 1
        if lang_idx < len(args):
            language = args[lang_idx]

    audio_bytes = convert_to_raw_pcm("./examples/speech-recognition.flac")

    print("Loading model…")
    task = create_task("transformers", "automatic-speech-recognition")

    params: dict = {"return_timestamps": use_timestamps, "sample_rate": SAMPLE_RATE}
    if language:
        params["language"] = language
    inp = InferenceRequest(
        inputs=[
            RequestInput(name="audio", shape=[1], datatype="BYTES", data=[audio_bytes]),
        ],
        parameters=params,
    )
    opts = []
    if use_timestamps:
        opts.append("timestamps")
    if language:
        opts.append(f"language={language}")
    print(f"Running ASR{' (' + ', '.join(opts) + ')' if opts else ''}…")
    resp = task(inp)

    text = (resp.get_output("text").data or [""])[0]
    print(f"\nTranscript: {text!r}")

    if use_timestamps:
        try:
            chunk_texts = resp.get_output("chunk_texts").data or []
            ts_starts = resp.get_output("chunk_ts_start").data or []
            ts_ends = resp.get_output("chunk_ts_end").data or []
            if chunk_texts:
                print(f"\nChunks ({len(chunk_texts[0])}):")
                for ct, ts, te in zip(chunk_texts[0], ts_starts[0], ts_ends[0]):
                    print(f"  [{ts:.2f}s → {te:.2f}s] {ct!r}")
        except KeyError:
            pass


if __name__ == "__main__":
    main()

