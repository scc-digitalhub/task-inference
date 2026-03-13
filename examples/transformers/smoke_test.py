# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test every Transformers task implementation.

For each test case:
  1. Load the task (model is downloaded from the Hub on first run and cached).
  2. Run inference using the sample files under ``examples/``.
  3. Print PASS / FAIL with a one-line summary and a final table.

Tasks exercised
---------------
- image-classification        (google/vit-base-patch16-224)
- object-detection            (facebook/detr-resnet-50)
- depth-estimation            (Intel/dpt-large)
- image-segmentation          (facebook/mask2former-swin-large-coco-panoptic)
- mask-generation             (facebook/sam-vit-base)
- visual-question-answering   (dandelin/vilt-b32-finetuned-vqa)
- zero-shot-image-classification (openai/clip-vit-base-patch32)
- zero-shot-object-detection  (google/owlvit-base-patch32)
- image-anonymization         (hustvl/yolos-tiny)
- audio-classification        (superb/wav2vec2-base-superb-ks)
- automatic-speech-recognition (openai/whisper-base)

Usage
-----
Run from the workspace root::

    python examples/transformers/smoke_test.py

    # Run only a specific case (substring match on case name)
    python examples/transformers/smoke_test.py --only segmentation
    python examples/transformers/smoke_test.py --only asr

Dependencies
------------
- ``scipy``       — required by the image-segmentation pipeline
- ``torchvision`` — required by the mask-generation pipeline
- ``ffmpeg``      — CLI tool required for audio cases
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]   # workspace root
_EXAMPLES = _ROOT / "examples"
SAMPLE_RATE = 16_000

# ---------------------------------------------------------------------------
# Colour helpers (no external deps)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m" if _USE_COLOUR else s


def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m" if _USE_COLOUR else s


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m" if _USE_COLOUR else s


def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m" if _USE_COLOUR else s


# ---------------------------------------------------------------------------
# Audio helper
# ---------------------------------------------------------------------------

def _convert_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert any audio file to raw float32 PCM via ffmpeg."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(path),
            "-ar", str(sample_rate), "-ac", "1", "-f", "f32le", "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Run-functions — one per task
# Each returns a short summary string on success; raises on failure.
# ---------------------------------------------------------------------------

def _run_image_classification() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "image-classification")
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "image-classification.jpg").read_bytes()],
        )],
        parameters={"top_k": 3},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"top_label={labels[0]!r}" if labels else "top_label=—"


def _run_object_detection() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "object-detection")
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "object-detection.jpg").read_bytes()],
        )],
        parameters={"threshold": 0.5},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"detections={len(labels)}"


def _run_depth_estimation() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "depth-estimation")
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "depth-estimation.jpg").read_bytes()],
        )],
    ))
    depth_flat = (resp.get_output("predicted_depth").data or [[]])[0]
    return f"depth_pixels={len(depth_flat)}"


def _run_image_segmentation() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "image-segmentation")
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "image-segmentation.jpg").read_bytes()],
        )],
        parameters={
            "subtask": "panoptic",
            "threshold": 0.9,
            "mask_threshold": 0.5,
            "overlap_mask_area_threshold": 0.5,
        },
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"segments={len(labels)}"


def _run_mask_generation() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "mask-generation")
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "mask-generation.jpg").read_bytes()],
        )],
        parameters={"pred_iou_thresh": 0.88, "stability_score_thresh": 0.95},
    ))
    masks = resp.get_output("mask").data or []
    return f"masks={len(masks)}"


def _run_vqa() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "visual-question-answering")
    resp = task(InferenceRequest(
        inputs=[
            RequestInput(
                name="image", shape=[1], datatype="BYTES",
                data=[(_EXAMPLES / "visual-question-answering.jpg").read_bytes()],
            ),
            RequestInput(
                name="question", shape=[1], datatype="BYTES",
                data=["What is in the image?"],
            ),
        ],
        parameters={"top_k": 1},
    ))
    answers = (resp.get_output("answer").data or [[]])[0]
    return f"answer={answers[0]!r}" if answers else "answer=—"


def _run_zero_shot_classification() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    candidate_labels = ["cat", "dog", "car", "person", "bird"]
    task = create_task("transformers", "zero-shot-image-classification")
    resp = task(InferenceRequest(
        inputs=[
            RequestInput(
                name="image", shape=[1], datatype="BYTES",
                data=[(_EXAMPLES / "zero-shot-image-classification.jpg").read_bytes()],
            ),
            RequestInput(
                name="candidate_labels",
                shape=[len(candidate_labels)],
                datatype="BYTES",
                data=candidate_labels,
            ),
        ],
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"top={labels[0]!r}" if labels else "top=—"


def _run_zero_shot_detection() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    candidate_labels = ["cat", "dog", "person", "car"]
    task = create_task("transformers", "zero-shot-object-detection")
    resp = task(InferenceRequest(
        inputs=[
            RequestInput(
                name="image", shape=[1], datatype="BYTES",
                data=[(_EXAMPLES / "zero-shot-object-detection.jpg").read_bytes()],
            ),
            RequestInput(
                name="candidate_labels",
                shape=[len(candidate_labels)],
                datatype="BYTES",
                data=candidate_labels,
            ),
        ],
        parameters={"threshold": 0.1},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"detections={len(labels)}"


def _run_image_anonymization() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("transformers", "image-anonymization")
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "image-anonymization.jpg").read_bytes()],
        )],
        parameters={"strategy": "black_box", "threshold": 0.3},
    ))
    counts = (resp.get_output("num_regions_anonymized").data or [0])[0]
    return f"regions_anonymized={counts}"


def _run_audio_classification() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    audio_bytes = _convert_audio(_EXAMPLES / "audio-classification.flac")
    task = create_task("transformers", "audio-classification")
    resp = task(InferenceRequest(
        inputs=[RequestInput(name="audio", shape=[1], datatype="BYTES", data=[audio_bytes])],
        parameters={"top_k": 1, "sample_rate": SAMPLE_RATE},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"label={labels[0]!r}" if labels else "label=—"


def _run_asr() -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    audio_bytes = _convert_audio(_EXAMPLES / "speech-recognition.flac")
    task = create_task("transformers", "automatic-speech-recognition")
    resp = task(InferenceRequest(
        inputs=[RequestInput(name="audio", shape=[1], datatype="BYTES", data=[audio_bytes])],
        parameters={"return_timestamps": False, "sample_rate": SAMPLE_RATE},
    ))
    text = ((resp.get_output("text").data or [""])[0] or "").replace("\n", " ")
    snippet = text[:60] + ("…" if len(text) > 60 else "")
    return f"transcript={snippet!r}"


# ---------------------------------------------------------------------------
# Test-case definition
# ---------------------------------------------------------------------------

@dataclass
class Case:
    """A single task smoke-test case."""
    name: str                      # task name
    run_fn: Callable[[], str]      # called to run inference; returns summary string
    notes: str = ""                # printed alongside the case name


def _build_cases() -> list[Case]:
    return [
        # ── vision ───────────────────────────────────────────────────────
        Case(
            name="image-classification",
            run_fn=_run_image_classification,
        ),
        Case(
            name="object-detection",
            run_fn=_run_object_detection,
        ),
        Case(
            name="depth-estimation",
            run_fn=_run_depth_estimation,
        ),
        Case(
            name="image-segmentation",
            run_fn=_run_image_segmentation,
            notes="requires scipy",
        ),
        Case(
            name="mask-generation",
            run_fn=_run_mask_generation,
            notes="requires torchvision",
        ),
        Case(
            name="visual-question-answering",
            run_fn=_run_vqa,
        ),
        Case(
            name="zero-shot-image-classification",
            run_fn=_run_zero_shot_classification,
        ),
        Case(
            name="zero-shot-object-detection",
            run_fn=_run_zero_shot_detection,
        ),
        Case(
            name="image-anonymization",
            run_fn=_run_image_anonymization,
        ),
        # ── audio ────────────────────────────────────────────────────────
        Case(
            name="audio-classification",
            run_fn=_run_audio_classification,
            notes="requires ffmpeg",
        ),
        Case(
            name="automatic-speech-recognition",
            run_fn=_run_asr,
            notes="requires ffmpeg",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_case(case: Case) -> tuple[str, str]:
    """Execute one test case.

    Returns
    -------
    status : ``"PASS"`` or ``"FAIL"``
    detail : short summary on PASS, traceback on FAIL
    """
    try:
        detail = case.run_fn()
        return "PASS", detail
    except Exception:
        return "FAIL", traceback.format_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test all Transformers task implementations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only", metavar="PATTERN",
        help="Run only cases whose name contains PATTERN (case-insensitive substring match).",
    )
    args = parser.parse_args()

    cases = _build_cases()

    if args.only:
        pattern = args.only.lower()
        cases = [c for c in cases if pattern in c.name.lower()]
        if not cases:
            sys.exit(f"No cases matched --only {args.only!r}")

    col_w = max(len(c.name) for c in cases) + 2
    sep = "─" * (col_w + 58)

    print()
    print(_bold(sep))
    print(_bold(f"  {'Case':<{col_w}}  {'Status':<6}  Detail"))
    print(_bold(sep))

    results: list[tuple[str, str, str]] = []

    for case in cases:
        note_str = f"  ({case.notes})" if case.notes else ""
        print(f"\n  {case.name}{note_str}")

        t0 = time.monotonic()
        status, detail = _run_case(case)
        elapsed = time.monotonic() - t0

        if status == "PASS":
            sym = _green("PASS")
            print(f"  → {sym}  ({elapsed:.1f}s)  {detail}")
        else:
            sym = _red("FAIL")
            print(f"  → {sym}  ({elapsed:.1f}s)")
            for line in detail.splitlines():
                print(f"      {line}")

        results.append((case.name, status, detail))

    # ── summary table ───────────────────────────────────────────────────
    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")

    print()
    print(_bold(sep))
    total_str = f"Total {len(results)}"
    pass_str = _green(f"PASS {n_pass}")
    fail_str = _red(f"FAIL {n_fail}") if n_fail else f"FAIL {n_fail}"
    print(_bold(f"  {total_str}  {pass_str}  {fail_str}"))

    failed = [(n, d) for n, s, d in results if s == "FAIL"]
    if failed:
        print()
        print("  Failed cases:")
        for name, _ in failed:
            print(f"    • {name}")

    print()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
