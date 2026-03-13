# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test every ORT task / adapter combination.

For each test case:
  1. Export the model to ONNX if the target directory does not yet contain a
     ``.onnx`` file (can be skipped with ``--skip-export``).
  2. Load the task and run inference using the sample files under
     ``examples/``.
  3. Print PASS / FAIL / SKIP with a one-line summary and a final table.

Adapter combinations exercised
-------------------------------
- image-classification        / pixel-logits                (google/vit-base-patch16-224)
- object-detection            / transformers-detr            (facebook/detr-resnet-50)
- object-detection            / transformers-yolos           (hustvl/yolos-tiny)
- object-detection            / torchvision-detection        (FasterRCNN via torch.onnx.export)
- object-detection            / yolov8                       (Ultralytics YOLOv8n)
- depth-estimation            / predicted-depth              (Intel/dpt-large)
- image-segmentation          / semantic-logits              (nvidia/segformer-b0-finetuned-ade-512-512)
- visual-question-answering   / transformers-vilt            (dandelin/vilt-b32-finetuned-vqa)
- zero-shot-image-classification / transformers-clip         (openai/clip-vit-base-patch32)
- zero-shot-object-detection  / transformers-owlvit          (google/owlvit-base-patch32)
- image-anonymization         / transformers-yolos           (hustvl/yolos-tiny, shared dir)
- audio-classification        / input-values-logits          (superb/wav2vec2-base-superb-ks)
- automatic-speech-recognition / wav2vec2-ctc               (facebook/wav2vec2-base-960h)
- automatic-speech-recognition / whisper-encoder-decoder    (openai/whisper-base)

Usage
-----
Run from the workspace root::

    python examples/onnxruntime/smoke_test.py

    # Skip model exports (use models already present in ./onnx/)
    python examples/onnxruntime/smoke_test.py --skip-export

    # Run only a specific case or dialect (substring match on case name)
    python examples/onnxruntime/smoke_test.py --only yolov8
    python examples/onnxruntime/smoke_test.py --only whisper
    python examples/onnxruntime/smoke_test.py --only vilt

    # Use a custom model cache directory
    python examples/onnxruntime/smoke_test.py --onnx-dir /data/onnx

Dependencies
------------
- ``optimum[exporters]``   — for HuggingFace model export
- ``ffmpeg``               — for audio cases (CLI tool, not a Python package)
- ``torch + torchvision``  — for the torchvision-detection export only
- ``ultralytics``          — for the yolov8 export only
- ``librosa``              — for whisper inference (mel-spectrogram)
"""
from __future__ import annotations

import argparse
import shutil
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
_DEFAULT_ONNX_DIR = _ROOT / "onnx"
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
# Export helpers
# ---------------------------------------------------------------------------

def _optimum_export(hf_model_id: str, model_dir: Path, extra: list[str] | None = None) -> None:
    """Export a HuggingFace model to ONNX with optimum-cli."""
    model_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["optimum-cli", "export", "onnx", "--model", hf_model_id, str(model_dir)]
    if extra:
        cmd += extra
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # optimum-cli exits with code 1 on validation errors even when the model
        # was successfully written; treat the export as successful if at least
        # one .onnx file ended up in model_dir.
        if list(model_dir.glob("*.onnx")):
            return
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


def _export_fasterrcnn(model_dir: Path) -> None:
    """Export TorchVision FasterRCNN-ResNet50-FPN to ONNX.

    Output tensor names are ``pred_boxes``, ``labels``, ``scores`` so the
    ``torchvision-detection`` adapter detects this model automatically.
    """
    import torch
    import torchvision  # type: ignore[import-untyped]

    class _Wrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, ...]:
            # FasterRCNN expects a list of [C, H, W] tensors in [0, 1] range
            dets = self.m([pixel_values[0]])
            return dets[0]["boxes"], dets[0]["labels"], dets[0]["scores"]

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
    model_dir.mkdir(parents=True, exist_ok=True)

    dummy = torch.zeros(1, 3, 800, 800)
    with torch.no_grad():
        torch.onnx.export(
            _Wrapper(model),
            dummy,
            str(model_dir / "model.onnx"),
            input_names=["pixel_values"],
            output_names=["pred_boxes", "labels", "scores"],
            dynamic_axes={
                "pixel_values":  {0: "batch", 2: "height", 3: "width"},
                "pred_boxes":    {0: "num_detections"},
                "labels":        {0: "num_detections"},
                "scores":        {0: "num_detections"},
            },
            opset_version=11,
        )


def _export_vilt(model_dir: Path) -> None:
    """Export VilT for VQA to ONNX via torch.onnx.export.

    ``optimum`` does not support the vilt model type, so we export manually.
    The resulting model has five inputs (``input_ids``, ``attention_mask``,
    ``token_type_ids``, ``pixel_values``, ``pixel_mask``) and one output
    (``logits``), which matches the ``transformers-vilt`` adapter.

    VilT's ``visual_embed`` uses ``torch.multinomial`` with a dynamically-
    computed ``num_samples``.  ONNX requires this argument to be a compile-
    time constant, so we monkey-patch ``visual_embed`` before tracing to call
    ``.item()`` at the right spots and convert tensors to Python ints.
    """
    import types
    import torch
    import torch.nn.functional as F
    from transformers import ViltForQuestionAnswering, ViltProcessor  # type: ignore[import-untyped]
    from transformers.pytorch_utils import meshgrid as _meshgrid  # type: ignore[import-untyped]

    model_id = "dandelin/vilt-b32-finetuned-vqa"
    model = ViltForQuestionAnswering.from_pretrained(model_id).eval()
    processor = ViltProcessor.from_pretrained(model_id)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Save support files that the ORT task needs at inference time.
    model.config.to_json_file(str(model_dir / "config.json"))
    processor.image_processor.to_json_file(str(model_dir / "preprocessor_config.json"))
    processor.tokenizer.save_pretrained(str(model_dir))

    # Patch the saved preprocessor config so that the ORT adapter always
    # resizes to exactly 384×384 — the same shape used by the ONNX export's
    # dummy input.  The original config uses ``shortest_edge=384`` which
    # preserves aspect ratio and produces non-square images; those would
    # not match the exported model's static pixel_mask shape.
    import json as _json
    pp_path = model_dir / "preprocessor_config.json"
    pp = _json.loads(pp_path.read_text())
    pp["size"] = {"height": 384, "width": 384}
    pp["do_pad"] = False
    pp_path.write_text(_json.dumps(pp, indent=2))

    # ------------------------------------------------------------------
    # Monkey-patch ViltEmbeddings.visual_embed so that the call to
    # torch.multinomial gets a Python int as num_samples instead of a
    # tensor (which ONNX can't handle as a dynamic num_samples).
    # The patch is identical to the original code except for the two
    # .item() calls that materialise tensor values into Python ints.
    # ------------------------------------------------------------------
    def _onnx_safe_visual_embed(
        self: object,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor,
        max_image_length: int = 200,
    ) -> tuple:
        x = self.patch_embeddings(pixel_values)  # type: ignore[attr-defined]
        x_mask = pixel_mask[:, None, :, :].float()
        x_mask = F.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        batch_size, num_channels, height, width = x.shape
        patch_dim = self.config.image_size // self.config.patch_size  # type: ignore[attr-defined]
        spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(  # type: ignore[attr-defined]
            1, num_channels, patch_dim, patch_dim
        )
        # Use .item() so that h/w are Python ints — required for
        # interpolate(size=...) and pad() inside ONNX tracing.
        pos_embed = torch.cat(
            [
                F.pad(
                    F.interpolate(
                        spatial_pos,
                        size=(int(h.item()), int(w.item())),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    (0, width - int(w.item()), 0, height - int(h.item())),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        patch_index = torch.stack(
            _meshgrid(
                torch.arange(x_mask.shape[-2]),
                torch.arange(x_mask.shape[-1]),
                indexing="ij",
            ),
            dim=-1,
        ).to(device=x_mask.device)
        patch_index = patch_index[None, None, :, :, :]
        patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
        patch_index = patch_index.flatten(1, 3)
        x_mask = x_mask.flatten(1)

        # KEY FIX: convert the effective-resolution tensor to a Python int so
        # that it becomes an ONNX-safe constant (not a ReduceMax node).
        effective_resolution = x_h * x_w
        eff = int(effective_resolution.max().item())
        if max_image_length < 0 or max_image_length is None or not isinstance(max_image_length, int):
            max_image_length = eff
        else:
            max_image_length = min(eff, max_image_length)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_length - v for v in valid_nums]

        select = []
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                # Take the first max_image_length valid patches in index order.
                # torch.multinomial(…, replacement=False, num_samples > 1) is not
                # supported by the ONNX exporter; torch.arange is a safe replacement
                # and is semantically equivalent when the pixel_mask is all-ones
                # (which is always true for VilT's fixed 384×384 exported model).
                valid_choice = torch.arange(max_image_length)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
                select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
        patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # type: ignore[attr-defined]
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed),  # type: ignore[attr-defined]
            dim=1,
        )
        x = x + pos_embed
        x = self.dropout(x)  # type: ignore[attr-defined]

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)
        return x, x_mask, (patch_index, (height, width))

    model.vilt.embeddings.visual_embed = types.MethodType(
        _onnx_safe_visual_embed, model.vilt.embeddings
    )

    # ------------------------------------------------------------------
    # Dummy inputs sized for the model's default image resolution (384×384).
    # ------------------------------------------------------------------
    dummy_text = processor.tokenizer(
        "What is in the image?",
        return_tensors="pt",
        padding="max_length",
        max_length=40,
        truncation=True,
    )
    dummy_pixel = torch.zeros(1, 3, 384, 384)
    dummy_mask = torch.ones(1, 384, 384, dtype=torch.long)
    dummy_inputs = (
        dummy_text["input_ids"],
        dummy_text["attention_mask"],
        dummy_text.get("token_type_ids", torch.zeros_like(dummy_text["input_ids"])),
        dummy_pixel,
        dummy_mask,
    )

    class _Wrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor,
            pixel_values: torch.Tensor,
            pixel_mask: torch.Tensor,
        ) -> torch.Tensor:
            return self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
            ).logits

    with torch.no_grad():
        torch.onnx.export(
            _Wrapper(model),
            dummy_inputs,
            str(model_dir / "model.onnx"),
            input_names=["input_ids", "attention_mask", "token_type_ids", "pixel_values", "pixel_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids":      {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "token_type_ids": {0: "batch", 1: "seq"},
                "pixel_values":   {0: "batch"},
                "pixel_mask":     {0: "batch"},
                "logits":         {0: "batch"},
            },
            opset_version=14,
        )


def _export_yolov8(model_dir: Path) -> None:
    """Export YOLOv8n to ONNX via the Ultralytics Python API.

    If ``yolov8n.pt`` does not exist inside *model_dir*, Ultralytics
    downloads it automatically from its CDN into its cache directory and
    the export proceeds normally.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]

    model_dir.mkdir(parents=True, exist_ok=True)
    pt_path = model_dir / "yolov8n.pt"
    yolo = YOLO(str(pt_path) if pt_path.exists() else "yolov8n.pt")
    exported = yolo.export(format="onnx", imgsz=640, dynamic=False)
    exported_path = Path(str(exported))
    dest = model_dir / "model.onnx"
    if exported_path.resolve() != dest.resolve() and exported_path.exists():
        shutil.copy2(exported_path, dest)


# ---------------------------------------------------------------------------
# Inference run-functions
# One function per task.  Each returns a short summary string on success;
# raises on failure.
# ---------------------------------------------------------------------------

def _run_image_classification(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("onnxruntime", "image-classification", model_name=str(model_dir))
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "image-classification.jpg").read_bytes()],
        )],
        parameters={"top_k": 3},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"top_label={labels[0]!r}" if labels else "top_label=—"


def _run_object_detection(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("onnxruntime", "object-detection", model_name=str(model_dir))
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "object-detection.jpg").read_bytes()],
        )],
        parameters={"threshold": 0.3},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"detections={len(labels)}"


def _run_depth_estimation(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("onnxruntime", "depth-estimation", model_name=str(model_dir))
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "depth-estimation.jpg").read_bytes()],
        )],
    ))
    depth_flat = (resp.get_output("predicted_depth").data or [[]])[0]
    return f"depth_pixels={len(depth_flat)}"


def _run_image_segmentation(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("onnxruntime", "image-segmentation", model_name=str(model_dir))
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "image-segmentation.jpg").read_bytes()],
        )],
        parameters={"threshold": 0.5, "mask_threshold": 0.3, "overlap_mask_area_threshold": 0.5},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"segments={len(labels)}"


def _run_vqa(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("onnxruntime", "visual-question-answering", model_name=str(model_dir))
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


def _run_zero_shot_classification(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    candidate_labels = ["cat", "dog", "car", "person", "bird"]
    task = create_task("onnxruntime", "zero-shot-image-classification", model_name=str(model_dir))
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


def _run_zero_shot_detection(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    candidate_labels = ["cat", "dog", "person", "car"]
    task = create_task("onnxruntime", "zero-shot-object-detection", model_name=str(model_dir))
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


def _run_image_anonymization(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    task = create_task("onnxruntime", "image-anonymization", model_name=str(model_dir))
    resp = task(InferenceRequest(
        inputs=[RequestInput(
            name="image", shape=[1], datatype="BYTES",
            data=[(_EXAMPLES / "image-anonymization.jpg").read_bytes()],
        )],
        parameters={"strategy": "black_box", "threshold": 0.3},
    ))
    counts = (resp.get_output("num_regions_anonymized").data or [0])[0]
    return f"regions_anonymized={counts}"


def _run_audio_classification(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    audio_bytes = _convert_audio(_EXAMPLES / "audio-classification.flac")
    task = create_task("onnxruntime", "audio-classification", model_name=str(model_dir))
    resp = task(InferenceRequest(
        inputs=[RequestInput(name="audio", shape=[1], datatype="BYTES", data=[audio_bytes])],
        parameters={"top_k": 1, "sample_rate": SAMPLE_RATE},
    ))
    labels = (resp.get_output("label").data or [[]])[0]
    return f"label={labels[0]!r}" if labels else "label=—"


def _run_asr(model_dir: Path) -> str:
    from task_inference import create_task
    from task_inference.protocol.v2 import InferenceRequest, RequestInput

    audio_bytes = _convert_audio(_EXAMPLES / "speech-recognition.flac")
    task = create_task("onnxruntime", "automatic-speech-recognition", model_name=str(model_dir))
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
    """A single task / adapter smoke-test case."""
    name: str                          # "task/dialect"
    model_dir: Path                    # local ONNX model directory
    export_fn: Callable[[Path], None]  # called when model_dir has no .onnx file
    run_fn: Callable[[Path], str]      # called to run inference
    notes: str = ""                    # printed alongside the case name


def _build_cases(onnx_root: Path) -> list[Case]:
    return [
        # ── image-classification ─────────────────────────────────────────
        Case(
            name="image-classification/pixel-logits",
            model_dir=onnx_root / "vit",
            export_fn=lambda d: _optimum_export("google/vit-base-patch16-224", d),
            run_fn=_run_image_classification,
        ),

        # ── object-detection ─────────────────────────────────────────────
        Case(
            name="object-detection/transformers-detr",
            model_dir=onnx_root / "detr",
            export_fn=lambda d: _optimum_export("facebook/detr-resnet-50", d),
            run_fn=_run_object_detection,
        ),
        Case(
            name="object-detection/transformers-yolos",
            model_dir=onnx_root / "yolos-tiny",
            export_fn=lambda d: _optimum_export("hustvl/yolos-tiny", d),
            run_fn=_run_object_detection,
        ),
        Case(
            name="object-detection/torchvision-detection",
            # Use a dedicated directory so the exported model has the correct
            # output names (pred_boxes, labels, scores — no 'logits'), which
            # prevents YolosAdapter from matching before TorchVisionDetectionAdapter.
            # The pre-existing onnx/fasterrcnn/fasterrcnn.onnx uses 'logits' as
            # an output name and therefore maps to the wrong dialect.
            model_dir=onnx_root / "fasterrcnn-ort",
            export_fn=_export_fasterrcnn,
            run_fn=_run_object_detection,
            notes="requires torch + torchvision",
        ),
        Case(
            name="object-detection/yolov8",
            model_dir=onnx_root / "yolo",
            export_fn=_export_yolov8,
            run_fn=_run_object_detection,
            notes="requires ultralytics",
        ),

        # ── depth-estimation ─────────────────────────────────────────────
        Case(
            name="depth-estimation/predicted-depth",
            model_dir=onnx_root / "dpt-large",
            export_fn=lambda d: _optimum_export("Intel/dpt-large", d),
            run_fn=_run_depth_estimation,
        ),

        # ── image-segmentation ───────────────────────────────────────────
        Case(
            name="image-segmentation/semantic-logits",
            model_dir=onnx_root / "segformer",
            export_fn=lambda d: _optimum_export("nvidia/segformer-b0-finetuned-ade-512-512", d),
            run_fn=_run_image_segmentation,
        ),

        # ── visual-question-answering ────────────────────────────────────
        # dandelin/vilt-b32-finetuned-vqa exports with token_type_ids
        # → auto-detected as the `transformers-vilt` dialect.
        # For the `transformers-vilt-no-token-type` dialect, substitute a
        # VilT checkpoint whose ONNX export omits token_type_ids.
        Case(
            name="visual-question-answering/transformers-vilt",
            model_dir=onnx_root / "vilt-vqa",
            export_fn=_export_vilt,
            run_fn=_run_vqa,
            notes="requires torch + transformers",
        ),

        # ── zero-shot-image-classification ───────────────────────────────
        Case(
            name="zero-shot-image-classification/transformers-clip",
            model_dir=onnx_root / "clip",
            export_fn=lambda d: _optimum_export("openai/clip-vit-base-patch32", d),
            run_fn=_run_zero_shot_classification,
        ),

        # ── zero-shot-object-detection ────────────────────────────────────
        Case(
            name="zero-shot-object-detection/transformers-owlvit",
            model_dir=onnx_root / "owlvit",
            export_fn=lambda d: _optimum_export("google/owlvit-base-patch32", d),
            run_fn=_run_zero_shot_detection,
        ),

        # ── image-anonymization ───────────────────────────────────────────
        # Shares the yolos-tiny model directory with the detection case above.
        Case(
            name="image-anonymization/transformers-yolos",
            model_dir=onnx_root / "yolos-tiny",
            export_fn=lambda d: _optimum_export("hustvl/yolos-tiny", d),
            run_fn=_run_image_anonymization,
            notes="shared model dir with object-detection/transformers-yolos",
        ),

        # ── audio-classification ──────────────────────────────────────────
        Case(
            name="audio-classification/input-values-logits",
            model_dir=onnx_root / "wav2vec2-ks",
            export_fn=lambda d: _optimum_export("superb/wav2vec2-base-superb-ks", d),
            run_fn=_run_audio_classification,
            notes="requires ffmpeg",
        ),

        # ── automatic-speech-recognition ─────────────────────────────────
        Case(
            name="automatic-speech-recognition/wav2vec2-ctc",
            model_dir=onnx_root / "wav2vec2",
            export_fn=lambda d: _optimum_export(
                "facebook/wav2vec2-base-960h", d,
                extra=["--task", "automatic-speech-recognition"],
            ),
            run_fn=_run_asr,
            notes="requires ffmpeg",
        ),
        Case(
            name="automatic-speech-recognition/whisper-encoder-decoder",
            model_dir=onnx_root / "whisper-base",
            export_fn=lambda d: _optimum_export("openai/whisper-base", d),
            run_fn=_run_asr,
            notes="requires ffmpeg + librosa",
        ),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _model_ready(model_dir: Path) -> bool:
    """Return True if *model_dir* contains at least one ``.onnx`` file."""
    return model_dir.is_dir() and bool(list(model_dir.glob("*.onnx")))


def _run_case(case: Case, skip_export: bool) -> tuple[str, str]:
    """Execute one test case.

    Returns
    -------
    status : ``"PASS"``, ``"FAIL"``, or ``"SKIP"``
    detail : short summary on PASS, reason on SKIP, traceback on FAIL
    """
    if not _model_ready(case.model_dir):
        if skip_export:
            return "SKIP", f"no .onnx file in {case.model_dir.relative_to(_ROOT)}"
        try:
            print(f"    exporting → {case.model_dir.relative_to(_ROOT)} …")
            case.export_fn(case.model_dir)
        except Exception:
            return "FAIL", traceback.format_exc()

    try:
        detail = case.run_fn(case.model_dir)
        return "PASS", detail
    except Exception:
        return "FAIL", traceback.format_exc()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test all ONNX Runtime task/adapter combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip model export; mark cases with missing models as SKIP.",
    )
    parser.add_argument(
        "--only", metavar="PATTERN",
        help="Run only cases whose name contains PATTERN (case-insensitive substring match).",
    )
    parser.add_argument(
        "--onnx-dir", metavar="DIR", default=str(_DEFAULT_ONNX_DIR),
        help=f"Base directory for cached ONNX models (default: {_DEFAULT_ONNX_DIR}).",
    )
    args = parser.parse_args()

    onnx_root = Path(args.onnx_dir)
    cases = _build_cases(onnx_root)

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
        status, detail = _run_case(case, skip_export=args.skip_export)
        elapsed = time.monotonic() - t0

        if status == "PASS":
            sym = _green("PASS")
            print(f"  → {sym}  ({elapsed:.1f}s)  {detail}")
        elif status == "SKIP":
            sym = _yellow("SKIP")
            print(f"  → {sym}  ({elapsed:.1f}s)  {detail}")
        else:
            sym = _red("FAIL")
            for line in detail.splitlines():
                print(f"      {line}")
            print(f"  → {sym}  ({elapsed:.1f}s)")

        results.append((case.name, status, detail))

    # Summary
    total   = len(results)
    passed  = sum(1 for _, s, _ in results if s == "PASS")
    failed  = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")

    print()
    print(_bold(sep))
    summary = (
        f"  Total {total}  "
        + _green(f"PASS {passed}")
        + "  "
        + (_red(f"FAIL {failed}") if failed else f"FAIL {failed}")
        + "  "
        + (_yellow(f"SKIP {skipped}") if skipped else f"SKIP {skipped}")
    )
    print(summary)

    if failed:
        print()
        print(_bold("  Failed cases:"))
        for name, status, _ in results:
            if status == "FAIL":
                print(f"    • {name}")

    print()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
