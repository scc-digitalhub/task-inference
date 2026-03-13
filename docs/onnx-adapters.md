# ONNX Runtime Adapters

The ONNX Runtime backend uses an **adapter pattern** to decouple each task's inference logic from the specific tensor contract (`dialect`) of the underlying ONNX model.  When a task is instantiated, the correct adapter is selected automatically by inspecting the model's input and output tensor names — no manual configuration is required.

- [Architecture](#architecture)
- [Task adapters reference](#task-adapters-reference)
  - [Image Classification](#image-classification)
  - [Object Detection](#object-detection)
  - [Depth Estimation](#depth-estimation)
  - [Image Segmentation](#image-segmentation)
  - [Visual Question Answering](#visual-question-answering)
  - [Zero-Shot Image Classification](#zero-shot-image-classification)
  - [Zero-Shot Object Detection](#zero-shot-object-detection)
  - [Audio Classification](#audio-classification)
  - [Automatic Speech Recognition](#automatic-speech-recognition)
- [Adding a new adapter](#adding-a-new-adapter)

---

## Architecture

```
OnnxDialectAdapter  (base.py)
│
├── ImageClassificationAdapter
│   └── PixelLogitsAdapter            pixel-logits
│
├── ObjectDetectionAdapter
│   ├── DetrAdapter                   transformers-detr
│   ├── YolosAdapter                  transformers-yolos
│   ├── TorchVisionDetectionAdapter   torchvision-detection
│   └── YoloV8Adapter                 yolov8
│
├── DepthEstimationAdapter
│   └── PredictedDepthAdapter         predicted-depth
│
├── ImageSegmentationAdapter
│   └── SemanticLogitsAdapter         semantic-logits
│
├── VQAAdapter
│   ├── ViltAdapter                   transformers-vilt
│   └── ViltNoTokenTypeAdapter        transformers-vilt-no-token-type
│
├── ZeroShotClassificationAdapter
│   └── ClipAdapter                   transformers-clip
│
├── ZeroShotDetectionAdapter
│   └── OwlVitAdapter                 transformers-owlvit
│
├── AudioClassificationAdapter
│   └── InputValuesLogitsAdapter      input-values-logits
│
└── ASRAdapter
    ├── Wav2Vec2CTCAdapter             wav2vec2-ctc
    └── WhisperAdapter                 whisper-encoder-decoder
```

### How dialect detection works

Every adapter implements `accepts(input_names, output_names, config) → bool`.  The per-task factory function (`resolve_*_adapter`) calls `accepts` on each registered adapter in order (most-specific first), and instantiates the first match.

```python
# src/task_inference/implementations/onnxruntime/adapters/base.py

def resolve_adapter(candidates, input_names, output_names, config) -> Type[T]:
    for adapter_cls in candidates:
        if adapter_cls.accepts(input_names, output_names, config):
            return adapter_cls
    raise ValueError(...)
```

Ordering matters: more-specific dialects must appear **before** catch-all ones in the `_ADAPTERS` list.

### Key base-class API

```python
class OnnxDialectAdapter(ABC):
    DIALECT: ClassVar[str]         # unique dialect identifier string

    @classmethod
    @abstractmethod
    def accepts(
        cls,
        input_names:  Sequence[str],
        output_names: Sequence[str],
        config:       dict[str, Any],
    ) -> bool: ...
```

Each per-task abstract subclass adds the task-specific inference method (e.g. `classify`, `detect`, `transcribe`).

---

## Task adapters reference

### Image Classification

**Module:** `adapters/vision/classification.py`  
**Factory:** `resolve_classification_adapter(session, pp_cfg, config) → ImageClassificationAdapter`  
**Inference method:** `adapter.classify(pil_image) → np.ndarray[num_classes]` — raw logits

#### `pixel-logits` — `PixelLogitsAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `pixel_values` | `[B, 3, H, W]` | normalised by preprocessor |
| Output | `logits`        | `[B, num_classes]` | raw pre-softmax scores |

**Detection rule:** `"pixel_values" in inputs AND "logits" in outputs`

**Covered models:** ViT, ResNet (transformers + torchvision), EfficientNet, ConvNeXt, Swin Transformer, DenseNet, …

**Export examples:**

```bash
# HuggingFace / Optimum
optimum-cli export onnx --model google/vit-base-patch16-224 ./onnx/vit/
optimum-cli export onnx --model microsoft/resnet-50         ./onnx/resnet/
```

```python
# TorchVision (torch.onnx.export)
import torch, torchvision
model = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()

class Wrapper(torch.nn.Module):
    def forward(self, pixel_values):
        return self.m(pixel_values)
    def __init__(self, m): super().__init__(); self.m = m

torch.onnx.export(
    Wrapper(model),
    torch.zeros(1, 3, 224, 224),
    "resnet50.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

---

### Object Detection

**Module:** `adapters/vision/detection.py`  
**Factory:** `resolve_detection_adapter(session, pp_cfg, config) → ObjectDetectionAdapter`  
**Inference method:** `adapter.detect(pil_image, threshold, id2label) → list[DetectedObject]`

Detection order: `DetrAdapter` → `YolosAdapter` → `TorchVisionDetectionAdapter` → `YoloV8Adapter`

#### `transformers-detr` — `DetrAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `images`     | `[B, 3, H, W]` | |
| Output | `logits`     | `[B, Q, C+1]`  | last column = background class |
| Output | `pred_boxes` | `[B, Q, 4]`    | normalised `(cx, cy, w, h)` |

**Detection rule:** `"images" in inputs AND "logits" in outputs AND "pred_boxes" in outputs`

```bash
optimum-cli export onnx --model facebook/detr-resnet-50        ./onnx/detr/
optimum-cli export onnx --model microsoft/conditional-detr-resnet-50 ./onnx/conditional-detr/
```

#### `transformers-yolos` — `YolosAdapter`

Same output contract as DETR; input is named `pixel_values`.

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `pixel_values` | `[B, 3, H, W]` | |
| Output | `logits`        | `[B, Q, C+1]`  | last column = background |
| Output | `pred_boxes`    | `[B, Q, 4]`    | normalised `(cx, cy, w, h)` |

**Detection rule:** `"pixel_values" in inputs AND "logits" in outputs AND "pred_boxes" in outputs`

```bash
optimum-cli export onnx --model hustvl/yolos-tiny ./onnx/yolos-tiny/
```

#### `torchvision-detection` — `TorchVisionDetectionAdapter`

TorchVision-style models with explicit per-detection outputs (no batch dimension on outputs).

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `pixel_values` or `images` | `[B, 3, H, W]` | |
| Output | `pred_boxes` or `boxes`    | `[N, 4]`       | pixel `(xmin, ymin, xmax, ymax)` |
| Output | `labels`                   | `[N]`          | integer class ids (1-based COCO convention) |
| Output | `scores` or `logits`       | `[N]`          | per-detection confidence |

**Detection rule:** `("pred_boxes" OR "boxes") in outputs AND "labels" in outputs`

```python
import torch, torchvision

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()

class FasterRCNNWrapper(torch.nn.Module):
    def forward(self, pixel_values):
        out = self.m([img[0] for img in pixel_values])
        return out[0]["boxes"], out[0]["labels"], out[0]["scores"]
    def __init__(self, m): super().__init__(); self.m = m

torch.onnx.export(
    FasterRCNNWrapper(model),
    torch.zeros(1, 3, 800, 800),
    "fasterrcnn.onnx",
    input_names=["pixel_values"],
    output_names=["pred_boxes", "labels", "scores"],
    opset_version=17,
)
```

#### `yolov8` — `YoloV8Adapter`

Catch-all for Ultralytics YOLOv8/v9/v10 and compatible single-output YOLO exports.  Letterbox preprocessing and greedy NMS (IoU ≥ 0.45) are applied internally.

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `images`   | `[B, 3, H, W]` | normalised to `[0,1]`, letterbox padded |
| Output | *(any)*    | `[B, 4+C, N]` or `[B, N, 4+C]` | layout auto-detected from shape |

Box coordinates are `(cx, cy, w, h)` in letterbox-image pixel space; the adapter un-letterboxes them to original image coordinates.

**Detection rule:** `"images" in inputs AND single output AND "pred_boxes" NOT in outputs AND "logits" NOT in outputs`

```bash
# Ultralytics CLI
yolo export model=yolov8n.pt format=onnx imgsz=640

# Python
from ultralytics import YOLO
YOLO("yolov8n.pt").export(format="onnx", imgsz=640, dynamic=False)
```

---

### Depth Estimation

**Module:** `adapters/vision/depth.py`  
**Factory:** `resolve_depth_adapter(session, pp_cfg, config) → DepthEstimationAdapter`  
**Inference method:** `adapter.estimate(pil_image) → np.ndarray[H, W]` — float32 depth map, resized to original image dimensions

#### `predicted-depth` — `PredictedDepthAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `pixel_values`    | `[B, 3, H, W]` | |
| Output | `predicted_depth` | `[B, H', W']`  | float32; bilinearly resized to original H×W |

**Detection rule:** `"pixel_values" in inputs AND "predicted_depth" in outputs`

```bash
optimum-cli export onnx --model Intel/dpt-large           ./onnx/dpt-large/
optimum-cli export onnx --model Intel/dpt-hybrid-midas    ./onnx/dpt-hybrid/
optimum-cli export onnx --model vinvino02/glpn-nyu        ./onnx/glpn/
```

---

### Image Segmentation

**Module:** `adapters/vision/segmentation.py`  
**Factory:** `resolve_segmentation_adapter(session, pp_cfg, config) → ImageSegmentationAdapter`  
**Inference method:** `adapter.segment(pil_image) → np.ndarray[C, H, W]` — per-class logits, bilinearly upsampled to original image dimensions

#### `semantic-logits` — `SemanticLogitsAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `pixel_values` | `[B, 3, H, W]` | |
| Output | `logits`        | `[B, C, H', W']` | per-class logits; `H'` may be `H/4` (stride-4) or full-resolution |

**Detection rule:** `"pixel_values" in inputs AND "logits" in outputs`

> **Note:** The YOLOS dialect also uses `pixel_values` + `logits`, but is detected first via the `pred_boxes` output; semantic segmentation models never produce `pred_boxes`.

```bash
optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 ./onnx/segformer/
optimum-cli export onnx --model nvidia/mit-b0                              ./onnx/mit-b0/
```

```python
# TorchVision DeepLabV3  (torch.onnx.export)
import torch, torchvision

model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").eval()

class DeepLabWrapper(torch.nn.Module):
    def forward(self, pixel_values):
        return self.m(pixel_values)["out"]
    def __init__(self, m): super().__init__(); self.m = m

torch.onnx.export(
    DeepLabWrapper(model),
    torch.zeros(1, 3, 520, 520),
    "deeplabv3.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

---

### Visual Question Answering

**Module:** `adapters/vision/vqa.py`  
**Factory:** `resolve_vqa_adapter(session, pp_cfg, config, tokenizer) → VQAAdapter`  
**Inference method:** `adapter.answer(pil_image, question) → np.ndarray[num_labels]` — raw logits over the label vocabulary

#### `transformers-vilt` — `ViltAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `input_ids`      | `[1, seq_len]` | tokenized question |
| Input  | `attention_mask` | `[1, seq_len]` | |
| Input  | `token_type_ids` | `[1, seq_len]` | segment ids |
| Input  | `pixel_values`   | `[1, 3, H, W]` | |
| Input  | `pixel_mask`     | `[1, H, W]`    | 1 for valid pixels |
| Output | `logits`         | `[1, num_labels]` | |

**Detection rule:** all five input names present (including `token_type_ids` and `pixel_mask`)

#### `transformers-vilt-no-token-type` — `ViltNoTokenTypeAdapter`

Same as above but without `token_type_ids` (some fine-tuned checkpoints omit it).

**Detection rule:** `pixel_mask` in inputs AND `token_type_ids` NOT in inputs

> **Note:** `optimum` does not support the `vilt` model type for ONNX export.
> Use `torch.onnx.export` directly (see `examples/onnxruntime/smoke_test.py` →
> `_export_vilt()`).  Key points:
>
> - Monkey-patch `ViltEmbeddings.visual_embed` to call `.item()` on dynamic
>   tensor values so that `torch.multinomial`'s `num_samples` argument becomes
>   a Python int (ONNX requires a compile-time constant).
> - Replace `torch.multinomial(…, replacement=False, num_samples > 1)` with
>   `torch.arange(max_image_length)` — the ONNX exporter does not support that
>   variant of multinomial.
> - Save `preprocessor_config.json` with `size = {"height": 384, "width": 384}`
>   (not `shortest_edge`) so the adapter always produces exactly 384 × 384
>   pixels, matching the model's static `pixel_mask` shape.

---

### Zero-Shot Image Classification

**Module:** `adapters/vision/zero_shot_classification.py`  
**Factory:** `resolve_zero_shot_classification_adapter(session, pp_cfg, config, tokenizer) → ZeroShotClassificationAdapter`  
**Inference method:** `adapter.classify_zero_shot(pil_image, labels) → np.ndarray[num_labels]` — softmax probabilities

#### `transformers-clip` — `ClipAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `input_ids`       | `[L, seq_len]` | tokenized candidate labels |
| Input  | `attention_mask`  | `[L, seq_len]` | |
| Input  | `pixel_values`    | `[1, 3, H, W]` | |
| Output | `logits_per_image`| `[1, L]`       | cosine-similarity logit per label |

**Detection rule:** `"input_ids" in inputs AND "attention_mask" in inputs AND "pixel_values" in inputs AND "logits_per_image" in outputs`

```bash
optimum-cli export onnx --model openai/clip-vit-base-patch32   ./onnx/clip/
optimum-cli export onnx --model openai/clip-vit-large-patch14  ./onnx/clip-large/
```

---

### Zero-Shot Object Detection

**Module:** `adapters/vision/zero_shot_detection.py`  
**Factory:** `resolve_zero_shot_detection_adapter(session, pp_cfg, config, tokenizer) → ZeroShotDetectionAdapter`  
**Inference method:** `adapter.detect_zero_shot(pil_image, labels, threshold) → list[DetectedObject]`

#### `transformers-owlvit` — `OwlVitAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `input_ids`      | `[L, seq_len]` | tokenized candidate labels |
| Input  | `attention_mask` | `[L, seq_len]` | |
| Input  | `pixel_values`   | `[1, 3, H, W]` | |
| Output | `logits`         | `[1, P, L]`    | sigmoid score per patch per label |
| Output | `pred_boxes`     | `[1, P, 4]`    | normalised `(cx, cy, w, h)` per patch |

**Detection rule:** `"input_ids" in inputs AND "attention_mask" in inputs AND "pixel_values" in inputs AND "logits" in outputs AND "pred_boxes" in outputs`

```bash
optimum-cli export onnx --model google/owlvit-base-patch32  ./onnx/owlvit/
optimum-cli export onnx --model google/owlvit-large-patch14 ./onnx/owlvit-large/
```

---

### Audio Classification

**Module:** `adapters/audio/classification.py`  
**Factory:** `resolve_audio_classification_adapter(session, do_normalize, config) → AudioClassificationAdapter`  
**Inference method:** `adapter.classify_audio(audio_1d) → np.ndarray[num_labels]` — raw logits

#### `input-values-logits` — `InputValuesLogitsAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `input_values` | `[1, seq_len]` | float32 PCM, optionally mean/std normalised |
| Output | `logits`        | `[1, num_labels]` | |

Normalisation (`do_normalize`) is read from `preprocessor_config.json` and applied inside the adapter.

**Detection rule:** `"input_values" in inputs AND "logits" in outputs`

```bash
optimum-cli export onnx --model superb/wav2vec2-base-superb-ks ./onnx/wav2vec2-ks/
optimum-cli export onnx --model MIT/ast-finetuned-audioset-10-10-0.4593 ./onnx/ast/
```

---

### Automatic Speech Recognition

**Module:** `adapters/audio/asr.py`  
**Factory:** `resolve_asr_adapter(model_dir, providers, config) → ASRAdapter`  
**Inference method:** `adapter.transcribe(audio_1d, language, return_timestamps) → tuple[str, list[ASRChunk] | None]`

The ASR factory is special: it checks for the Whisper dialect **before** creating any ORT session (by inspecting `config.json` → `model_type` and the presence of `encoder_model.onnx`) to avoid loading the wrong file.

#### `wav2vec2-ctc` — `Wav2Vec2CTCAdapter`

| Tensor | Name | Shape | Notes |
|--------|------|-------|-------|
| Input  | `input_values` | `[1, seq_len]` | float32 PCM, mean/std normalised |
| Output | `logits`        | `[1, T, vocab_size]` | CTC frame logits |

Decoding: greedy argmax → CTC collapse (blank = token 0) → vocabulary lookup via `tokenizer.json` / `vocab.json`.

**Detection rule:** `"input_values" in inputs AND "logits" in outputs` (and model is not Whisper)

```bash
optimum-cli export onnx --model facebook/wav2vec2-base-960h --task automatic-speech-recognition ./onnx/wav2vec2/
optimum-cli export onnx --model facebook/hubert-large-ls960-ft --task automatic-speech-recognition ./onnx/hubert/
```

#### `whisper-encoder-decoder` — `WhisperAdapter`

Two-session architecture — the encoder and decoder are each a separate ONNX file.

| File | Tensor | Name | Shape |
|------|--------|------|-------|
| `encoder_model.onnx` | Input  | `input_features` | `[B, n_mels, T]` |
| `encoder_model.onnx` | Output | `last_hidden_state` | `[B, T, hidden]` |
| `decoder_model.onnx` | Input  | `input_ids` | `[B, seq]` |
| `decoder_model.onnx` | Input  | `encoder_hidden_states` | `[B, T, hidden]` |
| `decoder_model.onnx` | Output | `logits` | `[B, seq, vocab]` |

Log-mel spectrogram computation requires `librosa` (`pip install librosa`).  Greedy token-by-token decoding with BPE text decoding via the `tokenizers` package.

**Detection rule:** `config.json` → `model_type == "whisper"` OR `encoder_model.onnx` exists in model directory

```bash
optimum-cli export onnx --model openai/whisper-base  ./onnx/whisper-base/
optimum-cli export onnx --model openai/whisper-small ./onnx/whisper-small/
optimum-cli export onnx --model openai/whisper-large-v3 ./onnx/whisper-large-v3/
```

---

## Adding a new adapter

### Step 1 — Identify the dialect

Inspect the ONNX model's I/O tensor names:

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
print("inputs: ", [i.name for i in session.get_inputs()])
print("outputs:", [o.name for o in session.get_outputs()])
```

Or use `Netron` (https://netron.app) for a visual overview.

### Step 2 — Choose the right adapter module

| Task | Module | Per-task base class |
|------|--------|---------------------|
| Image classification | `adapters/vision/classification.py` | `ImageClassificationAdapter` |
| Object detection | `adapters/vision/detection.py` | `ObjectDetectionAdapter` |
| Depth estimation | `adapters/vision/depth.py` | `DepthEstimationAdapter` |
| Image segmentation | `adapters/vision/segmentation.py` | `ImageSegmentationAdapter` |
| Visual question answering | `adapters/vision/vqa.py` | `VQAAdapter` |
| Zero-shot image classification | `adapters/vision/zero_shot_classification.py` | `ZeroShotClassificationAdapter` |
| Zero-shot object detection | `adapters/vision/zero_shot_detection.py` | `ZeroShotDetectionAdapter` |
| Audio classification | `adapters/audio/classification.py` | `AudioClassificationAdapter` |
| Automatic speech recognition | `adapters/audio/asr.py` | `ASRAdapter` |

### Step 3 — Implement the adapter class

```python
from typing import Any, ClassVar
import numpy as np
from PIL import Image

from ..base import OnnxDialectAdapter
from ...base import OnnxRuntimeTaskMixin as _M


class MyNewDetectionAdapter(ObjectDetectionAdapter):
    """One-line description of the dialect."""

    DIALECT: ClassVar[str] = "my-new-dialect"  # unique identifier

    # --- Constructor (call super().__init__ or initialise custom state) ---
    def __init__(self, session: Any, pp_cfg: dict[str, Any]) -> None:
        super().__init__(session, pp_cfg)
        # store any extra state derived from session metadata
        self._extra = session.get_outputs()[0].name

    # --- Detection rule (pure, no side effects) ---
    @classmethod
    def accepts(
        cls,
        input_names:  list[str],
        output_names: list[str],
        config:       dict[str, Any],
    ) -> bool:
        return "my_input" in input_names and "my_output" in output_names

    # --- Task-specific inference method ---
    def detect(
        self,
        pil_image: Image.Image,
        threshold: float,
        id2label:  dict[int, str],
    ) -> list[DetectedObject]:
        pixel_values, _ = _M._preprocess_image_from_config(pil_image, self._pp_cfg)
        raw = self._session.run([self._extra], {"my_input": pixel_values})
        # ... postprocess raw output into list[DetectedObject] ...
        return detections
```

Key rules:
- `DIALECT` must be a unique string (used in error messages).
- `accepts` must be a **pure classmethod** — no I/O, no side effects.
- Use `_M._preprocess_image_from_config` (and other `OnnxRuntimeTaskMixin` helpers) for image preprocessing to stay consistent with the rest of the backend.
- Return the domain object defined in `tasks/` (e.g. `DetectedObject`, `BoundingBox`), not raw arrays.

### Step 4 — Register the adapter

Add the new class to the `_ADAPTERS` list in the relevant module, **before** any more-generic catch-all:

```python
# adapters/vision/detection.py

_ADAPTERS: list[type[ObjectDetectionAdapter]] = [
    DetrAdapter,
    YolosAdapter,
    TorchVisionDetectionAdapter,
    MyNewDetectionAdapter,   # ← add here, before the catch-all
    YoloV8Adapter,           # catch-all stays last
]
```

### Step 5 — Add an `accepts` guard test

Add a unit test in `tests/test_implementations/` to verify that the detection rule fires correctly and does not misfire on other dialects:

```python
def test_my_new_detection_adapter_accepts():
    assert MyNewDetectionAdapter.accepts(["my_input"], ["my_output"], {})

def test_my_new_detection_adapter_does_not_steal_detr():
    assert not MyNewDetectionAdapter.accepts(["images"], ["logits", "pred_boxes"], {})
```

### Step 6 — Export a test model

Follow the export instructions for your chosen model family (see the relevant section above for examples), place the artefacts in `onnx/<model-name>/`, and add an integration test or an entry to `examples/export/onnx/export_all.py`.
