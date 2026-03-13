<!--
SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler

SPDX-License-Identifier: Apache-2.0
-->

# `examples/onnxruntime/`

Runnable example scripts for all tasks supported by the `onnxruntime` backend.
This backend uses **[ONNX Runtime](https://onnxruntime.ai/) only** — no
`torch`, no `transformers`, no `optimum` required at inference time.

---

## Why ONNX Runtime?

| Feature | Transformers backend | ONNX Runtime backend |
|---|---|---|
| Model format | PyTorch weights | ONNX (`.onnx` graph) |
| Runtime dependency | `torch` (large) | `onnxruntime` (lightweight) |
| Inference speed | Baseline | Faster (ORT graph optimizations) |
| GPU support | CUDA via PyTorch | CUDA via `onnxruntime-gpu` |
| Model source | Hub ID or local | **Local directory only** |
| `transformers` required | Yes | **No** |

---

## Prerequisites

Install the library with the `onnxruntime` extra from the workspace root:

```bash
pip install -e ".[onnxruntime]"
```

For GPU inference, additionally install the GPU variant:

```bash
pip install onnxruntime-gpu
```

---

## Exporting models to ONNX

All models **must be pre-exported** to a local ONNX directory before use.
Use the [Optimum CLI](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
to export any supported HuggingFace model (one-time operation):

```bash
# Install Optimum exporters (once, not required at inference time)
pip install "optimum[exporters]"

# Vision models
optimum-cli export onnx --model google/vit-base-patch16-224       ./onnx/vit/
optimum-cli export onnx --model facebook/detr-resnet-50            ./onnx/detr/
optimum-cli export onnx --model hustvl/yolos-tiny                  ./onnx/yolos-tiny/
optimum-cli export onnx --model Intel/dpt-large                    ./onnx/dpt-large/
optimum-cli export onnx --model nvidia/segformer-b0-finetuned-ade-512-512 ./onnx/segformer/
optimum-cli export onnx --model dandelin/vilt-b32-finetuned-vqa    ./onnx/vilt-vqa/
optimum-cli export onnx --model openai/clip-vit-base-patch32       ./onnx/clip/
optimum-cli export onnx --model google/owlvit-base-patch32         ./onnx/owlvit/

# Audio models
optimum-cli export onnx --model superb/wav2vec2-base-superb-ks     ./onnx/wav2vec2-ks/
optimum-cli export onnx --model openai/whisper-base                ./onnx/whisper-base/
```

Each exported directory will contain `model.onnx` (or multiple ONNX files for
encoder-decoder models), `config.json`, `preprocessor_config.json`, and
tokenizer files — everything needed for pure-ORT inference.

Then pass the local path via `--model`:

```bash
python examples/onnxruntime/vision/image_classification.py --model ./onnx/vit/
```

> **Note:** The `--model` argument is **required** for all ONNX Runtime example
> scripts.  There are no Hub defaults because the backend does not access the
> network at inference time.

---

## Vision tasks

> All vision examples look for a sample image at `./examples/<task-name>.jpg`.
> Edit the script to use your own image.

### Image Classification

```bash
python examples/onnxruntime/vision/image_classification.py --model ./onnx/vit/
```

---

### Object Detection

```bash
python examples/onnxruntime/vision/object_detection.py --model ./onnx/detr/
```

---

### Depth Estimation

```bash
# Stats only
python examples/onnxruntime/vision/depth_estimation.py --model ./onnx/dpt-large/

# Stats + save depth visualisation PNG
python examples/onnxruntime/vision/depth_estimation.py --model ./onnx/dpt-large/ ./output/
```

---

### Image Segmentation

> **Note:** The ONNX Runtime backend supports encoder-based **semantic**
> segmentation only.  For **panoptic** or **instance** segmentation use the
> `transformers` backend.

```bash
python examples/onnxruntime/vision/image_segmentation.py --model ./onnx/segformer/
```

---

### Image Anonymization

```bash
# Blur strategy (default)
python examples/onnxruntime/vision/image_anonymization.py --model ./onnx/yolos-tiny/ ./output/ blur

# Pixelate
python examples/onnxruntime/vision/image_anonymization.py --model ./onnx/yolos-tiny/ ./output/ pixelate

# Black box
python examples/onnxruntime/vision/image_anonymization.py --model ./onnx/yolos-tiny/ ./output/ black_box
```

---

### Visual Question Answering

```bash
python examples/onnxruntime/vision/visual_question_answering.py --model ./onnx/vilt-vqa/

# Custom question
python examples/onnxruntime/vision/visual_question_answering.py --model ./onnx/vilt-vqa/ "How many cats are there?"
```

---

### Zero-Shot Image Classification

```bash
python examples/onnxruntime/vision/zero_shot_image_classification.py --model ./onnx/clip/
```

---

### Zero-Shot Object Detection

```bash
python examples/onnxruntime/vision/zero_shot_object_detection.py --model ./onnx/owlvit/
```

---

## Audio tasks

> Audio tasks require `soundfile`.  WAV files at **16 kHz mono** are
> recommended.

### Audio Classification

```bash
python examples/onnxruntime/audio/audio_classification.py --model ./onnx/wav2vec2-ks/
```

---

### Speech Recognition (ASR)

Whisper models require `librosa` for mel-spectrogram computation (included in
the `onnxruntime` extra).  The encoder and decoder ONNX files are loaded
automatically from the model directory.

```bash
# Basic transcription
python examples/onnxruntime/audio/speech_recognition.py --model ./onnx/whisper-base/

# With timestamps
python examples/onnxruntime/audio/speech_recognition.py --model ./onnx/whisper-base/ --timestamps

# Force language
python examples/onnxruntime/audio/speech_recognition.py --model ./onnx/whisper-base/ --language fr
```

For CTC-based ASR (e.g. wav2vec2):

```bash
optimum-cli export onnx --model facebook/wav2vec2-base-960h ./onnx/wav2vec2-asr/
python examples/onnxruntime/audio/speech_recognition.py --model ./onnx/wav2vec2-asr/
```

---

## Task coverage

| Task | ONNX Runtime backend | Preprocessing | Notes |
|---|---|---|---|
| image-classification | ✅ | PIL + config | ViT, ResNet, EfficientNet |
| object-detection | ✅ | PIL + config | DETR, YOLOS (cx cy w h outputs) |
| depth-estimation | ✅ | PIL + config | DPT, GLPN |
| image-segmentation | ✅ | PIL + config | SegFormer (semantic only) |
| image-anonymization | ✅ | PIL + config | OD model + PIL redaction |
| visual-question-answering | ✅ | PIL + `tokenizers` | ViLT |
| zero-shot-image-classification | ✅ | PIL + `tokenizers` | CLIP |
| zero-shot-object-detection | ✅ | PIL + `tokenizers` | OWL-ViT |
| audio-classification | ✅ | numpy PCM | wav2vec2, HuBERT |
| automatic-speech-recognition | ✅ | numpy PCM / `librosa` mel | CTC (wav2vec2) and Whisper |
| mask-generation | ❌ | — | SAM ONNX export not supported; use `transformers` backend |


---

## Why ONNX Runtime?

| Feature | Transformers backend | ONNX Runtime backend |
|---|---|---|
| Model format | PyTorch weights | ONNX (`.onnx` graph) |
| Runtime dependency | `torch` (large) | `onnxruntime` (lightweight) |
| Inference speed | Baseline | Faster (ORT graph optimizations) |
| GPU support | CUDA via PyTorch | CUDA via `onnxruntime-gpu` |
| Model source | Hub ID | Hub ID **or** local ONNX directory |

---

## Prerequisites

Install the library with the `onnxruntime` extra from the workspace root:

```bash
pip install -e ".[onnxruntime]"
```

For GPU inference, replace `onnxruntime` with `onnxruntime-gpu` after installation:

```bash
pip install onnxruntime-gpu
```

> **First run** — When `export=True` (the default), Optimum auto-converts a PyTorch model to ONNX on the first call and caches the result under the HuggingFace Hub cache directory. Subsequent runs load the cached ONNX directly.

---

## Pre-exporting models manually

Exporting once and pointing the task at a local directory is the recommended production workflow:

```bash
# Install the Optimum CLI
pip install "optimum[exporters]"

# Export any supported model
optimum-cli export onnx --model google/vit-base-patch16-224  ./onnx/vit/
optimum-cli export onnx --model facebook/detr-resnet-50       ./onnx/detr/
optimum-cli export onnx --model openai/whisper-base           ./onnx/whisper-base/
# ... etc.
```

Then pass the local path via `--model`:

```bash
python examples/onnxruntime/vision/image_classification.py --model ./onnx/vit/
```

---

## Vision tasks

> All vision examples look for a sample image at `./examples/<task-name>.jpg`.
> The image path is hard-coded for simplicity; edit the script to use your own.

### Image Classification

**Default model:** `google/vit-base-patch16-224`

```bash
python examples/onnxruntime/vision/image_classification.py

# With a pre-exported model
python examples/onnxruntime/vision/image_classification.py --model ./onnx/vit/
```

---

### Object Detection

**Default model:** `facebook/detr-resnet-50`

```bash
python examples/onnxruntime/vision/object_detection.py
```

---

### Depth Estimation

**Default model:** `Intel/dpt-large`

```bash
# Stats only
python examples/onnxruntime/vision/depth_estimation.py

# Stats + save depth visualisation PNG
python examples/onnxruntime/vision/depth_estimation.py ./output/
```

---

### Image Segmentation

**Default model:** `nvidia/segformer-b0-finetuned-ade-512-512` (semantic, 150 ADE20K classes)

> **Note:** The ONNX Runtime backend (`ORTModelForSemanticSegmentation`) supports
> encoder-based **semantic** segmentation.  For **panoptic** or **instance**
> segmentation use the `transformers` backend.

```bash
python examples/onnxruntime/vision/image_segmentation.py
```

---

### Image Anonymization

**Default model:** `hustvl/yolos-tiny`

```bash
# Blur strategy (default)
python examples/onnxruntime/vision/image_anonymization.py ./output/ blur

# Pixelate
python examples/onnxruntime/vision/image_anonymization.py ./output/ pixelate

# Black box
python examples/onnxruntime/vision/image_anonymization.py ./output/ black_box
```

---

### Visual Question Answering

**Default model:** `dandelin/vilt-b32-finetuned-vqa`

```bash
python examples/onnxruntime/vision/visual_question_answering.py

# Custom question
python examples/onnxruntime/vision/visual_question_answering.py "How many cats are there?"
```

---

### Zero-Shot Image Classification

**Default model:** `openai/clip-vit-base-patch32`

```bash
python examples/onnxruntime/vision/zero_shot_image_classification.py
```

---

### Zero-Shot Object Detection

**Default model:** `google/owlvit-base-patch32`

```bash
python examples/onnxruntime/vision/zero_shot_object_detection.py
```

---

## Audio tasks

> Audio tasks require `soundfile` and `ffmpeg`.  WAV files at **16 kHz mono**
> are recommended.

### Audio Classification

**Default model:** `superb/wav2vec2-base-superb-ks` (keyword spotting)

```bash
python examples/onnxruntime/audio/audio_classification.py

# Pre-exported model
python examples/onnxruntime/audio/audio_classification.py --model ./onnx/wav2vec2-ks/
```

---

### Speech Recognition (ASR)

**Default model:** `openai/whisper-base`

```bash
# Basic transcription
python examples/onnxruntime/audio/speech_recognition.py

# With timestamps
python examples/onnxruntime/audio/speech_recognition.py --timestamps

# Force language
python examples/onnxruntime/audio/speech_recognition.py --language fr
```

---

## Task coverage

| Task | ONNX Runtime backend | Notes |
|---|---|---|
| image-classification | ✅ | `ORTModelForImageClassification` |
| object-detection | ✅ | `ORTModelForObjectDetection` |
| depth-estimation | ✅ | `ORTModelForDepthEstimation` |
| image-segmentation | ✅ | `ORTModelForSemanticSegmentation` (semantic only) |
| image-anonymization | ✅ | `ORTModelForObjectDetection` + PIL redaction |
| visual-question-answering | ✅ | `ORTModelForVisualQuestionAnswering` |
| zero-shot-image-classification | ✅ | `ORTModelForZeroShotImageClassification` |
| zero-shot-object-detection | ✅ | `ORTModelForZeroShotObjectDetection` |
| audio-classification | ✅ | `ORTModelForAudioClassification` |
| automatic-speech-recognition | ✅ | `ORTModelForSpeechSeq2Seq` |
| mask-generation | ❌ | SAM export to ONNX not yet supported; use `transformers` backend |
