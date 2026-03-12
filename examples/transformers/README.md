<!--
SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler

SPDX-License-Identifier: Apache-2.0
-->

# `examples/transformers/`

Runnable example scripts for every task exposed by the `task_inference` library, all backed by the [HuggingFace Transformers](https://huggingface.co/docs/transformers) pipeline interface.

Each script is self-contained: if you omit the path to an asset (image or audio file) it generates a small synthetic one so you can verify the pipeline end-to-end without sourcing external files.

---

## Prerequisites

Install the library with its `transformers` extras from the workspace root:

```bash
pip install -e ".[transformers]"
```

> **GPU/MPS acceleration** - Every task class accepts a `device` constructor argument (`"cpu"`, `"cuda"`, `"mps"`, or an integer device index). The examples default to `"cpu"` so they run on any machine.

> **First run** - HuggingFace pipelines download model weights on the first invocation. Make sure you have an internet connection and enough disk space (~hundreds of MB per model).

---

## Vision tasks

### Image Classification

Classify the dominant subject(s) in an image.

**Default model:** `google/vit-base-patch16-224`

```bash
# Synthetic image (no asset required)
python examples/transformers/vision/image_classification.py

# Your own image
python examples/transformers/vision/image_classification.py path/to/image.jpg
```

---

### Object Detection

Detect objects and return labelled bounding boxes.

**Default model:** `facebook/detr-resnet-50`

```bash
python examples/transformers/vision/object_detection.py path/to/image.jpg
```

---

### Depth Estimation

Predict a relative depth map for a batch of images.  An optional output
directory can be given to save the PNG depth visualisation.

**Default model:** `Intel/dpt-large`

```bash
# Print depth statistics only
python examples/transformers/vision/depth_estimation.py path/to/image.jpg

# Also save the depth-vis PNG
python examples/transformers/vision/depth_estimation.py path/to/image.jpg ./out/
```

---

### Image Segmentation

Segment an image into labelled regions (panoptic / semantic / instance).

**Default model:** `facebook/mask2former-swin-large-coco-panoptic`

```bash
# Default subtask
python examples/transformers/vision/image_segmentation.py path/to/image.jpg

# Force instance segmentation
python examples/transformers/vision/image_segmentation.py path/to/image.jpg instance
```

---

### Image Anonymization

Detect sensitive regions (persons, faces) and blur / pixelate / black-box them.

**Default model:** `hustvl/yolos-tiny`

```bash
# Blur (default strategy)
python examples/transformers/vision/image_anonymization.py path/to/image.jpg anonymized.png blur

# Pixelate
python examples/transformers/vision/image_anonymization.py path/to/image.jpg anonymized.png pixelate

# Black box
python examples/transformers/vision/image_anonymization.py path/to/image.jpg anonymized.png black_box
```

---


### Mask Generation (SAM-style)

Generate segmentation masks — automatically or guided by a prompt point.

**Default model:** `facebook/sam-vit-base`

```bash
# Automatic mode - segment everything
python examples/transformers/vision/mask_generation.py path/to/image.jpg ./masks/

# Prompted mode - segment the object at pixel (320, 240)
python examples/transformers/vision/mask_generation.py path/to/image.jpg ./masks/ 320 240
```

---

### Visual Question Answering

Answer a natural-language question about an image.

**Default model:** `dandelin/vilt-b32-finetuned-vqa`

```bash
# Default question: "What color is the image?"
python examples/transformers/vision/visual_question_answering.py path/to/image.jpg

# Custom question
python examples/transformers/vision/visual_question_answering.py path/to/image.jpg "How many people are there?"
```

---

### Image Text-to-Text

Generate text from an image with an optional instruction prompt (multi-modal LLM).

**Default model:** `llava-hf/llava-interleave-qwen-0.5b-hf`

```bash
python examples/transformers/vision/image_text_to_text.py path/to/image.jpg
```

---

### Zero-Shot Image Classification

Classify an image against arbitrary candidate labels without task-specific training.

**Default model:** `openai/clip-vit-base-patch32`

```bash
python examples/transformers/vision/zero_shot_image_classification.py path/to/image.jpg
```

---

### Zero-Shot Object Detection

Detect objects described by arbitrary candidate labels without task-specific training.

**Default model:** `google/owlvit-base-patch32`

```bash
python examples/transformers/vision/zero_shot_object_detection.py path/to/image.jpg
```

---

## Audio tasks

> Audio tasks require `soundfile` (installed automatically with the `transformers` extra): `pip install soundfile`.  
> WAV files at **16 kHz mono** are recommended for best compatibility.

### Audio Classification

Classify an audio clip (keyword spotting, sound events, etc.).

**Default model:** `superb/wav2vec2-base-superb-ks`  
For environmental sound classification: `MIT/ast-finetuned-audioset-10-10-0.4593`

```bash
python examples/transformers/audio/audio_classification.py path/to/audio.wav
```

---

### Speech Recognition (ASR)

Transcribe speech to text, optionally with timestamps.

**Default model:** `openai/whisper-base`

```bash
# Basic transcription
python examples/transformers/audio/speech_recognition.py path/to/audio.wav

# With chunk-level timestamps
python examples/transformers/audio/speech_recognition.py path/to/audio.wav --timestamps

# Force French transcription
python examples/transformers/audio/speech_recognition.py path/to/audio.wav --language fr
```

---


## Running all examples at once (smoke test)

```bash
for script in examples/transformers/vision/*.py examples/transformers/audio/*.py; do
    echo "=== $script ==="
    python "$script"
    echo
done
```

All scripts exit cleanly when run without arguments, using synthetic assets.
