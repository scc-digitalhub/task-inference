# Task Inference 

[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/scc-digitalhub/task-inference/LICENSE) ![GitHub Release](https://img.shields.io/github/v/release/scc-digitalhub/task-inference)
![Status](https://img.shields.io/badge/status-stable-gold)

A Python library that provides **task-oriented abstractions for AI inference**, bridging [HuggingFace Transformers](https://huggingface.co/docs/transformers/) pipeline tasks with the [Open Inference Protocol v2 (OIP v2)](https://github.com/open-inference/open-inference-protocol/blob/main/specification/protocol/inference_rest.md) tensor format.

## Key Design

Each task is modelled in three layers:

| Layer | Role |
|---|---|
| **Protocol** (`protocol/v2.py`) | Pydantic models for OIP v2 `InferenceRequest` / `InferenceResponse` |
| **Task** (`tasks/`) | Domain-specific input/output Pydantic schemas; `process` left abstract |
| **Implementation** (`implementations/transformers/`) | HuggingFace Transformers reference backend that fulfils `process` |

This separation allows swapping backends (ONNX, TensorRT, REST endpoint, …) without changing the task schema or OIP conversion logic.

### Input / Output schemas

Every task exposes a pair of Pydantic models (e.g. `ImageClassificationInput` / `ImageClassificationOutput`). These models own the conversion to and from OIP v2 tensors:

| Method | Direction |
|---|---|
| `XxxInput.to_inference_request()` | Python input → `InferenceRequest` |
| `XxxInput.from_inference_request(request)` | `InferenceRequest` → Python input |
| `XxxOutput.to_inference_response(model_name)` | Python output → `InferenceResponse` |
| `XxxOutput.from_inference_response(response)` | `InferenceResponse` → Python output |

The `process(inputs: XxxInput) -> XxxOutput` method on task classes works entirely with these domain objects — no raw OIP tensors required.

## Supported Tasks

### Vision
| Task | Class | Default Model |
|---|---|---|
| Image Classification | `TransformersImageClassificationTask` | `google/vit-base-patch16-224` |
| Image Segmentation | `TransformersImageSegmentationTask` | `facebook/mask2former-swin-large-coco-panoptic` |
| Object Detection | `TransformersObjectDetectionTask` | `facebook/detr-resnet-50` |
| Depth Estimation | `TransformersDepthEstimationTask` | `Intel/dpt-large` |
| Mask Generation | `TransformersMaskGenerationTask` | `facebook/sam-vit-base` |
| Visual Question Answering | `TransformersVQATask` | `dandelin/vilt-b32-finetuned-vqa` |
| Image Anonymization | `TransformersImageAnonymizationTask` | `hustvl/yolos-tiny` |

### Audio
| Task | Class | Default Model |
|---|---|---|
| Automatic Speech Recognition | `TransformersASRTask` | `openai/whisper-base` |
| Audio Classification | `TransformersAudioClassificationTask` | `superb/wav2vec2-base-superb-ks` |

## Installation

```bash
# Core (schemas + OIP v2 protocol only)
pip install task-inference

# With HuggingFace Transformers backend
pip install "task-inference[transformers]"

# With audio support
pip install "task-inference[all]"
```

## Quick Start

### Via the factory (recommended)

```python
from task_inference import create_task

with open("cat.jpg", "rb") as f:
    image_bytes = f.read()

task = create_task(
    backend="transformers",
    task_name="image-classification",
    model_name="google/vit-base-patch16-224",
    model_params={"device": "cpu"},
)

from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)

inp = ImageClassificationInput(image=image_bytes, top_k=3)
resp = task(inp.to_inference_request())
result = ImageClassificationOutput.from_inference_response(resp)
for r in result.results:
    print(r.label, r.score)
```

`model_params` is forwarded directly to the backend constructor, so any
backend-specific keyword argument (e.g. `device`, `chunk_length_s`,
`points_per_batch`) can be passed here.

To discover what backends and task names are available:

```python
from task_inference import supported_tasks

print(supported_tasks())
# {'transformers': ['audio-classification', 'automatic-speech-recognition', ...]}
```

### Direct instantiation

```python
from task_inference.implementations.transformers.vision import (
    TransformersImageClassificationTask,
)
from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)

task = TransformersImageClassificationTask(model_name="google/vit-base-patch16-224")

# Call via OIP v2 round-trip
inp = ImageClassificationInput(image=image_bytes, top_k=3)
resp = task(inp.to_inference_request())
result = ImageClassificationOutput.from_inference_response(resp)
for r in result.results:
    print(r.label, r.score)

# Convenience wrapper - build input from keyword arguments, returns InferenceResponse
resp = task.run(image=image_bytes, top_k=3)
result = ImageClassificationOutput.from_inference_response(resp)
```

### OIP v2 round-trip

The input/output models handle all serialisation, so you can integrate with any
OIP v2-compatible server without touching the task implementation:

```python
from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
)

# --- Client side ---
inputs  = ImageClassificationInput(image=image_bytes, top_k=3)
request = inputs.to_inference_request()   # → InferenceRequest (send over HTTP)

# --- Server side ---
response = task(request)                  # returns InferenceResponse directly

# --- Client side (parse response) ---
output = ImageClassificationOutput.from_inference_response(response)
for r in output.results:
    print(r.label, r.score)
```

## Factory reference

### `create_task(backend, task_name, model_name=None, model_params=None)`

| Parameter | Type | Description |
|---|---|---|
| `backend` | `str` | Backend name — currently `"transformers"` |
| `task_name` | `str` | Task identifier (see table below) |
| `model_name` | `str \| None` | HuggingFace model id or local path. When `None` the backend's built-in default model is used. |
| `model_params` | `dict \| None` | Extra keyword arguments passed to the constructor (e.g. `device`, `chunk_length_s`) |

Raises `ValueError` for unknown backends or task names.

### `supported_tasks(backend=None)`

Returns a `dict[str, list[str]]` mapping each backend to its supported task names. Pass a backend name to filter to a single backend.

### Task name reference

| Task name | Input class | Output class |
|---|---|---|
| `image-classification` | `ImageClassificationInput` | `ImageClassificationOutput` |
| `object-detection` | `ObjectDetectionInput` | `ObjectDetectionOutput` |
| `depth-estimation` | `DepthEstimationInput` | `DepthEstimationOutput` |
| `image-segmentation` | `ImageSegmentationInput` | `ImageSegmentationOutput` |
| `image-anonymization` | `ImageAnonymizationInput` | `ImageAnonymizationOutput` |
| `mask-generation` | `MaskGenerationInput` | `MaskGenerationOutput` |
| `visual-question-answering` | `VQAInput` | `VQAOutput` |
| `image-text-to-text` | `ImageTextToTextInput` | `ImageTextToTextOutput` |
| `zero-shot-image-classification` | `ZeroShotImageClassificationInput` | `ZeroShotImageClassificationOutput` |
| `zero-shot-object-detection` | `ZeroShotObjectDetectionInput` | `ZeroShotObjectDetectionOutput` |
| `audio-classification` | `AudioClassificationInput` | `AudioClassificationOutput` |
| `automatic-speech-recognition` | `ASRInput` | `ASROutput` |

## Project Structure

```
src/task_inference/
├── factory.py          # create_task() / supported_tasks() entry points
├── protocol/           # OIP v2 Pydantic models
│   └── v2.py
├── tasks/              # Abstract task definitions + input/output schemas
│   ├── base.py
│   ├── vision/         # Image-based tasks
│   └── audio/          # Audio-based tasks
├── implementations/
│   └── transformers/   # HuggingFace reference backend
│       ├── base.py     # Shared image/audio helpers
│       ├── vision/
│       └── audio/
└── utils.py            # Image/audio encode-decode helpers
```

## Extending

Implement a new backend by subclassing the relevant task and overriding `process`:

```python
from task_inference.tasks.vision.image_classification import (
    ImageClassificationInput,
    ImageClassificationOutput,
    ImageClassificationTask,
)

class MyOnnxImageClassificationTask(ImageClassificationTask):
    def process(self, inputs: ImageClassificationInput) -> ImageClassificationOutput:
        # your ONNX / TensorRT / remote-endpoint logic here
        ...
```


## Security Policy

The current release is the supported version. Security fixes are released together with all other fixes in each new release.

If you discover a security vulnerability in this project, please do not open a public issue.

Instead, report it privately by emailing us at digitalhub@fbk.eu. Include as much detail as possible to help us understand and address the issue quickly and responsibly.

## Contributing

To report a bug or request a feature, please first check the existing issues to avoid duplicates. If none exist, open a new issue with a clear title and a detailed description, including any steps to reproduce if it's a bug.

To contribute code, start by forking the repository. Clone your fork locally and create a new branch for your changes. Make sure your commits follow the [Conventional Commits v1.0](https://www.conventionalcommits.org/en/v1.0.0/) specification to keep history readable and consistent.

Once your changes are ready, push your branch to your fork and open a pull request against the main branch. Be sure to include a summary of what you changed and why. If your pull request addresses an issue, mention it in the description (e.g., “Closes #123”).

Please note that new contributors may be asked to sign a Contributor License Agreement (CLA) before their pull requests can be merged. This helps us ensure compliance with open source licensing standards.

We appreciate contributions and help in improving the project!

## Authors

This project is developed and maintained by **DSLab – Fondazione Bruno Kessler**, with contributions from the open source community. A complete list of contributors is available in the project’s commit history and pull requests.

For questions or inquiries, please contact: [digitalhub@fbk.eu](mailto:digitalhub@fbk.eu)

## Copyright and license

Copyright © 2025 DSLab – Fondazione Bruno Kessler and individual contributors.

This project is licensed under the Apache License, Version 2.0.
You may not use this file except in compliance with the License. Ownership of contributions remains with the original authors and is governed by the terms of the Apache 2.0 License, including the requirement to grant a license to the project.