<!--
SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler

SPDX-License-Identifier: Apache-2.0
-->

# Serverless Server

Exposes any task registered in `task-inference` as a [Nuclio](https://nuclio.io/) serverless function using the [OpenInference](https://github.com/open-inference/open-inference-protocol) protocol.

The server translates between Nuclio's event-based invocation model and the OIP v2 protocol used internally by the task implementations.

## Files

| File | Description |
|---|---|
| `handler.py` | Nuclio function handler — task factory bridge + request/response conversion |
| `oi-processor.yaml` | Nuclio function spec — trigger configuration, tensor I/O schema, ports |
| `Dockerfile` | Container image based on `digitalhub-serverless` Python runtime |
| `docker-compose.yml` | Single-command deployment with model weight caching |
| `_nuclio_wrapper.py` | Nuclio Python runtime wrapper (vendor copy — do not edit) |
| `test.py` | Smoke test using `OpenInferenceClient` |

---

## Quick Start

### Option A - Docker Compose (recommended)

```bash
# From the repository root:
docker compose -f examples/server/serverless/docker-compose.yml up --build
```

The server starts on:
- **HTTP** `localhost:8080`
- **gRPC** `localhost:9000`

### Option B - Run directly

```bash
pip install task-inference torch>=2.0 transformers==4.42.1 pillow>=10.0.0

# Start the Nuclio processor with the provided function spec
/usr/local/bin/processor --config=examples/server/serverless/oi-processor.yaml
```

---

## Configuration

All options can be set via **environment variables**.

| Environment variable | Default | Description |
|---|---|---|
| `TASK` | `image-classification` | Task name (see [Supported tasks](#supported-tasks)) |
| `BACKEND` | `transformers` | Inference backend |
| `MODEL` | *(task default)* | HuggingFace model ID or local path |
| `DEVICE` | `cpu` | PyTorch device (`cpu`, `cuda`, `cuda:0`, `mps`) |

### Examples

```bash
# Object detection with a specific model
TASK=object-detection MODEL=facebook/detr-resnet-50 \
  docker compose -f examples/server/serverless/docker-compose.yml up --build

# Automatic speech recognition
TASK=automatic-speech-recognition MODEL=openai/whisper-small \
  docker compose -f examples/server/serverless/docker-compose.yml up --build

# GPU inference
DEVICE=cuda \
  docker compose -f examples/server/serverless/docker-compose.yml up --build
```

---

## Supported Tasks

All tasks from the factory are supported.  The `TASK` value is the
HuggingFace pipeline / OIP v2 task identifier.

| Task | `TASK` value | Default model |
|---|---|---|
| Image classification | `image-classification` | `google/vit-base-patch16-224` |
| Object detection | `object-detection` | `facebook/detr-resnet-50` |
| Depth estimation | `depth-estimation` | `Intel/dpt-large` |
| Image segmentation | `image-segmentation` | `facebook/detr-resnet-50-panoptic` |
| Image anonymisation | `image-anonymization` | *(composite - detection + inpainting)* |
| Mask generation | `mask-generation` | `facebook/sam-vit-base` |
| Visual question answering | `visual-question-answering` | `Salesforce/blip-vqa-base` |
| Zero-shot image classification | `zero-shot-image-classification` | `openai/clip-vit-base-patch32` |
| Zero-shot object detection | `zero-shot-object-detection` | `google/owlvit-base-patch32` |
| Audio classification | `audio-classification` | `facebook/wav2vec2-base` |
| Automatic speech recognition | `automatic-speech-recognition` | `openai/whisper-small` |

---

## API / Wire Format

The server implements the [OpenInference REST and gRPC](https://github.com/open-inference/open-inference-protocol) inference protocol.

### Input tensors

Tensor names and shapes match the task's `METADATA_INPUTS`, as declared in
`oi-processor.yaml`.  Query the server's metadata endpoint to inspect them
at runtime:

```bash
curl http://localhost:8080/v2/models/google%2Fvit-base-patch16-224
```

Typical input for vision tasks:

| Name | Dtype | Shape | Description |
|---|---|---|---|
| `image` | `BYTES` | `[-1]` | Base64-encoded image bytes (JPEG / PNG / …), one per batch element |

### Output tensors

| Task | Output name | Dtype | Description |
|---|---|---|---|
| `image-classification` | `labels` | `BYTES` | JSON array of predicted class labels |
| `image-classification` | `scores` | `FP32` | Confidence scores matching each label |
| `object-detection` | `labels` | `BYTES` | JSON array of detected class labels |
| `object-detection` | `scores` | `FP32` | Detection confidence scores |

---

## Client Examples

### Python - `OpenInferenceClient`

```python
import base64
from open_inference.openapi.client import OpenInferenceClient, InferenceRequest

client = OpenInferenceClient(base_url="http://localhost:8080")

# Check liveness
client.check_server_readiness()

# ------------------------------------------------------------------ #
# Image classification
# ------------------------------------------------------------------ #
with open("cat.jpg", "rb") as f:
    image_bytes = f.read()

image_str = base64.b64encode(image_bytes)

result = client.model_infer(
    "google/vit-base-patch16-224",
    request=InferenceRequest(
        inputs=[
            {
                "name": "image",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_str],
            }
        ]
    ),
)

print(result)
```

### Python - raw HTTP

```python
import base64
import json
import requests

with open("cat.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

payload = {
    "inputs": [
        {
            "name": "image",
            "shape": [1],
            "datatype": "BYTES",
            "data": [image_b64],
        }
    ]
}

resp = requests.post(
    "http://localhost:8080/v2/models/google%2Fvit-base-patch16-224/infer",
    json=payload,
)
resp.raise_for_status()
print(resp.json())
```

---

## GPU Support

Set `DEVICE=cuda` and uncomment the `deploy.resources` block in
`docker-compose.yml`:

```yaml
environment:
  DEVICE: "cuda"
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

The host must have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

---

## Model Weight Caching

The `docker-compose.yml` mounts a named Docker volume
(`huggingface_cache → /workspace/.cache/huggingface`) so that model weights
are only downloaded once and survive container restarts.

To pre-populate the cache before starting the server:

```bash
python -c "
from transformers import pipeline
pipeline('image-classification', model='google/vit-base-patch16-224')
"
```

or simply let the server download on first startup.
