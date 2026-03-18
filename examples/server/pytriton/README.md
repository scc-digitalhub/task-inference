<!--
SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler

SPDX-License-Identifier: Apache-2.0
-->

# PyTriton Server

Exposes any task registered in `task-inference` as a [Triton Inference Server](https://github.com/triton-inference-server/server) model using [PyTriton](https://github.com/triton-inference-server/pytriton).

The server translates between PyTriton's numpy-based wire format and the OIP v2 protocol used internally by the task implementations.

## Files

| File | Description |
|---|---|
| `server.py` | Server implementation - task factory bridge + PyTriton binding |
| `Dockerfile` | Ubuntu 22.04 container image with all Transformers extras installed |
| `docker-compose.yml` | Single-command deployment with model weight caching |

---

## Quick Start

### Option A - Docker Compose (recommended)

```bash
# From the repository root:
docker compose -f examples/server/docker-compose.yml up --build
```

The server starts on:
- **HTTP** `localhost:8000`
- **gRPC** `localhost:8001`
- **Metrics** `localhost:8002`

### Option B - Run directly (Linux only)

> **Note:** PyTriton bundles Triton Inference Server binaries that are compiled for
> Linux x86-64.  The server cannot be started natively on macOS or Windows.

```bash
pip install nvidia-pytriton
pip install -e ".[transformers,audio]"

python examples/pytriton/server/server.py --task image-classification
```

---

## Configuration

All options can be set via **CLI flags** or **environment variables**.

| Environment variable | CLI flag | Default | Description |
|---|---|---|---|
| `TASK` | `--task` | `image-classification` | Task name (see [Supported tasks](#supported-tasks)) |
| `BACKEND` | `--backend` | `transformers` | Inference backend |
| `MODEL` | `--model` | *(task default)* | HuggingFace model ID or local path |
| `DEVICE` | `--device` | `cpu` | PyTorch device (`cpu`, `cuda`, `cuda:0`, `mps`) |
| `HTTP_PORT` | `--http-port` | `8000` | Triton HTTP port |
| `GRPC_PORT` | `--grpc-port` | `8001` | Triton gRPC port |
| `METRICS_PORT` | `--metrics-port` | `8002` | Prometheus metrics port |
| *(n/a)* | `--model-params` | - | JSON object of extra task constructor kwargs |

### Examples

```bash
# Mask generation with a specific model
TASK=mask-generation MODEL=facebook/sam-vit-base \
  docker compose -f examples/server/docker-compose.yml up --build

# ASR with a custom chunk length
python examples/server/pytriton/server.py \
    --task automatic-speech-recognition \
    --model openai/whisper-small \
    --model-params '{"chunk_length_s": 30}'

# GPU inference
python examples/server/pytriton/server.py \
    --task image-classification \
    --device cuda:0
```

---

## Supported Tasks

All tasks from the factory are supported.  The `--task` value is the
HuggingFace pipeline / OIP v2 task identifier.

| Task | `--task` value | Default model |
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

The server uses the standard [Triton HTTP and gRPC](https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_http.md) APIs.

### Input tensors

Tensor names and shapes match the task's `METADATA_INPUTS`.  Query the
server's metadata endpoint to inspect them at runtime:

```bash
curl http://localhost:8000/v2/models/image_classification
```

Typical input for vision tasks:

| Name | Dtype | Shape | Description |
|---|---|---|---|
| `image` | `BYTES` | `[-1]` | Raw image bytes (JPEG / PNG / …), one per batch element |

Optional parameter inputs (from `METADATA_PARAMETERS`) can be supplied as
scalar tensors to override the task defaults on a per-request basis:

| Name | Dtype | Shape | Description |
|---|---|---|---|
| `top_k` | `INT64` | `[1]` | *(image-classification)* Number of top predictions |
| `pred_iou_thresh` | `FP64` | `[1]` | *(mask-generation)* IoU quality threshold |
| … | … | … | See task metadata for the full list |

### Output tensors

All outputs are returned as **BYTES tensors** carrying **UTF-8 JSON blobs**,
one per batch element.  The JSON structure mirrors the OIP v2 response data:

| Task | Output name | JSON type | Example |
|---|---|---|---|
| `image-classification` | `label` | `list[str]` | `["tabby cat","tiger cat"]` |
| `image-classification` | `score` | `list[float]` | `[0.9412, 0.0321]` |
| `object-detection` | `label` | `list[str]` | `["cat","dog"]` |
| `object-detection` | `score` | `list[float]` | `[0.98, 0.94]` |
| `object-detection` | `box` | `list[list[float]]` | `[[x1,y1,x2,y2], …]` |
| `mask-generation` | `mask` | `list[str]` | *(PNG bytes per mask)* |
| `mask-generation` | `score` | `list[float]` | `[0.92, 0.89, …]` |

---

## Client Examples

### Python - `tritonclient.http`

```python
import json
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

# ------------------------------------------------------------------ #
# Image classification
# ------------------------------------------------------------------ #
with open("cat.jpg", "rb") as f:
    image_bytes = f.read()

# Single-image batch: shape [1] of BYTES
image_input = httpclient.InferInput("image", [1], "BYTES")
image_input.set_data_from_numpy(np.array([image_bytes], dtype=object))

result = client.infer("image_classification", inputs=[image_input])

labels = result.as_numpy("label")[0]   # e.g. ["tabby cat", "tiger cat"]
scores = result.as_numpy("score")[0]   # e.g. [0.9412, 0.0321]

for label, score in zip(labels, scores):
    print(f"  {label.decode("utf-8"):<30s}  {score:.4f}")
```

```python
# ------------------------------------------------------------------ #
# Mask generation - passing inference parameters
# ------------------------------------------------------------------ #
import json
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

with open("photo.jpg", "rb") as f:
    image_bytes = f.read()

image_input = httpclient.InferInput("image", [1], "BYTES")
image_input.set_data_from_numpy(np.array([image_bytes], dtype=np.bytes_))

# Override pred_iou_thresh for this request
thresh_input = httpclient.InferInput("pred_iou_thresh", [1], "FP64")
thresh_input.set_data_from_numpy(np.array([0.90], dtype=np.float64))

result = client.infer("mask_generation", inputs=[image_input, thresh_input])

masks  = result.as_numpy("mask")[0]   # list of PNG strings
scores = result.as_numpy("score")[0]  # list of float scores

print(f"Generated {len(masks)} mask(s)")
for i, (mask, score) in enumerate(zip(masks, scores), 1):
    import pathlib
    pathlib.Path(f"mask_{i:03d}.png").write_bytes(mask)
```

### Python - `tritonclient.grpc`

```python
import json
import numpy as np
import tritonclient.grpc as grpcclient

client = grpcclient.InferenceServerClient("localhost:8001")

with open("cat.jpg", "rb") as f:
    image_bytes = f.read()

image_input = grpcclient.InferInput("image", [1], "BYTES")
image_input.set_data_from_numpy(np.array([image_bytes], dtype=object))

result = client.infer("image_classification", inputs=[image_input])

labels = result.as_numpy("label")[0]
scores = result.as_numpy("score")[0]
```

## GPU Support

Uncomment the `deploy.resources` block in `docker-compose.yml` and set
`DEVICE=cuda`:

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
(`huggingface_cache → /root/.cache/huggingface`) so that model weights are
only downloaded once and survive container restarts.

To pre-populate the cache before starting the server:

```bash
python -c "
from transformers import pipeline
pipeline('image-classification', model='google/vit-base-patch16-224')
"
```

or simply let the server download on first startup.
