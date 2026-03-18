# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

# ------------------------------------------------------------------ #
# Image classification
# ------------------------------------------------------------------ #
with open("./examples/image-classification.jpg", "rb") as f:
    image_bytes = f.read()

# Single-image batch: shape [1] of BYTES
image_input = httpclient.InferInput("image", [1], "BYTES")
image_input.set_data_from_numpy(np.array([image_bytes], dtype=np.bytes_))

result = client.infer("image_classification", inputs=[image_input])

labels = result.as_numpy("label")[0]   
scores = result.as_numpy("score")[0]   

for label, score in zip(labels, scores):
    print(f"  {label.decode("utf-8"):<30s}  {score:.4f}")
