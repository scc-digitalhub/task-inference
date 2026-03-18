# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import base64

from open_inference.openapi.client import OpenInferenceClient, InferenceRequest

client = OpenInferenceClient(base_url='http://localhost:8080')

# Check that the server is live, and it has the iris model loaded
client.check_server_readiness()

# ------------------------------------------------------------------ #
# Image classification
# ------------------------------------------------------------------ #
with open("./examples/image-classification.jpg", "rb") as f:
    image_bytes = f.read()

image_str = base64.b64encode(image_bytes)

pred = client.model_infer(
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

print(repr(pred))