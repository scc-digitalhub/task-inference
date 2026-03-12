# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task_inference.protocol - Open Inference Protocol v2 models."""

from .v2 import (
    Datatype,
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    RequestOutput,
    ResponseOutput,
)

__all__ = [
    "Datatype",
    "InferenceRequest",
    "InferenceResponse",
    "RequestInput",
    "RequestOutput",
    "ResponseOutput",
]
