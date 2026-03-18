# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""PyTriton server for task-inference.

Exposes any task registered in the factory as a Triton Inference Server model
using PyTriton.  The OIP v2 protocol is used internally; PyTriton's numpy-based
interface is used on the external boundary.

Output encoding
---------------
All output tensors are returned as ``np.bytes_`` (BYTES dtype in Triton) where
each element is a UTF-8 JSON blob representing the per-sample result value.
Clients JSON-decode the received bytes to get the original typed data
(list of labels, list of scores, list of PNG masks, etc.).

Quick start
-----------
    python examples/server/pytriton/server.py --task image-classification

    # Custom model:
    python examples/server/pytriton/server.py \\
        --task mask-generation \\
        --model facebook/sam-vit-base \\
        --device cpu

Environment variables
---------------------
All CLI flags also accept the equivalent upper-case environment variable so
that the server can be configured entirely via Docker / Compose env vars
(``TASK``, ``BACKEND``, ``MODEL``, ``DEVICE``, ``HTTP_PORT``, ``GRPC_PORT``,
``METRICS_PORT``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List
from pytriton.proxy.types import Request
import numpy as np

from task_inference import create_task
from task_inference.protocol.v2 import Datatype, InferenceRequest, RequestInput, ResponseOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_OIP_TO_NUMPY: dict[Datatype, type] = {
    Datatype.BOOL: np.bool_,
    Datatype.UINT8: np.uint8,
    Datatype.UINT16: np.uint16,
    Datatype.UINT32: np.uint32,
    Datatype.UINT64: np.uint64,
    Datatype.INT8: np.int8,
    Datatype.INT16: np.int16,
    Datatype.INT32: np.int32,
    Datatype.INT64: np.int64,
    Datatype.FP16: np.float16,
    Datatype.FP32: np.float32,
    Datatype.FP64: np.float64,
    Datatype.BYTES: np.bytes_,
}

# ---------------------------------------------------------------------------
# Tensor spec builder
# ---------------------------------------------------------------------------

def _build_tensor_specs(task: Any) -> tuple[list[Any], list[Any]]:
    """Derive PyTriton :class:`Tensor` input/output specs from task metadata.

    Returns
    -------
    (input_tensors, output_tensors)
    """
    from pytriton.model_config import Tensor  # deferred - not installed in test env

    input_tensors = [
        Tensor(
            name=m.name,
            dtype=_OIP_TO_NUMPY[m.datatype],
            shape=m.shape,
        )
        for m in task.METADATA_INPUTS
    ]

    # All outputs are exposed as BYTES tensors carrying JSON-encoded data.
    # This handles variable-length, ragged, and nested outputs uniformly.
    output_tensors = [
        Tensor(name=m.name, dtype=_OIP_TO_NUMPY[m.datatype], shape=m.shape)
        for m in task.METADATA_OUTPUTS
    ]

    return input_tensors, output_tensors


# ---------------------------------------------------------------------------
# Inference function factory
# ---------------------------------------------------------------------------

def _build_infer_fn(task: Any):
    """Return a PyTriton inference function for *task*."""

    meta_in = task.METADATA_INPUTS
    meta_out = task.METADATA_OUTPUTS
    meta_params = task.METADATA_PARAMETERS or []

    def _infer_single(req: Request) -> Dict[str, np.ndarray]:
        """Run inference for a single request and return the output dict."""
        
        inputs = req.data  # dict[str, np.ndarray] of shape (N, ...)         
        # ------------------------------------------------------------------ #
        # 2. Build OIP v2 request inputs from numpy arrays
        # ------------------------------------------------------------------ #
        request_inputs: list[RequestInput] = []
        for meta in meta_in:
            arr = inputs[meta.name]  # shape (N, ...) after @batch decoration
            request_inputs.append(
                RequestInput.from_ndarray(
                    name=meta.name,
                    array=arr,
                    datatype=meta.datatype,
                )
            )

        # ------------------------------------------------------------------ #
        # 3. Extract optional parameter overrides (use first sample's value)
        # ------------------------------------------------------------------ #
        params: dict[str, Any] = req.parameters or None

        req = InferenceRequest(
            inputs=request_inputs,
            parameters=params if params else None,
        )

        # ------------------------------------------------------------------ #
        # 4. Run inference
        # ------------------------------------------------------------------ #
        logger.debug("Calling task %s", task.TASK_NAME)
        resp = task(req)

        # ------------------------------------------------------------------ #
        # 5. Convert outputs to numpy object arrays of JSON bytes
        # ------------------------------------------------------------------ #
        result: dict[str, np.ndarray] = {}
        for meta in meta_out:
            out = resp.get_output(meta.name)
            result[meta.name] = out.to_ndarray() if out.data is not None else np.empty(0, dtype=object)

        return result


    def infer_fn(requests: List[Request]) -> List[Dict[str, np.ndarray]]:
        result = []
        for req in requests:
            result.append(_infer_single(req))
        return result

    # Give the inner function a meaningful name for Triton logs
    infer_fn.__name__ = f"infer_{task.TASK_NAME.replace('-', '_')}"
    return infer_fn


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def serve(
    task_name: str,
    backend: str = "transformers",
    model_name: str | None = None,
    device: str = "cpu",
    http_port: int = 8000,
    grpc_port: int = 8001,
    metrics_port: int = 8002,
    model_params: dict[str, Any] | None = None,
) -> None:
    """Instantiate *task* and start the PyTriton server (blocks until stopped).

    Parameters
    ----------
    task_name:
        OIP v2 / HuggingFace pipeline task identifier, e.g.
        ``"image-classification"``.
    backend:
        Inference backend name as registered in the factory.  Currently only
        ``"transformers"`` is supported.
    model_name:
        HuggingFace model identifier or local path.  Uses the task default
        when ``None``.
    device:
        PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"``).
    http_port:
        Triton HTTP port.
    grpc_port:
        Triton gRPC port.
    metrics_port:
        Triton Prometheus metrics port.
    model_params:
        Additional keyword arguments forwarded to the task constructor.
    """
    from pytriton.model_config import ModelConfig
    from pytriton.triton import Triton, TritonConfig

    logger.info(
        "Loading task  backend=%s  task=%s  model=%s  device=%s",
        backend,
        task_name,
        model_name or "<default>",
        device,
    )

    constructor_kwargs: dict[str, Any] = {"device": device}
    if model_params:
        constructor_kwargs.update(model_params)

    task = create_task(
        backend=backend,
        task_name=task_name,
        model_name=model_name,
        model_params=constructor_kwargs,
    )
    logger.info("Task loaded: %s", task.__class__.__name__)

    infer_fn = _build_infer_fn(task)
    input_tensors, output_tensors = _build_tensor_specs(task)

    # Triton model name: use normalised task name so it is stable regardless of
    # which model weights are loaded.
    triton_model_name = task_name.replace("-", "_")

    triton_cfg = TritonConfig(
        http_port=http_port,
        grpc_port=grpc_port,
        metrics_port=metrics_port,
    )

    logger.info(
        "Starting Triton server  model=%s  http=%d  grpc=%d  metrics=%d",
        triton_model_name,
        http_port,
        grpc_port,
        metrics_port,
    )

    with Triton(config=triton_cfg) as triton:
        triton.bind(
            model_name=triton_model_name,
            infer_func=infer_fn,
            inputs=input_tensors,
            outputs=output_tensors,
            config=ModelConfig(batching=False),
        )
        triton.serve()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTriton server for task-inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "All flags also accept an equivalent upper-case environment variable "
            "(TASK, BACKEND, MODEL, DEVICE, HTTP_PORT, GRPC_PORT, METRICS_PORT)."
        ),
    )
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK", "image-classification"),
        help="OIP v2 task name (e.g. image-classification, mask-generation)",
    )
    parser.add_argument(
        "--backend",
        default=os.environ.get("BACKEND", "transformers"),
        help="Inference backend (currently only 'transformers' is supported)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL") or None,
        help="HuggingFace model identifier or local path (uses task default when omitted)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("DEVICE", "cpu"),
        help="Inference device: cpu, cuda, cuda:0, mps, …",
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=int(os.environ.get("HTTP_PORT", "8000")),
        help="Triton HTTP port",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=int(os.environ.get("GRPC_PORT", "8001")),
        help="Triton gRPC port",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=int(os.environ.get("METRICS_PORT", "8002")),
        help="Triton Prometheus metrics port",
    )
    parser.add_argument(
        "--model-params",
        default=None,
        metavar="JSON",
        help=(
            "JSON object of extra keyword arguments forwarded to the task constructor. "
            "Example: '{\"chunk_length_s\": 30}'"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    extra: dict[str, Any] | None = None
    if args.model_params:
        try:
            extra = json.loads(args.model_params)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --model-params JSON: %s", exc)
            sys.exit(1)

    serve(
        task_name=args.task,
        backend=args.backend,
        model_name=args.model,
        device=args.device,
        http_port=args.http_port,
        grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
        model_params=extra,
    )
