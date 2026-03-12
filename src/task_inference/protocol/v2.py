# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Open Inference Protocol v2 - Pydantic data models.

Specification:
  https://github.com/open-inference/open-inference-protocol/blob/main/specification/protocol/inference_rest.md
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Datatype(str, Enum):
    """Supported tensor element datatypes defined by OIP v2."""

    BOOL = "BOOL"
    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FP16 = "FP16"
    FP32 = "FP32"
    FP64 = "FP64"
    BYTES = "BYTES"


# ---------------------------------------------------------------------------
# NumPy ↔ Datatype mappings
# ---------------------------------------------------------------------------

_NUMPY_TO_DATATYPE: dict[np.dtype, Datatype] = {
    np.dtype("bool"): Datatype.BOOL,
    np.dtype("uint8"): Datatype.UINT8,
    np.dtype("uint16"): Datatype.UINT16,
    np.dtype("uint32"): Datatype.UINT32,
    np.dtype("uint64"): Datatype.UINT64,
    np.dtype("int8"): Datatype.INT8,
    np.dtype("int16"): Datatype.INT16,
    np.dtype("int32"): Datatype.INT32,
    np.dtype("int64"): Datatype.INT64,
    np.dtype("float16"): Datatype.FP16,
    np.dtype("float32"): Datatype.FP32,
    np.dtype("float64"): Datatype.FP64,
    np.dtype("bytes_"): Datatype.BYTES,
}

_DATATYPE_TO_NUMPY: dict[Datatype, np.dtype] = {v: k for k, v in _NUMPY_TO_DATATYPE.items()}


class RequestInput(BaseModel):
    """A single named input tensor carried inside an :class:`InferenceRequest`.

    ``data`` holds the tensor elements in row-major order.
    """

    name: str
    shape: list[int]
    datatype: Datatype
    data: list[Any] | None = None
    parameters: dict[str, Any] | None = None

    @classmethod
    def from_ndarray(
        cls,
        name: str,
        array: np.ndarray,
        datatype: Datatype | None = None,
    ) -> RequestInput:
        """Construct a :class:`RequestInput` from a NumPy ndarray.

        Parameters
        ----------
        name:
            Tensor name.
        array:
            Source array.  Its shape is used as-is; elements are flattened
            into row-major order.
        datatype:
            OIP datatype to use.  When *None* the type is inferred from
            ``array.dtype``; an unsupported dtype raises :class:`ValueError`.
        """
        if datatype is None:
            np_dtype = np.dtype(array.dtype)
            if np_dtype not in _NUMPY_TO_DATATYPE:
                raise ValueError(
                    f"Unsupported numpy dtype '{np_dtype}'. "
                    "Pass an explicit datatype= to override."
                )
            datatype = _NUMPY_TO_DATATYPE[np_dtype]

        shape = list(array.shape)

        if datatype == Datatype.BYTES:
            flat: list[Any] = []
            for item in array.ravel():
                flat.append(item)
        else:
            flat = array.ravel().tolist()

        return cls(name=name, shape=shape, datatype=datatype, data=flat)


class RequestOutput(BaseModel):
    """Describes a desired output tensor in an :class:`InferenceRequest`.

    When included in a request, only the named outputs will be returned.
    """

    name: str
    parameters: dict[str, Any] | None = None


class ResponseOutput(BaseModel):
    """A single named output tensor carried inside an :class:`InferenceResponse`.

    ``data`` serialisation rules are the same as for :class:`RequestInput`.
    """

    name: str
    shape: list[int]
    datatype: Datatype
    data: list[Any] | None = None
    parameters: dict[str, Any] | None = None

    def to_ndarray(self) -> np.ndarray:
        """Convert the output tensor to a NumPy ndarray.

        The returned array has the dtype mapped from :attr:`datatype` and is
        reshaped to :attr:`shape`.  For ``BYTES`` tensors an object array is
        returned; elements are left as-is.

        Raises
        ------
        ValueError
            When :attr:`data` is ``None``.
        """
        if self.data is None:
            raise ValueError(f"ResponseOutput '{self.name}' has no data")

        arr = np.array(self.data, dtype=_DATATYPE_TO_NUMPY[self.datatype])

        return arr.reshape(self.shape)


class InferenceRequest(BaseModel):
    """OIP v2 inference request envelope."""

    id: str | None = None
    inputs: list[RequestInput] = Field(default_factory=list)
    outputs: list[RequestOutput] | None = None
    parameters: dict[str, Any] | None = None


class InferenceResponse(BaseModel):
    """OIP v2 inference response envelope."""

    model_name: str
    id: str | None = None
    outputs: list[ResponseOutput] = Field(default_factory=list)
    parameters: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_output(self, name: str) -> ResponseOutput:
        """Return the output tensor with the given *name*.

        Raises :class:`KeyError` if no tensor with that name exists.
        """
        for tensor in self.outputs:
            if tensor.name == name:
                return tensor
        raise KeyError(f"Output tensor '{name}' not found in response")


# ---------------------------------------------------------------------------
# Model Metadata (GET v2/models/<name>)
# ---------------------------------------------------------------------------


class MetadataTensor(BaseModel):
    """Describes a single input or output tensor in a model metadata response.

    Corresponds to the ``$metadata_tensor`` object in the OIP v2 specification.
    Variable-size dimensions are expressed as ``-1``.
    """

    name: str
    datatype: Datatype
    shape: list[int]


class ParameterMetadata(BaseModel):
    """Extension: documents a single inference parameter accepted by a task.

    Parameters are passed in the ``parameters`` map of an
    :class:`InferenceRequest` rather than as tensors.  This model is added as
    a non-standard extension field (``parameters``) in
    :class:`ModelMetadataResponse` to make task requirements self-describing.
    """

    name: str
    datatype: str = Field(
        ...,
        description="JSON-compatible type: 'string', 'int64', 'fp64', or 'bool'",
    )
    description: str | None = None
    required: bool = False
    default: Any | None = None


class ModelMetadataResponse(BaseModel):
    """OIP v2 model metadata response (``GET v2/models/<name>``).

    Corresponds to the ``$metadata_model_response`` object in the OIP v2
    specification, extended with a non-standard ``parameters`` field that
    documents the inference parameters accepted by the task.
    """

    name: str
    versions: list[str] | None = None
    platform: str = ""
    inputs: list[MetadataTensor]
    outputs: list[MetadataTensor]
    parameters: list[ParameterMetadata] | None = None
