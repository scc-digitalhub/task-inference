# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the OIP v2 protocol models."""
from __future__ import annotations


import numpy as np
import pytest

from task_inference.protocol.v2 import (
    Datatype,
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    ResponseOutput,
)


def test_inference_request_roundtrip():
    req = InferenceRequest(
        id="test-1",
        inputs=[
            RequestInput(
                name="image",
                shape=[1],
                datatype=Datatype.BYTES,
                data=["aGVsbG8="],  # b64 "hello"
            )
        ],
        parameters={"top_k": 5},
    )
    json_str = req.model_dump_json()
    restored = InferenceRequest.model_validate_json(json_str)
    assert restored.id == "test-1"
    assert restored.inputs[0].name == "image"
    assert restored.inputs[0].datatype == Datatype.BYTES
    assert restored.parameters == {"top_k": 5}


def test_inference_response_get_output():
    resp = InferenceResponse(
        model_name="my-model",
        outputs=[
            ResponseOutput(name="labels", shape=[2], datatype=Datatype.BYTES, data=["cat", "dog"]),
            ResponseOutput(name="scores", shape=[2], datatype=Datatype.FP32, data=[0.9, 0.1]),
        ],
    )
    assert resp.get_output("labels").data == ["cat", "dog"]
    assert resp.get_output("scores").data == [0.9, 0.1]

    with pytest.raises(KeyError):
        resp.get_output("missing")


def test_datatype_enum_values():
    assert Datatype.FP32 == "FP32"
    assert Datatype.BYTES == "BYTES"


# ---------------------------------------------------------------------------
# RequestInput.from_ndarray
# ---------------------------------------------------------------------------


class TestRequestInputFromNdarray:
    def test_float32(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.name == "x"
        assert inp.shape == [3]
        assert inp.datatype == Datatype.FP32
        np.testing.assert_allclose(inp.data, [1.0, 2.0, 3.0], rtol=1e-6)

    def test_float64(self):
        arr = np.array([1.5, 2.5], dtype=np.float64)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.FP64
        assert inp.data == [1.5, 2.5]

    def test_float16(self):
        arr = np.array([0.5, 1.0], dtype=np.float16)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.FP16
        assert len(inp.data) == 2

    def test_int32_2d_row_major(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.shape == [2, 2]
        assert inp.datatype == Datatype.INT32
        assert inp.data == [1, 2, 3, 4]  # row-major flattening

    def test_int64(self):
        arr = np.array([10, 20, 30], dtype=np.int64)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.INT64
        assert inp.data == [10, 20, 30]

    def test_int16(self):
        arr = np.array([-100, 0, 100], dtype=np.int16)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.INT16

    def test_int8(self):
        arr = np.array([-1, 0, 1], dtype=np.int8)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.INT8

    def test_uint8(self):
        arr = np.array([0, 128, 255], dtype=np.uint8)
        inp = RequestInput.from_ndarray("pixels", arr)
        assert inp.datatype == Datatype.UINT8
        assert inp.data == [0, 128, 255]

    def test_uint16(self):
        arr = np.array([1000, 2000], dtype=np.uint16)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.UINT16

    def test_uint32(self):
        arr = np.array([1, 2], dtype=np.uint32)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.UINT32

    def test_uint64(self):
        arr = np.array([1, 2], dtype=np.uint64)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.datatype == Datatype.UINT64

    def test_bool(self):
        arr = np.array([True, False, True], dtype=np.bool_)
        inp = RequestInput.from_ndarray("mask", arr)
        assert inp.datatype == Datatype.BOOL
        assert inp.data == [True, False, True]

    def test_explicit_datatype_overrides_inferred(self):
        arr = np.array([1.0, 2.0], dtype=np.float64)
        inp = RequestInput.from_ndarray("x", arr, datatype=Datatype.FP32)
        assert inp.datatype == Datatype.FP32

    def test_3d_shape_preserved(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        inp = RequestInput.from_ndarray("x", arr)
        assert inp.shape == [2, 3, 4]
        assert len(inp.data) == 24  # 2*3*4 elements

    def test_unsupported_dtype_raises_value_error(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            RequestInput.from_ndarray("x", arr)


# ---------------------------------------------------------------------------
# ResponseOutput.to_ndarray
# ---------------------------------------------------------------------------


class TestResponseOutputToNdarray:
    def test_fp32(self):
        out = ResponseOutput(name="scores", shape=[3], datatype=Datatype.FP32, data=[1.0, 2.0, 3.0])
        arr = out.to_ndarray()
        assert arr.dtype == np.float32
        assert arr.shape == (3,)
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_fp64(self):
        out = ResponseOutput(name="x", shape=[2], datatype=Datatype.FP64, data=[0.1, 0.2])
        arr = out.to_ndarray()
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, [0.1, 0.2])

    def test_fp16(self):
        out = ResponseOutput(name="x", shape=[2], datatype=Datatype.FP16, data=[1.0, 2.0])
        arr = out.to_ndarray()
        assert arr.dtype == np.float16

    def test_int32(self):
        out = ResponseOutput(name="ids", shape=[4], datatype=Datatype.INT32, data=[1, 2, 3, 4])
        arr = out.to_ndarray()
        assert arr.dtype == np.int32
        np.testing.assert_array_equal(arr, [1, 2, 3, 4])

    def test_int64(self):
        out = ResponseOutput(name="ids", shape=[2], datatype=Datatype.INT64, data=[100, 200])
        arr = out.to_ndarray()
        assert arr.dtype == np.int64

    def test_int16(self):
        out = ResponseOutput(name="x", shape=[2], datatype=Datatype.INT16, data=[-10, 10])
        arr = out.to_ndarray()
        assert arr.dtype == np.int16

    def test_int8(self):
        out = ResponseOutput(name="x", shape=[3], datatype=Datatype.INT8, data=[-1, 0, 1])
        arr = out.to_ndarray()
        assert arr.dtype == np.int8
        np.testing.assert_array_equal(arr, [-1, 0, 1])

    def test_uint8(self):
        out = ResponseOutput(name="pixels", shape=[3], datatype=Datatype.UINT8, data=[0, 128, 255])
        arr = out.to_ndarray()
        assert arr.dtype == np.uint8
        np.testing.assert_array_equal(arr, [0, 128, 255])

    def test_uint16(self):
        out = ResponseOutput(name="x", shape=[2], datatype=Datatype.UINT16, data=[1000, 2000])
        arr = out.to_ndarray()
        assert arr.dtype == np.uint16

    def test_uint32(self):
        out = ResponseOutput(name="x", shape=[2], datatype=Datatype.UINT32, data=[1, 2])
        arr = out.to_ndarray()
        assert arr.dtype == np.uint32

    def test_uint64(self):
        out = ResponseOutput(name="x", shape=[2], datatype=Datatype.UINT64, data=[1, 2])
        arr = out.to_ndarray()
        assert arr.dtype == np.uint64

    def test_bool(self):
        out = ResponseOutput(name="mask", shape=[3], datatype=Datatype.BOOL, data=[True, False, True])
        arr = out.to_ndarray()
        assert arr.dtype == np.bool_
        np.testing.assert_array_equal(arr, [True, False, True])

    def test_2d_shape(self):
        out = ResponseOutput(
            name="matrix", shape=[2, 3], datatype=Datatype.INT32, data=[1, 2, 3, 4, 5, 6]
        )
        arr = out.to_ndarray()
        assert arr.shape == (2, 3)
        np.testing.assert_array_equal(arr, [[1, 2, 3], [4, 5, 6]])

    def test_none_data_raises_value_error(self):
        out = ResponseOutput(name="x", shape=[3], datatype=Datatype.FP32, data=None)
        with pytest.raises(ValueError, match="has no data"):
            out.to_ndarray()

    def test_roundtrip_numeric(self):
        """from_ndarray -> ResponseOutput.to_ndarray should reproduce the original array."""
        arr_in = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        inp = RequestInput.from_ndarray("x", arr_in)
        out = ResponseOutput(name="x", shape=inp.shape, datatype=inp.datatype, data=inp.data)
        arr_out = out.to_ndarray()
        np.testing.assert_array_equal(arr_in, arr_out)
