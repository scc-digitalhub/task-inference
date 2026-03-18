"""
Video Anonymization Service using face detection and blurring.

This microservice provides HTTP endpoints for anonymizing images by detecting
and blurring faces (and optionally license plates). It integrates with the
digitalhub-servicegraph framework for real-time video stream processing.
"""

import base64
import os
import time
from urllib import request

import numpy as np
from io import BytesIO
from PIL import Image
import json

from typing import Any, Dict, List

from task_inference import create_task
from task_inference.protocol.v2 import Datatype, InferenceRequest, RequestInput, ResponseOutput

def init_model(context):

    task = os.getenv("TASK", "image-classification")
    backend = os.getenv("BACKEND", "transformers")
    device = os.getenv("DEVICE", "cpu")
    model = os.getenv("MODEL", None)

    constructor_kwargs: dict[str, Any] = {"device": device}

    task = create_task(
        backend=backend,
        task_name=task,
        model_name=model,
        model_params=constructor_kwargs,
    )
    context.logger.info("Task loaded: %s", task.__class__.__name__)
    setattr(context, "task", task)

def init_context(context):
    init_model(context)

def handler(context, event):

    request: ServerlessInferenceRequest = ServerlessInferenceRequest(event.body)
    task  = getattr(context, 'task', None)
    if task is None:
        init_model(context)
        task = getattr(context, 'task', None)
        if task is None:
            return {"error": "Task not initialized"}
    meta_in = task.METADATA_INPUTS
    meta_out = task.METADATA_OUTPUTS

    try:
        request_inputs: list[RequestInput] = []
        for inp in request.inputs:
            request_inputs.append(
                RequestInput(
                    name=inp.name,
                    datatype=inp.datatype,
                    shape=inp.shape,
                    data=inp.data,
                    parameters=inp.parameters,
                )
            )
        params: dict[str, Any] = request.parameters or None
        req = InferenceRequest(
            inputs=request_inputs,
            parameters=params if params else None,
        )

        context.logger.debug("Calling task %s", task.TASK_NAME)
        resp = task(req)

        return resp.dict()
        
    except Exception as e:
        context.logger.error(f"Error processing image: {e}")
        return {
            "error": str(e),
            "status": "error"
        }


class ServerlessInferenceRequest:
    id: str | None = None
    parameters: dict | None = None
    inputs: list = []
    outputs: list = []

    def __init__(self, request: dict) -> None:
        self.id = request.get("id")
        self.parameters = request.get("parameters")
        self.inputs = [ServerlessRequestInput(**input) for input in request.get("inputs", [])]
        self.outputs = [ServerlessRequestOutput(**output) for output in request.get("outputs", [])]

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"ServerlessInferenceRequest(id={self.id}, parameters={self.parameters}, inputs={self.inputs}, outputs={self.outputs})"

    def dict(self) -> dict:
        return {
            "id": self.id,
            "parameters": self.parameters,
            "inputs": [i.dict() for i in self.inputs],
            "outputs": [o.dict() for o in self.outputs],
        }

    def json(self) -> str:
        return json.dumps(self.dict())


class ServerlessRequestInput:
    name: str
    datatype: str
    shape: list[int]
    data: list[any]
    parameters: dict | None = {}

    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get("name")
        self.datatype = kwargs.get("datatype")
        self.shape = kwargs.get("shape")
        self.data = kwargs.get("data")
        if self.datatype == "BYTES" and "data" in kwargs:
            self.data = [base64.b64decode(d) for d in kwargs.get("data", [])]
        self.parameters = kwargs.get("parameters")

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"ServerlessRequestInput(name={self.name}, datatype={self.datatype}, shape={self.shape}, data={self.data}, parameters={self.parameters})"

    def dict(self) -> dict:
        return {
            "name": self.name,
            "datatype": self.datatype,
            "shape": self.shape,
            "data": self.data,
            "parameters": self.parameters,
        }

    def json(self) -> str:
        return json.dumps(self.dict())

class ServerlessRequestOutput:
    name: str
    datatype: str
    shape: list[int]
    parameters: dict | None = {}

    def __init__(self, **kwargs) -> None:
        self.name = kwargs.get("name")
        self.datatype = kwargs.get("datatype")
        self.shape = kwargs.get("shape")
        self.parameters = kwargs.get("parameters")

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f"ServerlessRequestOutput(name={self.name}, datatype={self.datatype}, shape={self.shape}, parameters={self.parameters})"

    def dict(self) -> dict:
        return {
            "name": self.name,
            "datatype": self.datatype,
            "shape": self.shape,
            "parameters": self.parameters,
        }

    def json(self) -> str:
        return json.dumps(self.dict())

