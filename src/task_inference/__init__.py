# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""task-inference - AI task abstractions for the Open Inference Protocol v2.

Quickstart
----------
Directly::

    from task_inference.implementations.transformers.vision import (
        TransformersImageClassificationTask,
    )
    from task_inference.tasks.vision.image_classification import (
        ImageClassificationInput,
    )

    task = TransformersImageClassificationTask()
    with open("image.jpg", "rb") as f:
        result = task(ImageClassificationInput(image=f.read(), top_k=3))
    for r in result.results:
        print(r.label, r.score)

Via the factory::

    from task_inference import create_task

    task = create_task(
        backend="transformers",
        task_name="image-classification",
        model_name="google/vit-base-patch16-224",
        model_params={"device": "cpu"},
    )
"""

from .factory import create_task, supported_tasks
from .protocol import (
    Datatype,
    InferenceRequest,
    InferenceResponse,
    RequestInput,
    RequestOutput,
    ResponseOutput,
)
from .tasks.base import BaseTask

__all__ = [
    "BaseTask",
    "Datatype",
    "InferenceRequest",
    "InferenceResponse",
    "RequestInput",
    "RequestOutput",
    "ResponseOutput",
]
