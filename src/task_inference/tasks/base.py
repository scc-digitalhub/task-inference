# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Abstract base class for all AI task services."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from ..protocol.v2 import InferenceRequest, InferenceResponse, MetadataTensor, ModelMetadataResponse, ParameterMetadata

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseTask(ABC, Generic[InputT, OutputT]):
    """Abstract base class for every AI task.

    Responsibilities
    ----------------
    * Declares the task's **input** and **output** Pydantic schemas via
      :attr:`INPUT_SCHEMA` / :attr:`OUTPUT_SCHEMA`.
    * Exposes :meth:`__call__` as the **OIP v2 boundary**: accepts an
      :class:`~task_inference.protocol.v2.InferenceRequest` and returns an
      :class:`~task_inference.protocol.v2.InferenceResponse`.

    * Declares :meth:`process` as the **extension point**: concrete backends
      (e.g. the Transformers reference implementation) override this method to
      perform actual inference.

    Usage
    -----
    Implement a new backend by subclassing the concrete task (e.g.
    :class:`~task_inference.tasks.vision.image_classification.ImageClassificationTask`)
    and overriding :meth:`process`::

        class MyBackendTask(ImageClassificationTask):
            def process(self, inputs: ImageClassificationInput) -> ImageClassificationOutput:
                ...
    """

    #: Unique task identifier (mirrors the HuggingFace pipeline task name).
    TASK_NAME: str = ""

    #: Pydantic model class for task inputs.  Subclasses *must* set this.
    INPUT_SCHEMA: type[InputT]  # type: ignore[assignment]

    #: Pydantic model class for task outputs.  Subclasses *must* set this.
    OUTPUT_SCHEMA: type[OutputT]  # type: ignore[assignment]

    #: OIP v2 input tensor descriptors for :meth:`get_metadata`.
    METADATA_INPUTS: list[MetadataTensor] = []

    #: OIP v2 output tensor descriptors for :meth:`get_metadata`.
    METADATA_OUTPUTS: list[MetadataTensor] = []

    #: Inference parameter descriptors for :meth:`get_metadata` (extension).
    METADATA_PARAMETERS: list[ParameterMetadata] | None = None

    # ------------------------------------------------------------------
    # Abstract process method
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, inputs: InputT) -> OutputT:
        """Execute inference and return the task's native output object.

        This is the primary **extension point**.  Concrete backend classes
        (e.g. the HuggingFace Transformers reference implementation) override
        this method to perform actual model inference.

        Parameters
        ----------
        inputs:
            An instance of :attr:`INPUT_SCHEMA`.

        Returns
        -------
        OutputT
            An instance of :attr:`OUTPUT_SCHEMA`.
        """

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __call__(self, request: InferenceRequest) -> InferenceResponse:
        """Run the OIP v2 round-trip: deserialise request, run :meth:`process`, serialise response."""
        inputs: InputT = self.INPUT_SCHEMA.from_inference_request(request)
        output: OutputT = self.process(inputs)
        return output.to_inference_response(self.TASK_NAME)

    def run(self, **kwargs: Any) -> InferenceResponse:
        """Convenience wrapper - build an :attr:`INPUT_SCHEMA` instance from
        keyword arguments, convert to an :class:`~task_inference.protocol.v2.InferenceRequest`,
        and run :meth:`__call__`.

        Example
        -------
        ::

            response = task.run(image=image_bytes, top_k=5)
        """
        inputs: InputT = self.INPUT_SCHEMA(**kwargs)
        return self(inputs.to_inference_request())

    def get_metadata(
        self,
        *,
        model_name: str | None = None,
        platform: str = "",
        versions: list[str] | None = None,
    ) -> ModelMetadataResponse:
        """Return an OIP v2 model metadata response for this task.

        The returned object describes the input and output tensors (per the
        ``$metadata_model_response`` OIP v2 schema) and, as a non-standard
        extension, the inference parameters accepted in the request's
        ``parameters`` map.

        Parameters
        ----------
        model_name:
            Name to embed in the response.  Defaults to :attr:`TASK_NAME`.
        platform:
            Framework/backend string (e.g. ``"pytorch_torchscript"``).
            Defaults to an empty string at the abstract task level; backend
            implementations may override :attr:`METADATA_INPUTS` /
            :attr:`METADATA_OUTPUTS` or pass a specific value here.
        versions:
            Optional list of model version strings.
        """
        return ModelMetadataResponse(
            name=model_name if model_name is not None else self.TASK_NAME,
            versions=versions,
            platform=platform,
            inputs=self.METADATA_INPUTS,
            outputs=self.METADATA_OUTPUTS,
            parameters=self.METADATA_PARAMETERS if self.METADATA_PARAMETERS else None,
        )
