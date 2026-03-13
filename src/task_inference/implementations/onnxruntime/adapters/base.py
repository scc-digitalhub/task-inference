# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""Base adapter class and dialect-resolution helpers.

Architecture
------------
A *dialect* is the specific ONNX I/O contract used by a model family
(e.g. "transformers-detr", "yolov8", "torchvision-detection").  Each
dialect is represented by a concrete :class:`OnnxDialectAdapter` subclass.

Auto-detection works in two steps:

1. **Signal collection** — the adapter candidate list and the model's
   declared input/output tensor names (read from the ORT session via
   ``session.get_inputs()`` / ``session.get_outputs()``) plus the parsed
   ``config.json`` are gathered.
2. **Matching** — each candidate's :meth:`OnnxDialectAdapter.accepts`
   classmethod is called in registration order; the first match wins.

Adapters own the ORT session and all preprocessing / postprocessing logic
for their dialect. Each per-task abstract subclass (e.g.
``ObjectDetectionAdapter``) defines the task-specific ``run()`` method.
Concrete subclasses implement both ``accepts()`` and the ``run()`` method.

Usage (inside a task ``__init__``)::

    session = self._create_session(onnx_path, providers)
    self._adapter = resolve_detection_adapter(session, pp_cfg, config)

Usage (inside ``process()``)::

    detections = self._adapter.detect(pil_image, threshold, id2label)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Sequence, Type, TypeVar

T = TypeVar("T", bound="OnnxDialectAdapter")


class OnnxDialectAdapter(ABC):
    """Base class for all ONNX dialect adapters.

    Subclasses must set :attr:`DIALECT` and implement :meth:`accepts`.
    Each per-task abstract subclass also defines the task-specific inference
    method(s).

    Concrete adapters are instantiated with at minimum ``(session, pp_cfg)``
    and store all state needed to run inference.
    """

    DIALECT: ClassVar[str] = "unknown"

    @classmethod
    @abstractmethod
    def accepts(
        cls,
        input_names: Sequence[str],
        output_names: Sequence[str],
        config: dict[str, Any],
    ) -> bool:
        """Return ``True`` if this adapter is compatible with the given model signature.

        Parameters
        ----------
        input_names:
            Tensor names from ``InferenceSession.get_inputs()``.
        output_names:
            Tensor names from ``InferenceSession.get_outputs()``.
        config:
            Parsed ``config.json`` dict (may be empty ``{}``).
        """


def resolve_adapter(
    candidates: Sequence[Type[T]],
    input_names: Sequence[str],
    output_names: Sequence[str],
    config: dict[str, Any],
) -> Type[T]:
    """Return the first matching adapter *class* from *candidates*.

    Parameters
    ----------
    candidates:
        Ordered list of adapter classes.  More-specific dialects should be
        listed before catch-all ones.
    input_names / output_names:
        Tensor names reported by the ORT session.
    config:
        Parsed ``config.json`` (may be empty).

    Returns
    -------
    Type[T]
        The matching adapter *class* (not an instance).  Instantiate with
        the constructor arguments appropriate for the per-task base class.

    Raises
    ------
    ValueError
        When no candidate matches.
    """
    for adapter_cls in candidates:
        if adapter_cls.accepts(input_names, output_names, config):
            return adapter_cls
    raise ValueError(
        f"No adapter matched model signature — "
        f"inputs={list(input_names)}, outputs={list(output_names)}. "
        f"Available dialects: {[c.DIALECT for c in candidates]}"
    )


def io_names_from_session(session: Any) -> tuple[list[str], list[str]]:
    """Extract input and output tensor names from an ORT InferenceSession."""
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    return input_names, output_names
