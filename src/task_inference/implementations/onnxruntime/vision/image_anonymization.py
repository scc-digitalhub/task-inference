# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime implementation - image anonymization (pure ORT, no transformers).

Strategy
--------
1. Run ONNX object detection to locate sensitive regions.
2. Apply the requested anonymization strategy (blur, pixelate, black box)
   to each detected bounding box using Pillow.
3. Return the modified image as PNG bytes.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ....tasks.vision.image_anonymization import (
    AnonymizationStrategy,
    ImageAnonymizationInput,
    ImageAnonymizationOutput,
    ImageAnonymizationTask,
)
from ..base import OnnxRuntimeTaskMixin
from .object_detection import OnnxObjectDetectionTask


class OnnxImageAnonymizationTask(OnnxRuntimeTaskMixin, ImageAnonymizationTask):
    """Image anonymization using a local ONNX object-detection model.

    Runs DETR/YOLOS-style object detection to locate objects, then applies
    a Pillow-based redaction strategy.

    Parameters
    ----------
    model_name:
        **Local directory** path to an ONNX object-detection model
        (e.g. exported DETR or YOLOS).
    device:
        ORT execution device.

    Example
    -------
    ::

        optimum-cli export onnx --model hustvl/yolos-tiny ./onnx/yolos-tiny/
        task = OnnxImageAnonymizationTask(model_name="./onnx/yolos-tiny/")
    """

    def __init__(self, model_name: str, device: str | int = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._od = OnnxObjectDetectionTask(model_name=model_name, device=device)

    # ------------------------------------------------------------------
    # Redaction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_blur(
        image: Image.Image, box: tuple[int, int, int, int], radius: int
    ) -> Image.Image:
        region = image.crop(box)
        image.paste(region.filter(ImageFilter.GaussianBlur(radius=radius)), box)
        return image

    @staticmethod
    def _apply_pixelate(
        image: Image.Image, box: tuple[int, int, int, int], factor: int
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        region = image.crop(box)
        w, h = region.size
        small_w, small_h = max(1, w // factor), max(1, h // factor)
        region = region.resize((small_w, small_h), Image.BOX).resize((w, h), Image.NEAREST)
        image.paste(region, (x1, y1))
        return image

    @staticmethod
    def _apply_black_box(
        image: Image.Image, box: tuple[int, int, int, int]
    ) -> Image.Image:
        ImageDraw.Draw(image).rectangle(box, fill=(0, 0, 0))
        return image

    def _redact(
        self,
        pil_image: Image.Image,
        detections: list,
        inputs: ImageAnonymizationInput,
        classes: set[str] | None,
    ) -> tuple[bytes, int]:
        anon = pil_image.copy()
        count = 0
        for det in detections:
            if classes is not None and det.label.lower() not in classes:
                continue
            if det.score < inputs.threshold:
                continue
            b = det.box
            box = (int(b.xmin), int(b.ymin), int(b.xmax), int(b.ymax))
            if inputs.strategy == AnonymizationStrategy.BLUR:
                anon = self._apply_blur(anon, box, inputs.blur_radius)
            elif inputs.strategy == AnonymizationStrategy.PIXELATE:
                anon = self._apply_pixelate(anon, box, inputs.pixelate_factor)
            else:
                anon = self._apply_black_box(anon, box)
            count += 1
        return self._pil_to_bytes(anon, fmt="PNG"), count

    def process(self, inputs: ImageAnonymizationInput) -> ImageAnonymizationOutput:
        from ....tasks.vision.object_detection import ObjectDetectionInput  # noqa: PLC0415

        classes: set[str] | None = (
            {c.lower() for c in inputs.classes} if inputs.classes else None
        )

        od_output = self._od.process(
            ObjectDetectionInput(
                images=inputs.images,
                threshold=inputs.threshold,
            )
        )

        anonymized_images: list[bytes] = []
        counts: list[int] = []
        for img_bytes, dets in zip(inputs.images, od_output.detections):
            pil_image = self._raw_to_pil(img_bytes)
            img_out, count = self._redact(pil_image, dets, inputs, classes)
            anonymized_images.append(img_out)
            counts.append(count)

        return ImageAnonymizationOutput(
            images=anonymized_images,
            num_regions_anonymized=counts,
        )
