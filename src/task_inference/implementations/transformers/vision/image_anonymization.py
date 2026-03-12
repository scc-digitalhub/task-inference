# SPDX-FileCopyrightText: © 2026 DSLab - Fondazione Bruno Kessler
#
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Transformers reference implementation - image anonymization.

Strategy
--------
1. Run an object detection pipeline to locate sensitive regions (faces,
   persons, or any user-specified classes).
2. Apply the requested anonymization strategy (blur, pixelate, black box)
   to each detected bounding box using Pillow.
3. Return the modified image as a PNG-encoded BYTES tensor.

The default model (``hustvl/yolos-tiny``) is a lightweight YOLOS detector.
For better face-detection accuracy, swap in a dedicated face-detection model
such as ``PekingU/rtdetr_r50vd_coco_o365``.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFilter
from transformers import pipeline

from ....tasks.vision.image_anonymization import (
    AnonymizationStrategy,
    ImageAnonymizationInput,
    ImageAnonymizationOutput,
    ImageAnonymizationTask,
)
from ..base import TransformersTaskMixin

_DEFAULT_MODEL = "hustvl/yolos-tiny"
_DEFAULT_DETECT_CLASSES = {"person", "face"}


class TransformersImageAnonymizationTask(
    TransformersTaskMixin, ImageAnonymizationTask
):
    """Image anonymization backed by an object detection pipeline.

    Parameters
    ----------
    model_name:
        An ``object-detection`` compatible model used to locate sensitive
        regions.  Defaults to ``hustvl/yolos-tiny``.
    device:
        Inference device.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str | int = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._pipe = pipeline(
            "object-detection",
            model=model_name,
            device=device,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_blur(
        image: Image.Image,
        box: tuple[int, int, int, int],
        radius: int,
    ) -> Image.Image:
        region = image.crop(box)
        blurred = region.filter(ImageFilter.GaussianBlur(radius=radius))
        image.paste(blurred, box)
        return image

    @staticmethod
    def _apply_pixelate(
        image: Image.Image,
        box: tuple[int, int, int, int],
        factor: int,
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        region = image.crop(box)
        w, h = region.size
        small_w = max(1, w // factor)
        small_h = max(1, h // factor)
        region = region.resize((small_w, small_h), Image.BOX)
        region = region.resize((w, h), Image.NEAREST)
        image.paste(region, box[:2])
        return image

    @staticmethod
    def _apply_black_box(
        image: Image.Image,
        box: tuple[int, int, int, int],
    ) -> Image.Image:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, fill=(0, 0, 0))
        return image

    # ------------------------------------------------------------------
    # Main process method
    # ------------------------------------------------------------------

    def _anonymize_image(
        self,
        pil_image: Image.Image,
        detections: list[dict],
        inputs: ImageAnonymizationInput,
        classes: set[str] | None,
    ) -> tuple[bytes, int]:
        """Apply anonymization to one image given pre-computed detections."""
        anonymized = pil_image.copy()
        count = 0

        for det in detections:
            label = det.get("label", "").lower()
            if classes is not None and label not in classes:
                continue
            b = det.get("box", {})
            box = (
                int(b.get("xmin", 0)),
                int(b.get("ymin", 0)),
                int(b.get("xmax", 0)),
                int(b.get("ymax", 0)),
            )
            if inputs.strategy == AnonymizationStrategy.BLUR:
                anonymized = self._apply_blur(anonymized, box, inputs.blur_radius)
            elif inputs.strategy == AnonymizationStrategy.PIXELATE:
                anonymized = self._apply_pixelate(anonymized, box, max(2, inputs.blur_radius // 5))
            else:  # BLACK_BOX
                anonymized = self._apply_black_box(anonymized, box)
            count += 1

        return self._pil_to_bytes(anonymized, fmt="PNG"), count

    def process(self, inputs: ImageAnonymizationInput) -> ImageAnonymizationOutput:
        classes: set[str] | None = (
            {c.lower() for c in inputs.classes}
            if inputs.classes is not None
            else None
        )

        pil_images = [self._raw_to_pil(img).convert("RGB") for img in inputs.images]

        # Single batched pipeline call for all images
        batch_detections = self._pipe(pil_images, threshold=inputs.threshold)

        result_images: list[bytes] = []
        result_counts: list[int] = []
        for pil_image, detections in zip(pil_images, batch_detections):
            img_out, count = self._anonymize_image(pil_image, detections, inputs, classes)
            result_images.append(img_out)
            result_counts.append(count)

        return ImageAnonymizationOutput(
            images=result_images,
            num_regions_anonymized=result_counts,
        )
