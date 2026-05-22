from __future__ import annotations

from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image


def altitude_to_class_index(height: float, h_min: float = 100.0, delta_h: float = 50.0) -> int:
    """Map a metric altitude to a zero-based fixed-interval altitude class."""
    if height < h_min:
        raise ValueError(f"height={height} is below h_min={h_min}")
    return int((height - h_min) // delta_h)


def class_to_altitude(class_index: int, h_min: float = 100.0, delta_h: float = 50.0) -> float:
    """Map a zero-based fixed-interval class index to its center altitude."""
    if class_index < 0:
        raise ValueError("class_index must be non-negative")
    return h_min + (class_index + 0.5) * delta_h


def lower_bound_class_id_to_altitude(class_id: int, delta_h: float = 50.0) -> float:
    """Map this codebase's lower-bound class IDs, such as 100 or 150, to centers."""
    return class_id + 0.5 * delta_h


def adjust_altitude_for_focal_length(
    estimated_altitude: float,
    deployed_focal_length: float,
    nominal_focal_length: float,
) -> float:
    """Apply the paper's focal-length-ratio correction for deployment cameras."""
    if nominal_focal_length <= 0 or deployed_focal_length <= 0:
        raise ValueError("focal lengths must be positive")
    return estimated_altitude * deployed_focal_length / nominal_focal_length


def _to_hw(size: Sequence[int] | int) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) == 1:
        return int(size[0]), int(size[0])
    if len(size) == 2:
        return int(size[0]), int(size[1])
    raise ValueError("size must be an int or a sequence of one/two ints")


def crop_to_canonical_altitude(
    image: Image.Image | torch.Tensor,
    estimated_altitude: float,
    canonical_altitude: float = 125.0,
    output_size: Sequence[int] | int | None = None,
) -> Image.Image | torch.Tensor:
    """Resize by H/H_db and center-crop back to the primitive-map scale.

    This implements the paper's Crop(I_in, H_hat) operation. For a PIL image,
    the returned object is a PIL image. For a tensor, the expected shape is
    [..., H, W] and the returned object is a tensor.
    """
    if estimated_altitude <= 0 or canonical_altitude <= 0:
        raise ValueError("altitudes must be positive")
    scale = estimated_altitude / canonical_altitude

    if isinstance(image, Image.Image):
        width, height = image.size
        target_height, target_width = _to_hw(output_size) if output_size is not None else (height, width)
        resized = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
        return TF.center_crop(resized, [target_height, target_width])

    if torch.is_tensor(image):
        height, width = image.shape[-2:]
        target_height, target_width = _to_hw(output_size) if output_size is not None else (height, width)
        resized = TF.resize(image, [round(height * scale), round(width * scale)], antialias=True)
        return TF.center_crop(resized, [target_height, target_width])

    raise TypeError("image must be a PIL.Image.Image or torch.Tensor")
