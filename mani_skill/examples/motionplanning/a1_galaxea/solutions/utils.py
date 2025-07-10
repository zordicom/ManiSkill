"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def center_crop_and_resize_image(
    image: Optional[np.ndarray],
    image_size: Tuple[int, int],
    pad_ratio: float = 0.0,
) -> np.ndarray:
    """Perform center crop with padding based on the shorter dimension.

    Args:
        image: Input image as numpy array
        image_size: Tuple of (height, width)
        pad_ratio: Ratio to determine padding size (e.g., 0.2 for 20% padding)

    Returns:
        Cropped image as numpy array

    """
    if image is None or image.size == 0:
        return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    if image.shape[:2] == image_size:
        return image

    height, width = image.shape[:2]
    shorter_dim = min(width, height)
    crop_size = int(shorter_dim * (1.0 - pad_ratio))

    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2

    cropped = image[start_y : start_y + crop_size, start_x : start_x + crop_size]

    # Use appropriate interpolation method based on image type
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):  # noqa: PLR2004
        # Single-channel image (likely depth) - use nearest neighbor to preserve values
        interpolation = cv2.INTER_NEAREST
    elif image.dtype in {np.uint16, np.int16}:
        # 16-bit images (depth images) - use nearest neighbor to preserve depth values
        interpolation = cv2.INTER_NEAREST
    else:
        # Multi-channel RGB images - use bilinear interpolation for smooth results
        interpolation = cv2.INTER_LINEAR

    resized = cv2.resize(cropped, image_size, interpolation=interpolation)
    return resized


def joint_states_to_str(joint_states: np.ndarray) -> str:
    """Convert joint states to a string."""
    return "[" + ", ".join(f"{state:7.4f}" for state in joint_states) + "]"


def adjust_k_for_crop_and_resize(
    k: np.ndarray,
    orig_size: tuple[int, int],
    final_size: tuple[int, int],
    pad_ratio: float = 0.0,
) -> np.ndarray:
    """
    Adjust camera intrinsics after center crop and resize operations.

    This function handles the same transformations as center_crop_and_resize_image:
    1. Center crop based on shorter dimension with optional padding
    2. Resize to final dimensions

    Args:
        k: (3, 3) intrinsic matrix from original image.
        orig_size: (width, height) of the original image.
        final_size: (width, height) of the final processed image.
        pad_ratio: Padding ratio used in cropping (default: 0.0).

    Returns:
        New (3, 3) intrinsic matrix for the processed image.
    """
    k_new = k.copy()
    ow, oh = orig_size
    fw, fh = final_size

    # Step 1: Handle center crop transformation
    # Calculate crop size based on shorter dimension
    # (matching center_crop_and_resize_image logic)
    shorter_dim = min(ow, oh)
    crop_size = int(shorter_dim * (1.0 - pad_ratio))

    # Calculate crop offsets (center crop)
    crop_offset_x = (ow - crop_size) / 2
    crop_offset_y = (oh - crop_size) / 2

    # Adjust principal point for crop offset
    k_new[0, 2] -= crop_offset_x  # shift cx
    k_new[1, 2] -= crop_offset_y  # shift cy

    # Step 2: Handle resize transformation
    # Calculate scale factors for resize operation
    scale_x = fw / crop_size
    scale_y = fh / crop_size

    # Scale focal lengths and principal point
    k_new[0, 0] *= scale_x  # scale fx
    k_new[1, 1] *= scale_y  # scale fy
    k_new[0, 2] *= scale_x  # scale cx
    k_new[1, 2] *= scale_y  # scale cy

    return k_new


def colorize_depth_image(
    depth_image: np.ndarray, min_depth: float = 0.05, max_depth: float = 2.0
) -> np.ndarray:
    """Colorize depth image using Intel's HUE-based colorization technique.

    Args:
        depth_image: 16-bit depth image in millimeters
        min_depth: Minimum depth in meters
        max_depth: Maximum depth in meters

    Returns:
        Colorized BGR image that can be compressed with standard codecs
    """
    # Convert depth from mm to meters and clip to range
    depth_m = depth_image.astype(np.float32) / 1000.0
    depth_m = np.clip(depth_m, min_depth, max_depth)

    # Normalize depth to 0-1 range
    depth_normalized = (depth_m - min_depth) / (max_depth - min_depth)

    # Create HSV image
    hsv = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

    # Map depth to HUE (0-179 in OpenCV HSV)
    hsv[:, :, 0] = (depth_normalized * 179).astype(np.uint8)

    # Set saturation and value to maximum for valid pixels
    valid_mask = depth_image > 0
    hsv[:, :, 1] = np.where(valid_mask, 255, 0)  # Saturation
    hsv[:, :, 2] = np.where(valid_mask, 255, 0)  # Value

    # Convert HSV to BGR
    bgr_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr_image


def recover_depth_from_colorized(
    colorized_image: np.ndarray, min_depth: float = 0.05, max_depth: float = 2.0
) -> np.ndarray:
    """Recover depth image from colorized image.

    Args:
        colorized_image: BGR colorized depth image
        min_depth: Minimum depth in meters (same as used for colorization)
        max_depth: Maximum depth in meters (same as used for colorization)

    Returns:
        Recovered 16-bit depth image in millimeters
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(colorized_image, cv2.COLOR_BGR2HSV)

    # Extract HUE channel and normalize
    hue = hsv[:, :, 0].astype(np.float32) / 179.0

    # Convert normalized hue back to depth
    depth_m = hue * (max_depth - min_depth) + min_depth

    # Convert to millimeters and 16-bit
    depth_mm = (depth_m * 1000.0).astype(np.uint16)

    # Zero out invalid pixels (where saturation is 0)
    valid_mask = hsv[:, :, 1] > 0
    depth_mm = np.where(valid_mask, depth_mm, 0)

    return depth_mm


def load_depth_image(
    depth_image_path: str, metadata_path: str | None = None
) -> np.ndarray:
    """Load and recover depth image, handling both colorized and traditional formats.

    Args:
        depth_image_path: Path to the depth image file
        metadata_path: Path to depth metadata JSON (for colorized images)

    Returns:
        16-bit depth image in millimeters
    """
    depth_path = Path(depth_image_path)

    if depth_path.suffix.lower() == ".png":
        # Traditional 16-bit PNG depth image
        return cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    elif depth_path.suffix.lower() in {".jpg", ".jpeg"}:
        # Colorized depth image - need metadata for recovery
        if metadata_path is None:
            metadata_path = depth_path.parent / "depth_metadata.json"

        if not Path(metadata_path).exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Load colorized image and recover depth
        colorized_img = cv2.imread(str(depth_path), cv2.IMREAD_COLOR)
        return recover_depth_from_colorized(
            colorized_img, metadata["min_depth"], metadata["max_depth"]
        )

    else:
        raise ValueError(f"Unsupported depth image format: {depth_path.suffix}")


def apply_flying_pixel_filter(
    depth_image: np.ndarray, kernel_size: int = 3
) -> np.ndarray:
    """Apply flying pixel filter to depth image (useful for compressed depth).

    Based on Intel's recommendation for post-processing compressed depth images.

    Args:
        depth_image: 16-bit depth image in millimeters
        kernel_size: Size of the median filter kernel

    Returns:
        Filtered depth image
    """
    # Apply median filter to remove flying pixels
    filtered = cv2.medianBlur(depth_image, kernel_size)

    # Only apply filter to non-zero pixels to preserve edges
    mask = depth_image > 0
    result = np.where(mask, filtered, depth_image)

    return result


def custom_json_dumps(obj: dict, max_indent_level: int = 3) -> str:
    """Custom JSON serializer that indents only up to specified level.

    Args:
        obj: Dictionary to serialize
        max_indent_level: Maximum depth to apply indentation (default: 3)

    Returns:
        JSON string with custom indentation
    """

    def _serialize_with_level(obj, level=0, indent_size=2):
        if level >= max_indent_level:
            # Beyond max level, serialize inline without indentation
            return json.dumps(obj, separators=(",", ":"))

        if isinstance(obj, dict):
            if not obj:
                return "{}"

            indent = " " * (level * indent_size)
            next_indent = " " * ((level + 1) * indent_size)

            items = []
            for key, value in obj.items():
                key_str = json.dumps(key)
                if level + 1 >= max_indent_level:
                    # Next level should be inline
                    value_str = json.dumps(value, separators=(",", ":"))
                else:
                    value_str = _serialize_with_level(value, level + 1, indent_size)
                items.append(f"{next_indent}{key_str}: {value_str}")

            return "{\n" + ",\n".join(items) + f"\n{indent}}}"

        elif isinstance(obj, list):
            if not obj:
                return "[]"

            indent = " " * (level * indent_size)
            next_indent = " " * ((level + 1) * indent_size)

            items = []
            for item in obj:
                if level + 1 >= max_indent_level:
                    # Next level should be inline
                    item_str = json.dumps(item, separators=(",", ":"))
                else:
                    item_str = _serialize_with_level(item, level + 1, indent_size)
                items.append(f"{next_indent}{item_str}")

            return "[\n" + ",\n".join(items) + f"\n{indent}]"

        else:
            # Primitive value
            return json.dumps(obj)

    return _serialize_with_level(obj)
