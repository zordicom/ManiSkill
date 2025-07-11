#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Object Detection and Segmentation WebSocket Client

This client communicates with the object detection WebSocket server to perform
object detection and segmentation. It provides a simple API for detection requests.

Usage:
    python ws_det_client.py --host localhost --port 10015 --object-name "box"
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from zordi_policy_rpc.direct.client import DirectClient

logger = logging.getLogger(__name__)


def center_crop_to_square(image: np.ndarray) -> np.ndarray:
    """Center crop image to make it square using the smaller dimension.

    Args:
        image: Input image as numpy array (H, W, C)

    Returns:
        Square image as numpy array after center cropping
    """
    height, width = image.shape[:2]

    # Use the smaller dimension as the crop size
    crop_size = min(width, height)

    # Calculate crop coordinates for center crop
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    # Perform center crop
    cropped_image = image[top:bottom, left:right]

    logger.debug(
        f"Center cropped image from {width}x{height} to {crop_size}x{crop_size}"
    )
    return cropped_image


class ObjectDetectionClient(DirectClient):
    """Client for object detection WebSocket server."""

    def __init__(self, host: str, port: int):
        """Initialize the object detection client.

        Args:
            host: Server hostname
            port: Server port
        """
        super().__init__(host, port)

    def get_server_info(self) -> dict:
        """Get server information from metadata."""
        metadata = self.get_metadata()
        return metadata.get("service_metadata", {})

    def detect_objects(
        self,
        rgb_image: np.ndarray,
        object_name: str,
    ) -> Tuple[
        bool, str, List[List[float]], List[float], List[str], Optional[np.ndarray]
    ]:
        """Detect objects and get segmentation mask.

        Args:
            rgb_image: RGB image (H, W, 3) uint8
            object_name: Name of the target object to detect

        Returns:
            Tuple of (success, message, boxes, scores, labels, mask)
            - success: Whether detection was successful
            - message: Status message
            - boxes: List of bounding boxes in format [[x0, y0, x1, y1], ...]
            - scores: List of detection confidence scores
            - labels: List of object labels
            - mask: Segmentation mask as numpy array (H, W) or None if no mask
        """
        # Encode image
        rgb_bytes = self._encode_image(rgb_image, ".png")

        # Prepare request
        request = {
            "type": "detect",
            "rgb": rgb_bytes,
            "object_name": object_name,
        }

        try:
            response = self.request(request)

            success = response.get("success", False)
            message = response.get("message", "Unknown error")
            num_detections = response.get("num_detections", 0)
            boxes = response.get("boxes", [])
            scores = response.get("scores", [])
            labels = response.get("labels", [])
            mask_bytes = response.get("mask", b"")

            # Decode mask if available
            mask = None
            if mask_bytes and len(mask_bytes) > 0:
                mask = self._decode_mask(mask_bytes)

            if success:
                logger.info(
                    f"Detection successful: {num_detections} objects found for "
                    f"'{object_name}'"
                )
            else:
                logger.error(f"Detection failed: {message}")

            return success, message, boxes, scores, labels, mask

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return False, f"Communication error: {e!s}", [], [], [], None

    def _encode_image(self, image: np.ndarray, ext: str) -> bytes:
        """Encode image as bytes for transmission.

        Args:
            image: Image array
            ext: File extension (e.g., '.png')

        Returns:
            Encoded image bytes
        """
        success, buffer = cv2.imencode(ext, image)
        if not success:
            raise ValueError(f"Failed to encode image with extension {ext}")
        return buffer.tobytes()

    def _decode_mask(self, mask_bytes: bytes) -> np.ndarray:
        """Decode mask from bytes.

        Args:
            mask_bytes: PNG encoded mask bytes

        Returns:
            Binary mask as numpy array (H, W) with values 0 or 1
        """
        # Decode PNG bytes to numpy array
        img_array = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError("Failed to decode mask")

        # Convert to binary mask (0 or 1)
        return (mask > 127).astype(np.uint8)


def load_demo_image(image_path: str) -> np.ndarray:
    """Load demo image for testing.

    Args:
        image_path: Path to the demo image

    Returns:
        RGB image as numpy array (center cropped to square)
    """
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Demo image not found: {image_path}")

    # Load image using cv2 (BGR format)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply center crop to make it square
    rgb_image = center_crop_to_square(rgb_image)

    logger.info(f"Loaded demo image: {image_path}")
    logger.info(f"  Image shape: {rgb_image.shape}")

    return rgb_image


def visualize_results(
    image: np.ndarray,
    boxes: List[List[float]],
    scores: List[float],
    labels: List[str],
    mask: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Visualize detection results with bounding boxes and mask.

    Args:
        image: Original RGB image
        boxes: List of bounding boxes [[x0, y0, x1, y1], ...]
        scores: List of detection scores
        labels: List of object labels
        mask: Optional segmentation mask
        save_path: Optional path to save visualization

    Returns:
        Visualization image as numpy array
    """
    # Convert to PIL for easier drawing
    pil_image = Image.fromarray(image)

    # Apply mask overlay if available
    if mask is not None:
        # Resize mask to match image size if needed
        if mask.shape != (image.shape[0], image.shape[1]):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            mask_pil = mask_pil.resize(
                (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
            )
            mask = np.array(mask_pil) / 255.0

        # Create RGBA overlay
        mask_overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        mask_bool = mask > 0.5
        mask_overlay[mask_bool] = [0, 255, 0, 100]  # Green with transparency

        # Apply overlay
        mask_overlay_pil = Image.fromarray(mask_overlay, "RGBA")
        pil_image = Image.alpha_composite(
            pil_image.convert("RGBA"), mask_overlay_pil
        ).convert("RGB")

    # Convert back to numpy for OpenCV drawing
    vis_image = np.array(pil_image)

    # Draw bounding boxes and labels
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x0, y0, x1, y1 = [int(coord) for coord in box]

        # Draw rectangle
        cv2.rectangle(vis_image, (x0, y0), (x1, y1), (0, 0, 255), 2)  # Red box

        # Draw label and score
        label_text = f"{label}: {score:.2f}"
        cv2.putText(
            vis_image,
            label_text,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),  # Red text
            2,
        )

    # Save if requested
    if save_path:
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        logger.info(f"Visualization saved to: {save_path}")

    return vis_image


def test_object_detection(host: str, port: int, object_name: str, image_path: str):
    """Test object detection with demo image.

    Args:
        host: Server hostname
        port: Server port
        object_name: Object name to detect
        image_path: Path to test image
    """
    logger.info(
        f"Testing object detection for '{object_name}' using image: {image_path}"
    )

    try:
        # Load demo image
        rgb_image = load_demo_image(image_path)

        # Create client and connect
        client = ObjectDetectionClient(host, port)

        with client:
            # Check server info
            server_info = client.get_server_info()
            logger.info(f"Server info: {server_info}")

            # Test detection
            logger.info(f"Running detection for object: '{object_name}'...")

            start_time = time.time()
            success, message, boxes, scores, labels, mask = client.detect_objects(
                rgb_image, object_name
            )
            detection_time = time.time() - start_time

            if not success:
                logger.error(f"Detection failed: {message}")
                return

            logger.info(f"Detection successful in {detection_time:.3f}s: {message}")
            logger.info(f"Found {len(boxes)} detections:")

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                logger.info(f"  Detection {i + 1}:")
                logger.info(
                    f"    Box: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
                )
                logger.info(f"    Score: {score:.3f}")
                logger.info(f"    Label: {label}")

            # Log mask info
            if mask is not None:
                mask_pixels = np.sum(mask > 0)
                total_pixels = mask.shape[0] * mask.shape[1]
                mask_percentage = (mask_pixels / total_pixels) * 100
                logger.info("Segmentation mask:")
                logger.info(f"  Mask shape: {mask.shape}")
                logger.info(
                    f"  Mask coverage: {mask_percentage:.1f}% "
                    f"({mask_pixels}/{total_pixels} pixels)"
                )
            else:
                logger.info("No segmentation mask returned")

            # Create visualization
            vis_dir = Path("./vis_imgs/detection_client")
            vis_dir.mkdir(parents=True, exist_ok=True)

            vis_path = vis_dir / f"{object_name}_detection_result.jpg"
            visualize_results(rgb_image, boxes, scores, labels, mask, str(vis_path))

            # Save mask separately if available
            if mask is not None:
                mask_path = vis_dir / f"{object_name}_mask.png"
                cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
                logger.info(f"Mask saved to: {mask_path}")

            # Test multiple requests for performance
            logger.info("Testing multiple requests for performance...")
            num_requests = 10
            times = []

            for i in range(num_requests):
                start_time = time.time()
                success, _, _, _, _, _ = client.detect_objects(rgb_image, object_name)
                request_time = time.time() - start_time
                times.append(request_time)

                if i % 5 == 0:
                    logger.info(
                        f"  Request {i + 1}/{num_requests}: {request_time:.3f}s"
                    )

            avg_time = np.mean(times)
            std_time = np.std(times)
            logger.info("Performance test completed:")
            logger.info(f"  Average request time: {avg_time:.3f}s ± {std_time:.3f}s")
            logger.info(f"  Min/Max time: {min(times):.3f}s / {max(times):.3f}s")

        logger.info("All tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Object Detection WebSocket Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=10015, help="Server port")
    parser.add_argument(
        "--object-name",
        default="box",
        help="Object name to detect (e.g., 'box', 'gray basket', 'small white box')",
    )
    parser.add_argument(
        "--image-path",
        default="galaxea_human_demos/box_pnp/2025/07/02/20250702_115926/raw_data/static_top_rgb/000000.jpg",
        help="Path to test image",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run test
    test_object_detection(args.host, args.port, args.object_name, args.image_path)


if __name__ == "__main__":
    main()
