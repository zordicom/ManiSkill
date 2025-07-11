#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Multi-Object FoundationPose WebSocket Client (Optimized)

This client provides a high-performance interface to communicate with the Multi-Object
FoundationPose WebSocket server for simultaneous pose estimation of multiple objects.

Key optimizations:
- Direct bytes encoding (no base64 overhead)
- Robust error handling with graceful fallbacks
- Persistent connection management via DirectClient
- Full compatibility with optimized multi_object_server.py

Server Optimization Benefits:
- Fast object re-detection: ~100ms instead of 5-10 seconds
- Pre-initialized estimators at server startup
- No re-initialization needed when objects disappear/reappear
- Seamless handling of non-continuous scenes

Usage:
    from ws_pose_multi_client import MultiObjectFoundationPoseClient

    client = MultiObjectFoundationPoseClient(host="localhost", port=10014)
    with client:
        # Fast object registration (optimized - uses pre-initialized estimators)
        response = client.initialize_objects(objects_data)

        # Or use the explicit method for re-detection
        response = client.register_objects(objects_data)

        # Predict poses (high-performance loop)
        for frame in camera_stream:
            response = client.predict_poses(rgb_image, depth_image)
            # Process poses for robot control
"""

import logging
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from zordi_policy_rpc.direct.client import DirectClient

logger = logging.getLogger(__name__)


class MultiObjectFoundationPoseClient(DirectClient):
    """Client for the Multi-Object FoundationPose WebSocket server."""

    def __init__(self, host: str = "localhost", port: int = 10014):
        """Initialize the client.

        Args:
            host: Server host
            port: Server port
        """
        super().__init__(host, port)
        self.session_active = False

    def encode_image(self, image: np.ndarray, format: str = "png") -> bytes:
        """Encode image to bytes for transmission (optimized version).

        Args:
            image: Image array to encode
            format: Image format ("png" or "jpg")

        Returns:
            Encoded image bytes
        """
        if format == "png":
            success, buffer = cv2.imencode(".png", image)
        elif format == "jpg":
            success, buffer = cv2.imencode(".jpg", image)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if not success:
            raise ValueError(f"Failed to encode image with format {format}")

        return buffer.tobytes()

    def initialize_objects(self, objects_data: List[Dict]) -> Dict:
        """Initialize multiple objects for tracking.

        NOTE: This method now uses pre-initialized estimators on the server side,
        making it much faster (~100ms instead of 5-10 seconds). It can be called
        repeatedly for re-detection when objects disappear and reappear.

        Args:
            objects_data: List of object initialization data, each containing:
                - object_name: str
                - rgb_image: np.ndarray
                - depth_image: np.ndarray
                - mask_image: np.ndarray
                - camera_intrinsics: np.ndarray (optional)

        Returns:
            Server response with poses and status
        """
        encoded_objects = []

        for obj_data in objects_data:
            encoded_obj = {
                "object_name": obj_data["object_name"],
                "rgb": self.encode_image(obj_data["rgb_image"]),
                "depth": self.encode_image(obj_data["depth_image"]),
                "mask": self.encode_image(obj_data["mask_image"]),
            }

            if "camera_intrinsics" in obj_data:
                encoded_obj["camera_intrinsics"] = (
                    obj_data["camera_intrinsics"].flatten().tolist()
                )

            encoded_objects.append(encoded_obj)

        request = {"type": "initialize_multi", "objects": encoded_objects}

        try:
            response = self.request(request)

            if response.get("success", False):
                self.session_active = True
                logger.info(
                    f"Successfully initialized objects: {response.get('initialized_objects', [])}"
                )
            else:
                logger.error(
                    f"Initialization failed: {response.get('message', 'Unknown error')}"
                )

            return response
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            return {
                "success": False,
                "message": f"Communication error: {e!s}",
                "poses": {},
                "initialized_objects": [],
            }

    def register_objects(self, objects_data: List[Dict]) -> Dict:
        """Register/re-detect multiple objects for tracking (optimized for fast re-detection).

        This method provides a clear API for fast object re-detection when objects
        disappear and reappear in the scene. It uses the same optimized backend as
        initialize_objects but with clearer semantics for re-detection scenarios.

        Args:
            objects_data: List of object registration data, each containing:
                - object_name: str
                - rgb_image: np.ndarray
                - depth_image: np.ndarray
                - mask_image: np.ndarray
                - camera_intrinsics: np.ndarray (optional)

        Returns:
            Server response with poses and status
        """
        encoded_objects = []

        for obj_data in objects_data:
            encoded_obj = {
                "object_name": obj_data["object_name"],
                "rgb": self.encode_image(obj_data["rgb_image"]),
                "depth": self.encode_image(obj_data["depth_image"]),
                "mask": self.encode_image(obj_data["mask_image"]),
            }

            if "camera_intrinsics" in obj_data:
                encoded_obj["camera_intrinsics"] = (
                    obj_data["camera_intrinsics"].flatten().tolist()
                )

            encoded_objects.append(encoded_obj)

        request = {"type": "register_objects", "objects": encoded_objects}

        try:
            response = self.request(request)

            if response.get("success", False):
                self.session_active = True
                logger.info(
                    f"Successfully registered objects: {response.get('initialized_objects', [])}"
                )
            else:
                logger.error(
                    f"Registration failed: {response.get('message', 'Unknown error')}"
                )

            return response
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            return {
                "success": False,
                "message": f"Communication error: {e!s}",
                "poses": {},
                "initialized_objects": [],
            }

    def predict_poses(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        object_masks: Optional[Dict[str, np.ndarray]] = None,
        camera_intrinsics: Optional[np.ndarray] = None,
    ) -> Dict:
        """Predict poses for all active objects.

        Args:
            rgb_image: RGB image
            depth_image: Depth image
            object_masks: Optional per-object masks {object_name: mask_image}
            camera_intrinsics: Optional camera intrinsics matrix

        Returns:
            Server response with poses and confidences
        """
        if not self.session_active:
            raise RuntimeError(
                "Session not initialized. Call initialize_objects first."
            )

        request: Dict[str, Any] = {
            "type": "predict_multi",
            "rgb": self.encode_image(rgb_image),
            "depth": self.encode_image(depth_image),
        }

        # Always include masks field, even if empty
        if object_masks:
            encoded_masks = {}
            for obj_name, mask in object_masks.items():
                encoded_masks[obj_name] = self.encode_image(mask)
            request["masks"] = encoded_masks
        else:
            request["masks"] = {}  # Empty masks dict for server compatibility

        if camera_intrinsics is not None:
            request["camera_intrinsics"] = camera_intrinsics.flatten().tolist()
        else:
            request["camera_intrinsics"] = []  # Empty list for server compatibility

        try:
            response = self.request(request)
            return response
        except Exception as e:
            logger.error(f"Error during pose prediction: {e}")
            return {
                "poses": {},
                "confidences": {},
                "frame_id": -1,
                "error": f"Communication error: {e!s}",
            }

    def reset_session(self) -> Dict:
        """Reset the tracking session.

        Returns:
            Server response with success status and message
        """
        try:
            response = self.request({"type": "reset_session"})

            if response.get("success", False):
                self.session_active = False
                logger.info("Session reset successfully")

            return response
        except Exception as e:
            logger.error(f"Error during session reset: {e}")
            return {"success": False, "message": f"Communication error: {e!s}"}

    def pose_7d_to_matrix(self, pose_7d: List[float]) -> np.ndarray:
        """Convert 7D pose [x, y, z, qx, qy, qz, qw] to 4x4 matrix.

        Args:
            pose_7d: 7-element pose list [x, y, z, qx, qy, qz, qw]

        Returns:
            4x4 transformation matrix
        """
        translation = np.array(pose_7d[:3])
        quaternion = np.array(pose_7d[3:7])  # [qx, qy, qz, qw]

        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = translation

        return pose_matrix

    def get_available_objects(self) -> List[str]:
        """Get list of available objects from server metadata.

        Returns:
            List of available object names
        """
        try:
            metadata = self.get_metadata()
            return metadata.get("service_metadata", {}).get("available_objects", [])
        except Exception as e:
            logger.warning(f"Failed to get available objects: {e}")
            return []

    def get_default_camera_intrinsics(self) -> Optional[np.ndarray]:
        """Get default camera intrinsics from server metadata.

        Returns:
            Camera intrinsics matrix (3, 3) or None if not available
        """
        try:
            metadata = self.get_metadata()
            default_K = metadata.get("service_metadata", {}).get(
                "default_camera_intrinsics"
            )
            if default_K:
                return np.array(default_K).reshape(3, 3)
        except Exception as e:
            logger.warning(f"Failed to get default camera intrinsics: {e}")
        return None

    def get_server_optimization_info(self) -> Dict[str, Any]:
        """Get server optimization information.

        Returns:
            Dictionary with optimization status and capabilities
        """
        try:
            metadata = self.get_metadata()
            optimization_info = metadata.get("service_metadata", {}).get(
                "optimization_info", {}
            )

            # Add some computed info
            optimization_info["server_optimized"] = optimization_info.get(
                "pre_initialized_estimators", False
            )
            optimization_info["fast_detection_available"] = optimization_info.get(
                "fast_re_detection", False
            )

            return optimization_info
        except Exception as e:
            logger.warning(f"Failed to get server optimization info: {e}")
            return {
                "server_optimized": False,
                "fast_detection_available": False,
                "error": str(e),
            }

    def is_server_optimized(self) -> bool:
        """Check if the server has pre-initialized estimators for fast detection.

        Returns:
            True if server is optimized, False otherwise
        """
        optimization_info = self.get_server_optimization_info()
        return optimization_info.get("server_optimized", False)


def main():
    """Example usage of the MultiObjectFoundationPoseClient."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Object FoundationPose Client Example"
    )
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=10014, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create client and test connection
    client = MultiObjectFoundationPoseClient(args.host, args.port)

    try:
        with client:
            # Test server connection
            logger.info("Testing connection to multi-object pose server...")

            # Get available objects
            available_objects = client.get_available_objects()
            logger.info(f"Available objects: {available_objects}")

            # Get default camera intrinsics
            default_K = client.get_default_camera_intrinsics()
            if default_K is not None:
                logger.info(f"Default camera intrinsics available: {default_K.shape}")
            else:
                logger.info("No default camera intrinsics available")

            # Get server optimization info
            optimization_info = client.get_server_optimization_info()
            logger.info(f"Server Optimization Info: {optimization_info}")
            logger.info(f"Is server optimized: {client.is_server_optimized()}")

            logger.info("Connection test successful!")

    except Exception as e:
        logger.error(f"Connection test failed: {e}")


if __name__ == "__main__":
    main()
