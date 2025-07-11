#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Calibration utilities for camera-world coordinate transformations.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class CalibrationData:
    """Container for camera calibration data with transformation utilities."""

    def __init__(self, calibration_dict: dict[str, Any]) -> None:
        """Initialize from calibration dictionary.

        Args:
            calibration_dict: Dictionary loaded from calibration JSON file
        """
        self.calibration_dict = calibration_dict
        self.metadata = calibration_dict.get("metadata", {})
        self.camera_parameters = calibration_dict.get("camera_parameters", {})

        # Extract camera-to-world transformation matrix
        transform_data = calibration_dict.get("camera_to_world_transform", {})
        self.camera_to_world_matrix = np.array(transform_data.get("matrix", np.eye(4)))

        # Validate matrix dimensions
        if self.camera_to_world_matrix.shape != (4, 4):
            raise ValueError(
                f"Invalid camera-to-world matrix shape: {self.camera_to_world_matrix.shape}, "
                "expected (4, 4)"
            )

        # Extract camera parameters
        self.camera_matrix = np.array(
            self.camera_parameters.get("camera_matrix", np.eye(3))
        )
        self.distortion_coefficients = np.array(
            self.camera_parameters.get("distortion_coefficients", np.zeros(5))
        )

        # Store calibration metadata
        self.tag_size = self.metadata.get("tag_size_m", 0.04)
        self.num_detections = self.metadata.get("num_detections", 0)
        self.calibration_date = self.metadata.get("calibration_date", "Unknown")

    def transform_pose_camera_to_world(
        self, camera_pose_7d: list[float]
    ) -> list[float]:
        """Transform a 7D pose from camera coordinates to world coordinates.

        Args:
            camera_pose_7d: 7D pose in camera coordinates [x, y, z, qx, qy, qz, qw]

        Returns:
            7D pose in world coordinates [x, y, z, qx, qy, qz, qw]
        """
        # Convert 7D pose to 4x4 transformation matrix
        camera_pose_matrix = self.pose_7d_to_matrix(camera_pose_7d)

        # Apply camera-to-world transformation
        world_pose_matrix = self.camera_to_world_matrix @ camera_pose_matrix

        # Convert back to 7D pose
        return self.matrix_to_pose_7d(world_pose_matrix)

    def transform_poses_camera_to_world(
        self, camera_poses_7d: dict[str, list[float]]
    ) -> dict[str, list[float]]:
        """Transform multiple 7D poses from camera coordinates to world coordinates.

        Args:
            camera_poses_7d: Dictionary of 7D poses in camera coordinates

        Returns:
            Dictionary of 7D poses in world coordinates
        """
        world_poses = {}
        for name, pose in camera_poses_7d.items():
            world_poses[name] = self.transform_pose_camera_to_world(pose)
        return world_poses

    @staticmethod
    def pose_7d_to_matrix(pose_7d: list[float]) -> np.ndarray:
        """Convert 7D pose to 4x4 transformation matrix.

        Args:
            pose_7d: 7D pose [x, y, z, qx, qy, qz, qw]

        Returns:
            4x4 transformation matrix
        """
        if len(pose_7d) != 7:
            raise ValueError(f"Expected 7D pose, got {len(pose_7d)}D")

        # Extract translation and quaternion
        translation = np.array(pose_7d[:3])
        quaternion = np.array(pose_7d[3:7])  # [qx, qy, qz, qw]

        # Convert quaternion to rotation matrix
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        return transform_matrix

    @staticmethod
    def matrix_to_pose_7d(matrix: np.ndarray) -> list[float]:
        """Convert 4x4 transformation matrix to 7D pose.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            7D pose [x, y, z, qx, qy, qz, qw]
        """
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix, got {matrix.shape}")

        # Extract translation
        translation = matrix[:3, 3]

        # Extract rotation matrix and convert to quaternion
        rotation_matrix = matrix[:3, :3]
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()  # [qx, qy, qz, qw]

        # Combine into 7D pose
        return translation.tolist() + quaternion.tolist()

    def get_camera_info(self) -> dict[str, Any]:
        """Get camera intrinsic parameters.

        Returns:
            Dictionary with camera matrix and distortion coefficients
        """
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "camera_matrix_np": self.camera_matrix,
            "distortion_coefficients_np": self.distortion_coefficients,
        }

    def get_metadata(self) -> dict[str, Any]:
        """Get calibration metadata.

        Returns:
            Dictionary with calibration metadata
        """
        return {
            "tag_size_m": self.tag_size,
            "num_detections": self.num_detections,
            "calibration_date": self.calibration_date,
            "camera_to_world_matrix": self.camera_to_world_matrix.tolist(),
        }

    def __str__(self) -> str:
        """String representation of calibration data."""
        return (
            f"CalibrationData(\n"
            f"  date='{self.calibration_date}',\n"
            f"  detections={self.num_detections},\n"
            f"  tag_size={self.tag_size}m,\n"
            f"  camera_matrix_shape={self.camera_matrix.shape},\n"
            f"  transform_matrix_shape={self.camera_to_world_matrix.shape}\n"
            f")"
        )


def load_calibration(file_path: str | Path) -> CalibrationData:
    """Load camera calibration data from JSON file.

    Args:
        file_path: Path to calibration JSON file

    Returns:
        CalibrationData object with transformation utilities

    Raises:
        FileNotFoundError: If calibration file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
        ValueError: If calibration data is invalid
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            calibration_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in calibration file: {e}")

    # Validate required fields
    required_fields = ["metadata", "camera_parameters", "camera_to_world_transform"]
    for field in required_fields:
        if field not in calibration_dict:
            raise ValueError(f"Missing required field in calibration: {field}")

    return CalibrationData(calibration_dict)


def transform_pose_camera_to_world(
    camera_pose_7d: list[float], calibration_file: str | Path
) -> list[float]:
    """Convenience function to transform a single pose using calibration file.

    Args:
        camera_pose_7d: 7D pose in camera coordinates [x, y, z, qx, qy, qz, qw]
        calibration_file: Path to calibration JSON file

    Returns:
        7D pose in world coordinates [x, y, z, qx, qy, qz, qw]
    """
    calibration = load_calibration(calibration_file)
    return calibration.transform_pose_camera_to_world(camera_pose_7d)


def transform_poses_camera_to_world(
    camera_poses_7d: dict[str, list[float]], calibration_file: str | Path
) -> dict[str, list[float]]:
    """Convenience function to transform multiple poses using calibration file.

    Args:
        camera_poses_7d: Dictionary of 7D poses in camera coordinates
        calibration_file: Path to calibration JSON file

    Returns:
        Dictionary of 7D poses in world coordinates
    """
    calibration = load_calibration(calibration_file)
    return calibration.transform_poses_camera_to_world(camera_poses_7d)


# Example usage and testing functions
def test_calibration_utils() -> None:
    """Test calibration utilities with sample data."""
    print("Testing calibration utilities...")

    # Test pose conversion functions
    test_pose_7d = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]  # Sample pose

    # Convert to matrix and back
    matrix = CalibrationData.pose_7d_to_matrix(test_pose_7d)
    recovered_pose = CalibrationData.matrix_to_pose_7d(matrix)

    print(f"Original pose: {test_pose_7d}")
    print(f"Recovered pose: {recovered_pose}")
    print(
        f"Conversion error: {np.linalg.norm(np.array(test_pose_7d) - np.array(recovered_pose))}"
    )

    # Test with actual calibration file if it exists
    calibration_file = Path("camera_calibration.json")
    if calibration_file.exists():
        print(f"\nTesting with actual calibration file: {calibration_file}")

        try:
            calibration = load_calibration(calibration_file)
            print(f"Loaded calibration: {calibration}")

            # Test transformation
            test_camera_pose = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
            world_pose = calibration.transform_pose_camera_to_world(test_camera_pose)

            print(f"Test camera pose: {test_camera_pose}")
            print(f"Transformed world pose: {world_pose}")

        except Exception as e:
            print(f"Error testing with calibration file: {e}")
    else:
        print(f"Calibration file not found: {calibration_file}")


if __name__ == "__main__":
    test_calibration_utils()
