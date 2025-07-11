#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Camera calibration script using AprilTag detection.
This script detects an AprilTag placed at the world origin and calculates
the camera-world transformation matrix for coordinate system conversion.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

try:
    import apriltag
except ImportError:
    print("AprilTag library not found. Install with: pip install apriltag")
    print("Run: pip install apriltag")
    sys.exit(1)

# Configuration constants
CAMERA_TOPIC = "/camera/static_rs405_top/color/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/static_rs405_top/color/camera_info"
QUEUE_SIZE = 10
TAG_FAMILY = "tag36h11"
TAG_ID = 11
NUM_DETECTIONS = 100  # Number of detections to average
DETECTION_TIMEOUT = 30.0  # Timeout in seconds
TAG_SIZE = 0.04  # AprilTag size in meters (4cm), adjust as needed
DEFAULT_OUTPUT_FILE = "camera_calibration.json"


class AprilTagCalibration(Node):
    """
    ROS2 node for camera calibration using AprilTag detection.

    This node subscribes to a camera stream, detects AprilTags, and calculates
    the camera-world transformation matrix by averaging multiple detections.
    """

    def __init__(
        self,
        node_name: str = "apriltag_calibration",
        output_file: str = DEFAULT_OUTPUT_FILE,
        tag_size: float = TAG_SIZE,
        num_detections: int = NUM_DETECTIONS,
    ) -> None:
        super().__init__(node_name)

        self.output_file = Path(output_file)
        self.tag_size = tag_size
        self.num_detections = num_detections

        # Initialize AprilTag detector
        self.detector = apriltag.Detector(apriltag.DetectorOptions(families=TAG_FAMILY))

        # CV Bridge for ROS image conversion
        self.bridge = CvBridge()

        # Shutdown signal
        self.shutdown_future = rclpy.Future()

        # Camera parameters
        self.camera_matrix: np.ndarray | None = None
        self.dist_coeffs: np.ndarray | None = None
        self.camera_info_received = False

        # Detection storage
        self.detections: list[dict[str, Any]] = []
        self.detection_count = 0
        self.start_time = time.time()

        # Store latest image for validation
        self.latest_image: np.ndarray | None = None

        # Create subscribers
        self.image_subscriber = self.create_subscription(
            Image, CAMERA_TOPIC, self.image_callback, QUEUE_SIZE
        )
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, QUEUE_SIZE
        )

        self.get_logger().info("AprilTag calibration node initialized")
        self.get_logger().info(f"Target: {TAG_FAMILY} tag ID {TAG_ID}")
        self.get_logger().info(f"Tag size: {self.tag_size}m")
        self.get_logger().info(f"Target detections: {self.num_detections}")
        self.get_logger().info(f"Output file: {self.output_file}")
        self.get_logger().info(f"Camera topic: {CAMERA_TOPIC}")
        self.get_logger().info(f"Camera info topic: {CAMERA_INFO_TOPIC}")

    def get_shutdown_future(self) -> rclpy.Future:
        """Get the future that signals shutdown."""
        return self.shutdown_future

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Process camera info messages and extract intrinsic parameters."""
        if self.camera_info_received:
            return

        try:
            # Extract camera matrix (K)
            if len(msg.k) == 9:
                self.camera_matrix = np.array(msg.k).reshape(3, 3)

                # Extract distortion coefficients
                self.dist_coeffs = np.array(msg.d) if msg.d else np.zeros(5)

                self.camera_info_received = True
                self.get_logger().info("✅ Camera parameters received")
                self.get_logger().info(f"Camera matrix:\n{self.camera_matrix}")
                self.get_logger().info(f"Distortion coefficients: {self.dist_coeffs}")

            else:
                self.get_logger().warning(
                    f"Invalid camera matrix size: expected 9, got {len(msg.k)}"
                )
        except Exception as e:
            self.get_logger().error(f"Error processing camera info: {e}")

    def image_callback(self, msg: Image) -> None:
        """Process camera images and detect AprilTags."""
        # Check if we have enough detections
        if self.detection_count >= self.num_detections:
            return

        # Check timeout
        if time.time() - self.start_time > DETECTION_TIMEOUT:
            self.get_logger().warning(
                f"Detection timeout reached ({DETECTION_TIMEOUT}s). "
                f"Only {self.detection_count} detections collected."
            )
            self.finalize_calibration()
            return

        # Wait for camera parameters
        if not self.camera_info_received:
            self.get_logger().info("Waiting for camera parameters...")
            return

        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Store latest image for validation
            self.latest_image = cv_image.copy()

            # Convert to grayscale for AprilTag detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            results = self.detector.detect(gray)

            # Process detections
            target_detection = None
            for detection in results:
                if detection.tag_id == TAG_ID:
                    target_detection = detection
                    break

            if target_detection is not None:
                self.process_detection(target_detection, cv_image)

                # Draw detection for visualization
                self.draw_detection(cv_image, target_detection)

                # Show image with detection
                cv2.imshow("AprilTag Detection", cv_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def process_detection(self, detection: Any, image: np.ndarray) -> None:
        """Process a single AprilTag detection and estimate pose."""
        try:
            # Get tag corners in image coordinates
            corners = detection.corners.reshape(-1, 2)

            # Define 3D object points for the tag in counter-clockwise order
            # starting from top-left, to match the AprilTag detector's output
            half_size = self.tag_size / 2.0
            object_points = np.array(
                [
                    [-half_size, -half_size, 0],  # Top-left
                    [half_size, -half_size, 0],  # Top-right
                    [half_size, half_size, 0],  # Bottom-right
                    [-half_size, half_size, 0],  # Bottom-left
                ],
                dtype=np.float32,
            )

            # Reorder AprilTag corners to match object_points definition
            # AprilTag corners: top-left, top-right, bottom-right, bottom-left (counter-clockwise)
            # This is already consistent with our object_points, so no reordering is needed

            # Estimate pose using PnP
            success, rvec, tvec = cv2.solvePnP(
                object_points, corners, self.camera_matrix, self.dist_coeffs
            )

            if success:
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)

                # Store detection data
                detection_data = {
                    "translation": tvec.flatten().tolist(),
                    "rotation_matrix": rmat.tolist(),
                    "rotation_vector": rvec.flatten().tolist(),
                    "corners": corners.tolist(),
                    "tag_id": detection.tag_id,
                    "timestamp": time.time(),
                }

                self.detections.append(detection_data)
                self.detection_count += 1

                # Log progress
                progress = (self.detection_count / self.num_detections) * 100
                self.get_logger().info(
                    f"Detection {self.detection_count}/{self.num_detections} "
                    f"({progress:.1f}%) - "
                    f"Pose: t={tvec.flatten()}"
                )

                # Check if we have enough detections
                if self.detection_count >= self.num_detections:
                    self.finalize_calibration()

        except Exception as e:
            self.get_logger().error(f"Error processing detection: {e}")

    def draw_detection(self, image: np.ndarray, detection: Any) -> None:
        """Draw AprilTag detection on image."""
        try:
            # Draw tag outline
            corners = detection.corners.reshape(-1, 2).astype(int)
            cv2.polylines(image, [corners], True, (0, 255, 0), 2)

            # Draw tag ID
            center = tuple(map(int, detection.center))
            cv2.putText(
                image,
                f"ID: {detection.tag_id}",
                (center[0] - 30, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Draw center point
            cv2.circle(image, center, 5, (0, 0, 255), -1)

        except Exception as e:
            self.get_logger().error(f"Error drawing detection: {e}")

    def finalize_calibration(self) -> None:
        """Calculate average pose and save calibration data."""
        if self.shutdown_future.done():
            return  # Avoid multiple calls

        if not self.detections:
            self.get_logger().error("No detections collected!")
            self.shutdown_future.set_result(True)  # Signal shutdown
            return

        self.get_logger().info(
            f"Finalizing calibration with {len(self.detections)} detections..."
        )

        # Calculate average translation
        translations = np.array([d["translation"] for d in self.detections])
        avg_translation = np.mean(translations, axis=0)

        # Calculate average rotation using rotation matrices
        rotation_matrices = np.array([d["rotation_matrix"] for d in self.detections])
        avg_rotation_matrix = self.average_rotation_matrices(rotation_matrices)

        # Convert to different representations
        avg_rotation = Rotation.from_matrix(avg_rotation_matrix)
        avg_quaternion = avg_rotation.as_quat()  # [x, y, z, w]
        avg_euler = avg_rotation.as_euler("xyz", degrees=True)

        # Calculate camera-world transform
        # Since the AprilTag is at world origin, the camera pose in world coordinates
        # is simply the inverse of the tag pose in camera coordinates
        tag_to_camera = np.eye(4)
        tag_to_camera[:3, :3] = avg_rotation_matrix
        tag_to_camera[:3, 3] = avg_translation

        # Camera to world transform (inverse of tag to camera)
        camera_to_world = np.linalg.inv(tag_to_camera)

        # Apply coordinate system correction
        # The detected pose might need rotation to align with world coordinates
        # X: near-to-far, Y: right-to-left, Z: low-to-high
        world_correction = self.get_world_coordinate_correction()
        camera_to_world = world_correction @ camera_to_world

        # Prepare calibration data
        calibration_data = {
            "metadata": {
                "calibration_date": time.ctime(),
                "tag_family": TAG_FAMILY,
                "tag_id": TAG_ID,
                "tag_size_m": self.tag_size,
                "num_detections": len(self.detections),
                "camera_topic": CAMERA_TOPIC,
                "coordinate_system": "X: near-to-far, Y: right-to-left, Z: low-to-high (Corrected)",
            },
            "camera_parameters": {
                "camera_matrix": self.camera_matrix.tolist(),
                "distortion_coefficients": self.dist_coeffs.tolist(),
            },
            "tag_pose_in_camera": {
                "translation": avg_translation.tolist(),
                "rotation_matrix": avg_rotation_matrix.tolist(),
                "quaternion": avg_quaternion.tolist(),  # [x, y, z, w]
                "euler_angles_deg": avg_euler.tolist(),  # [x, y, z] in degrees
            },
            "camera_to_world_transform": {
                "matrix": camera_to_world.tolist(),
                "translation": camera_to_world[:3, 3].tolist(),
                "rotation_matrix": camera_to_world[:3, :3].tolist(),
            },
            "statistics": {
                "translation_std": np.std(translations, axis=0).tolist(),
                "detection_timestamps": [d["timestamp"] for d in self.detections],
            },
        }

        # Save calibration data
        try:
            with open(self.output_file, "w") as f:
                json.dump(calibration_data, f, indent=2)

            self.get_logger().info(f"✅ Calibration saved to: {self.output_file}")

            # Log summary
            self.log_calibration_summary(calibration_data)

            # Validate calibration with visualization
            self.validate_calibration()

        except Exception as e:
            self.get_logger().error(f"Error saving calibration: {e}")

        # Close OpenCV windows
        cv2.destroyAllWindows()

        # Shutdown node
        self.get_logger().info("Calibration complete. Shutting down...")
        self.shutdown_future.set_result(True)  # Signal main loop to shut down

    def average_rotation_matrices(self, rotation_matrices: np.ndarray) -> np.ndarray:
        """Calculate average rotation matrix using SVD method."""
        # Convert to quaternions for averaging
        rotations = [Rotation.from_matrix(R) for R in rotation_matrices]

        # Average quaternions (simple method)
        quaternions = np.array([r.as_quat() for r in rotations])

        # Ensure all quaternions are in the same hemisphere
        for i in range(1, len(quaternions)):
            if np.dot(quaternions[0], quaternions[i]) < 0:
                quaternions[i] *= -1

        # Average and normalize
        avg_quat = np.mean(quaternions, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        # Convert back to rotation matrix
        return Rotation.from_quat(avg_quat).as_matrix()

    def get_world_coordinate_correction(self) -> np.ndarray:
        """
        Get transformation matrix to correct the natural AprilTag coordinate system
        to the desired world coordinate system.

        Desired World System (to match robot coordinates):
        - X-axis: near-to-far (forward)
        - Y-axis: right-to-left (left)
        - Z-axis: low-to-high (up)

        This correction flips the natural AprilTag axes:
        Natural Tag Frame (camera view):
            X: left -> right
            Y: far -> near (towards camera)
            Z: high -> low (down)

        Mapping to Robot Frame:
            Robot X = - Tag Y  (near -> far)
            Robot Y = - Tag X  (right -> left)
            Robot Z = - Tag Z  (low -> high)

        The resulting rotation matrix has determinant +1 (right-handed).

        Returns:
            4x4 transformation matrix for coordinate correction.
        """
        correction = np.eye(4)
        # Rotation matrix implementing the mapping described above.
        correction[:3, :3] = np.array(
            [
                [0, -1, 0],  # Robot X  <- -Tag Y
                [-1, 0, 0],  # Robot Y  <- -Tag X
                [0, 0, -1],
            ],  # Robot Z  <- -Tag Z
            dtype=np.float32,
        )
        return correction

    def log_calibration_summary(self, calibration_data: dict) -> None:
        """Log a summary of the calibration results."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("CALIBRATION SUMMARY")
        self.get_logger().info("=" * 60)

        metadata = calibration_data["metadata"]
        tag_pose = calibration_data["tag_pose_in_camera"]
        camera_transform = calibration_data["camera_to_world_transform"]
        stats = calibration_data["statistics"]

        self.get_logger().info(f"Date: {metadata['calibration_date']}")
        self.get_logger().info(f"Tag: {metadata['tag_family']} ID {metadata['tag_id']}")
        self.get_logger().info(f"Tag size: {metadata['tag_size_m']}m")
        self.get_logger().info(f"Detections: {metadata['num_detections']}")
        self.get_logger().info("")

        self.get_logger().info("Tag pose in camera coordinates:")
        t = tag_pose["translation"]
        self.get_logger().info(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

        euler = tag_pose["euler_angles_deg"]
        self.get_logger().info(
            f"  Rotation (XYZ): [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]"
        )

        self.get_logger().info("")
        self.get_logger().info("Camera position in world coordinates:")
        world_t = camera_transform["translation"]
        self.get_logger().info(
            f"  Translation: [{world_t[0]:.4f}, {world_t[1]:.4f}, {world_t[2]:.4f}]"
        )

        self.get_logger().info("")
        self.get_logger().info("Detection stability:")
        std = stats["translation_std"]
        self.get_logger().info(
            f"  Translation std: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]"
        )

        self.get_logger().info("=" * 60)

    def validate_calibration(self) -> None:
        """Validate calibration by visualizing a coordinate cube on the image."""
        if self.latest_image is None:
            self.get_logger().warning("No image available for validation")
            return

        try:
            # Load the saved calibration data
            with open(self.output_file, "r") as f:
                calibration_data = json.load(f)

            # Extract camera parameters
            camera_matrix = np.array(
                calibration_data["camera_parameters"]["camera_matrix"]
            )
            dist_coeffs = np.array(
                calibration_data["camera_parameters"]["distortion_coefficients"]
            )

            # Extract camera-to-world transform
            camera_to_world = np.array(
                calibration_data["camera_to_world_transform"]["matrix"]
            )

            # Get world-to-camera transform (inverse)
            world_to_camera = np.linalg.inv(camera_to_world)

            # Create validation image
            validation_image = self.latest_image.copy()

            # Define world coordinate axes points for visualization (5cm long)
            axis_length = 0.05
            world_points = np.array(
                [
                    [0.0, 0.0, 0.0],  # Origin
                    [axis_length, 0.0, 0.0],  # X-axis end
                    [0.0, axis_length, 0.0],  # Y-axis end
                    [0.0, 0.0, axis_length],  # Z-axis end
                ],
                dtype=np.float32,
            )

            # Transform world points to camera coordinates
            camera_points = self.transform_world_to_camera(
                world_points, world_to_camera
            )

            # Project 3D points to image plane
            projected_points, _ = cv2.projectPoints(
                camera_points,
                np.zeros(3),  # No additional rotation
                np.zeros(3),  # No additional translation
                camera_matrix,
                dist_coeffs,
            )

            # Draw coordinate axes on the image
            self.draw_coordinate_axes(validation_image, projected_points)

            # Save validation image
            output_path = self.output_file.with_suffix(".jpg")
            cv2.imwrite(str(output_path), validation_image)

            self.get_logger().info(
                f"✅ Validation visualization saved to: {output_path}"
            )

            # Display validation metrics
            self.display_validation_metrics(calibration_data, camera_points)

        except Exception as e:
            self.get_logger().error(f"Error during calibration validation: {e}")
            traceback.print_exc()

    def transform_world_to_camera(
        self, world_points: np.ndarray, world_to_camera: np.ndarray
    ) -> np.ndarray:
        """Transform world points to camera coordinates."""
        # Convert to homogeneous coordinates
        ones = np.ones((world_points.shape[0], 1))
        world_points_homo = np.hstack([world_points, ones])

        # Transform to camera coordinates
        camera_points_homo = (world_to_camera @ world_points_homo.T).T

        # Convert back to 3D coordinates
        return camera_points_homo[:, :3]

    def draw_coordinate_axes(
        self, image: np.ndarray, projected_points: np.ndarray
    ) -> None:
        """Draw clear, labeled coordinate axes on the image."""
        # Extract projected points
        origin = tuple(projected_points[0][0].astype(int))
        x_end = tuple(projected_points[1][0].astype(int))
        y_end = tuple(projected_points[2][0].astype(int))
        z_end = tuple(projected_points[3][0].astype(int))

        # Define drawing properties for high visibility
        arrow_thickness = 3
        tip_length = 0.25
        font_scale = 0.8
        font_thickness = 2
        text_offset = 15

        # Draw X-axis (Red)
        cv2.arrowedLine(
            image, origin, x_end, (0, 0, 255), arrow_thickness, tipLength=tip_length
        )
        cv2.putText(
            image,
            "X",
            (x_end[0] + text_offset, x_end[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            font_thickness,
        )

        # Draw Y-axis (Green)
        cv2.arrowedLine(
            image, origin, y_end, (0, 255, 0), arrow_thickness, tipLength=tip_length
        )
        cv2.putText(
            image,
            "Y",
            (y_end[0] + text_offset, y_end[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            font_thickness,
        )

        # Draw Z-axis (Blue)
        cv2.arrowedLine(
            image, origin, z_end, (255, 0, 0), arrow_thickness, tipLength=tip_length
        )
        cv2.putText(
            image,
            "Z",
            (z_end[0] + text_offset, z_end[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 0),
            font_thickness,
        )

        # Draw a prominent origin point
        cv2.circle(image, origin, 5, (255, 255, 255), -1)
        cv2.circle(image, origin, 5, (0, 0, 0), 2)
        cv2.putText(
            image,
            "Origin",
            (origin[0] + text_offset, origin[1] - text_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def display_validation_metrics(
        self, calibration_data: dict, camera_points: np.ndarray
    ) -> None:
        """Display validation metrics and coordinate information."""
        self.get_logger().info("=" * 60)
        self.get_logger().info("CALIBRATION VALIDATION")
        self.get_logger().info("=" * 60)

        # World origin in camera coordinates
        origin_in_camera = camera_points[0]
        self.get_logger().info(
            f"World origin in camera coordinates: [{origin_in_camera[0]:.4f}, {origin_in_camera[1]:.4f}, {origin_in_camera[2]:.4f}]"
        )

        # Coordinate axes endpoints in camera coordinates
        x_axis_end = camera_points[1]
        y_axis_end = camera_points[2]
        z_axis_end = camera_points[3]

        self.get_logger().info(
            f"X-axis end in camera coordinates: [{x_axis_end[0]:.4f}, {x_axis_end[1]:.4f}, {x_axis_end[2]:.4f}]"
        )
        self.get_logger().info(
            f"Y-axis end in camera coordinates: [{y_axis_end[0]:.4f}, {y_axis_end[1]:.4f}, {y_axis_end[2]:.4f}]"
        )
        self.get_logger().info(
            f"Z-axis end in camera coordinates: [{z_axis_end[0]:.4f}, {z_axis_end[1]:.4f}, {z_axis_end[2]:.4f}]"
        )

        # Verify coordinate directions
        self.get_logger().info("")
        self.get_logger().info("Natural Coordinate Direction Vectors:")

        # Check X-axis direction
        x_diff = x_axis_end - origin_in_camera
        self.get_logger().info(
            f"X+ direction vector: [{x_diff[0]:.4f}, {x_diff[1]:.4f}, {x_diff[2]:.4f}]"
        )

        # Check Y-axis direction
        y_diff = y_axis_end - origin_in_camera
        self.get_logger().info(
            f"Y+ direction vector: [{y_diff[0]:.4f}, {y_diff[1]:.4f}, {y_diff[2]:.4f}]"
        )

        # Check Z-axis direction
        z_diff = z_axis_end - origin_in_camera
        self.get_logger().info(
            f"Z+ direction vector: [{z_diff[0]:.4f}, {z_diff[1]:.4f}, {z_diff[2]:.4f}]"
        )

        # Compare with expected AprilTag pose
        tag_pose = calibration_data["tag_pose_in_camera"]
        tag_translation = np.array(tag_pose["translation"])

        self.get_logger().info("")
        self.get_logger().info("Consistency Check:")
        origin_error = np.linalg.norm(origin_in_camera - tag_translation)
        self.get_logger().info(
            f"World origin vs AprilTag center error: {origin_error:.6f}m"
        )

        if origin_error < 0.01:  # 1cm tolerance
            self.get_logger().info("✅ Calibration appears consistent!")
        else:
            self.get_logger().warning(
                "⚠️  Large error detected - check calibration setup"
            )

        self.get_logger().info("")
        self.get_logger().info("Instructions:")
        self.get_logger().info(
            "- Look at the visualization to understand the natural coordinate system"
        )
        self.get_logger().info("- Red arrow shows X+ direction")
        self.get_logger().info("- Green arrow shows Y+ direction")
        self.get_logger().info("- Blue arrow shows Z+ direction")
        self.get_logger().info(
            "- This is the raw coordinate system from AprilTag detection"
        )

        self.get_logger().info("=" * 60)


def main() -> int:  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Camera calibration using AprilTag detection"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output calibration file (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--tag-size",
        type=float,
        default=TAG_SIZE,
        help=f"AprilTag size in meters (default: {TAG_SIZE})",
    )
    parser.add_argument(
        "--num-detections",
        type=int,
        default=NUM_DETECTIONS,
        help=f"Number of detections to average (default: {NUM_DETECTIONS})",
    )

    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()

    # Create calibration node
    calibration_node = AprilTagCalibration(
        output_file=args.output,
        tag_size=args.tag_size,
        num_detections=args.num_detections,
    )

    # Get the shutdown future to allow graceful exit
    shutdown_future = calibration_node.get_shutdown_future()

    try:
        # Run calibration until shutdown is signaled
        rclpy.spin_until_future_complete(calibration_node, shutdown_future)
    except KeyboardInterrupt:
        calibration_node.get_logger().info("Calibration interrupted by user")
    finally:
        calibration_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
