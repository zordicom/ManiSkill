#!/usr/bin/env python3
"""
Copyright 2025 Zordi, Inc. All rights reserved.

Multi-Object FoundationPose WebSocket Server

This server provides pose estimation services for multiple objects simultaneously via WebSocket.
It extends the original single-object server to support:
- Multiple object initialization in one session
- Simultaneous pose tracking for all initialized objects
- Object-specific mask handling
- Combined visualization output
- Performance monitoring across all objects

Usage:
    python multi_object_server.py --host 0.0.0.0 --port 10014
"""

import argparse
import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List

import cv2
import imageio
import numpy as np
import nvdiffrast.torch as dr
import trimesh

# FoundationPose imports
from estimator import FoundationPose, PoseRefinePredictor, ScorePredictor
from scipy.spatial.transform import Rotation
from Utils import draw_posed_3d_box, draw_xyz_axis, set_seed
from zordi_policy_rpc.direct.metadata import Metadata
from zordi_policy_rpc.direct.server import DirectServer
from zordi_policy_rpc.direct.server.interface import Server

logger = logging.getLogger(__name__)


class MultiObjectFoundationPoseServer(Server):
    """WebSocket server for multi-object FoundationPose estimation."""

    def __init__(self, available_objects: Dict[str, str]):
        """Initialize the multi-object FoundationPose server.

        Args:
            available_objects: Dict mapping object names to their mesh file paths
        """
        set_seed(0)

        self.available_objects = available_objects
        self.estimators: Dict[str, FoundationPose] = {}  # object_name -> estimator
        self.mesh_data: Dict[
            str, Dict
        ] = {}  # object_name -> {mesh, bbox, to_origin_inv}
        self.debug_dir = Path("./recv_imgs_multi")
        self.debug_dir.mkdir(parents=True, exist_ok=True)

        # Load default camera intrinsics
        self.default_K = self._load_default_camera_intrinsics()

        # Shared FoundationPose components (expensive to initialize)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

        # Session state
        self.session_initialized = False
        self.frame_count = 0
        self.active_objects: List[str] = []  # List of currently tracked objects

        # Performance monitoring
        self.request_count = 0
        self.initialization_times = []
        self.prediction_times = []
        self.communication_times = []
        self.log_interval = 50

        # Performance optimizations
        self.debug_level = 1  # 0=none, 1=minimal, 2=full debug
        self.save_visualizations = False  # Set to True only if needed
        self.save_individual_masks = False  # Set to True only if needed

        # Configurable iteration parameters (optimized from run_demo.py)
        self.register_iterations = 5  # More iterations for initial registration
        self.track_iterations = 2  # Fewer iterations for tracking (faster)

        # Mesh optimization parameters (to reduce VRAM usage)
        self.max_vertices = 50000  # Maximum vertices per mesh (reduce for lower VRAM)
        self.decimation_factor = 0.5  # Decimation factor when mesh is too large
        self.enable_mesh_decimation = True  # Enable automatic mesh decimation

        # Pre-load and cache all mesh data at startup (major optimization)
        self._preload_mesh_data()

        # Log GPU memory usage after initialization
        self._log_gpu_memory_usage("server_initialization")

        logger.info(
            f"Multi-object FoundationPose server initialized with objects: "
            f"{list(available_objects.keys())}"
        )
        logger.info(f"Default camera intrinsics loaded: {self.default_K.shape}")
        logger.info(
            f"Debug level: {self.debug_level}, Visualizations: {self.save_visualizations}"
        )
        logger.info(f"Mesh data pre-loaded for {len(self.mesh_data)} objects")
        logger.info(
            f"Mesh decimation: {'enabled' if self.enable_mesh_decimation else 'disabled'} "
            f"(max_vertices: {self.max_vertices}, decimation_factor: {self.decimation_factor})"
        )

    def _preload_mesh_data(self) -> None:
        """Pre-load and cache mesh data for all available objects (performance optimization)."""
        logger.info("Pre-loading mesh data for faster initialization...")

        for object_name, mesh_file in self.available_objects.items():
            try:
                # Load mesh - force='mesh' handles multi-material OBJ files that load as Scene
                mesh = trimesh.load(mesh_file, force="mesh")

                # Ensure we have a proper mesh object with vertices
                if not hasattr(mesh, "vertices") or not hasattr(mesh, "vertex_normals"):
                    # Fallback: try loading as Scene and converting to mesh
                    logger.warning(
                        f"Mesh for {object_name} loaded without vertices/normals, trying Scene.to_mesh()"
                    )
                    try:
                        scene = trimesh.load(mesh_file)
                        mesh = scene.to_mesh()  # type: ignore
                    except Exception as fallback_error:
                        raise ValueError(
                            f"Could not convert loaded object to mesh for {object_name}: {fallback_error}"
                        )

                # Final check after fallback
                if not hasattr(mesh, "vertices") or not hasattr(mesh, "vertex_normals"):
                    raise ValueError(
                        f"Loaded mesh for {object_name} does not have required vertices/normals"
                    )

                # Mesh decimation to reduce VRAM usage
                vertices = getattr(mesh, "vertices", None)
                if (
                    self.enable_mesh_decimation
                    and vertices is not None
                    and len(vertices) > self.max_vertices
                ):
                    original_count = len(vertices)
                    logger.info(
                        f"Mesh {object_name} has {original_count} vertices, decimating to reduce VRAM usage..."
                    )

                    try:
                        # Calculate target vertex count
                        faces = getattr(mesh, "faces", None)
                        simplify_method = getattr(
                            mesh, "simplify_quadric_decimation", None
                        )

                        if faces is not None and simplify_method is not None:
                            target_faces = int(len(faces) * self.decimation_factor)
                            target_faces = max(target_faces, 1000)  # Minimum 1000 faces

                            # Use trimesh's quadric decimation
                            mesh = simplify_method(target_faces)

                            # Ensure we still have vertex normals after decimation
                            vertex_normals = getattr(mesh, "vertex_normals", None)
                            if vertex_normals is None and hasattr(
                                mesh, "compute_vertex_normals"
                            ):
                                mesh.compute_vertex_normals()

                            new_vertices = getattr(mesh, "vertices", None)
                            if new_vertices is not None:
                                logger.info(
                                    f"Decimated {object_name}: {original_count} -> {len(new_vertices)} vertices "
                                    f"({len(new_vertices) / original_count * 100:.1f}% reduction)"
                                )
                        else:
                            logger.warning(
                                f"Mesh {object_name} doesn't support decimation, using original mesh"
                            )

                    except Exception as decimation_error:
                        logger.warning(
                            f"Failed to decimate mesh {object_name}: {decimation_error}. "
                            f"Using original mesh with {original_count} vertices."
                        )

                # Calculate bounding box and transformation (from run_demo.py pattern)
                to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
                mesh_bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
                to_origin_inv = np.linalg.inv(to_origin)

                # Cache mesh data - copy arrays to avoid reference issues
                # Use getattr to access mesh attributes for linter compatibility
                mesh_vertices = getattr(mesh, "vertices")
                mesh_normals = getattr(mesh, "vertex_normals")

                self.mesh_data[object_name] = {
                    "mesh": mesh,
                    "bbox": mesh_bbox,
                    "to_origin_inv": to_origin_inv,
                    "vertices": np.array(
                        mesh_vertices, dtype=np.float32
                    ),  # Cache vertices for faster access
                    "vertex_normals": np.array(
                        mesh_normals, dtype=np.float32
                    ),  # Cache normals
                }

                logger.debug(
                    f"Pre-loaded mesh data for {object_name}: {mesh_vertices.shape[0]} vertices"
                )

            except Exception as e:
                logger.error(f"Failed to pre-load mesh for {object_name}: {e}")
                # Remove from available objects if mesh loading fails
                if object_name in self.available_objects:
                    del self.available_objects[object_name]

    def _matrix_to_pose_7d(self, pose_matrix: np.ndarray) -> list:
        """Convert 4x4 pose matrix to [x, y, z, qx, qy, qz, qw] format (optimized)."""
        # Extract translation (faster indexing)
        translation = pose_matrix[:3, 3]

        # Extract rotation matrix and convert to quaternion (optimized)
        rotation_matrix = pose_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # Returns [x, y, z, w] format

        # Combine using np.concatenate for better performance
        pose_7d = np.concatenate([translation, quaternion])
        return pose_7d.tolist()

    def _log_performance_stats(self, reason: str = "periodic") -> None:
        """Log average initialization, prediction, and communication times."""
        if not self.communication_times:
            return

        avg_communication = (
            sum(self.communication_times) / len(self.communication_times) * 1000
        )

        logger.info("--------------------------------")
        logger.info(
            f"Multi-Object Performance Stats ({reason}) - {len(self.communication_times)} requests:"
        )
        logger.info(f"  Average communication time: {avg_communication:.2f}ms")

        if self.initialization_times:
            avg_initialization = (
                sum(self.initialization_times) / len(self.initialization_times) * 1000
            )
            logger.info(
                f"  Average initialization time: {avg_initialization:.2f}ms "
                f"({len(self.initialization_times)} initializations)"
            )

        if self.prediction_times:
            avg_prediction = (
                sum(self.prediction_times) / len(self.prediction_times) * 1000
            )
            logger.info(
                f"  Average prediction time: {avg_prediction:.2f}ms "
                f"({len(self.prediction_times)} predictions)"
            )

        # Reset counters
        self.initialization_times.clear()
        self.prediction_times.clear()
        self.communication_times.clear()
        self.request_count = 0

    @staticmethod
    def _load_default_camera_intrinsics() -> np.ndarray:
        """Load default camera intrinsics from config file."""
        script_dir = Path(__file__).parent
        K_file = script_dir / "config" / "cam_K.txt"

        if not K_file.exists():
            logger.warning(f"Default camera intrinsics file not found: {K_file}")
            return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

        try:
            K = np.loadtxt(str(K_file)).reshape(3, 3)
            logger.info(f"Loaded default camera intrinsics from {K_file}")
            return K
        except Exception as e:
            logger.error(f"Failed to load camera intrinsics: {e}")
            return np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    def get_metadata(self) -> Metadata:
        """Return server metadata for multi-object tracking."""
        return {
            "server_info": {
                "server_name": "multi_object_foundationpose_server",
                "service_type": "multi-object pose estimation",
            },
            "service_metadata": {
                "available_objects": list(self.available_objects.keys()),
                "supports_multi_object_tracking": True,
                "max_simultaneous_objects": len(self.available_objects),
                "debug_mode": True,
                "default_camera_intrinsics": self.default_K.flatten().tolist(),
            },
            "request_format": {
                "initialize_multi": {
                    "objects": [
                        {
                            "object_name": str,
                            "rgb": bytes,
                            "depth": bytes,
                            "mask": bytes,
                        }
                    ]
                },
                "predict_multi": {
                    "rgb": bytes,
                    "depth": bytes,
                },
                "reset_session": {},
            },
            "response_format": {
                "initialize_multi": {
                    "success": bool,
                    "message": str,
                    "poses": {
                        "object_name": list
                    },  # 7-element [x, y, z, qx, qy, qz, qw]
                    "initialized_objects": list,
                },
                "predict_multi": {
                    "poses": {
                        "object_name": list
                    },  # 7-element [x, y, z, qx, qy, qz, qw]
                    "confidences": {"object_name": float},
                    "frame_id": int,
                },
                "reset_session": {
                    "success": bool,
                    "message": str,
                },
            },
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming requests with performance monitoring."""
        request_type = request.get("type")
        start_time = time.time()

        try:
            if request_type == "initialize_multi":
                response = self._handle_initialize_multi(request)
            elif request_type == "predict_multi":
                response = self._handle_predict_multi(request)
            elif request_type == "reset_session":
                response = self._handle_reset_session(request)
            else:
                response = {"error": f"Unknown request type: {request_type}"}

            # Track communication time
            if request_type in {"initialize_multi", "predict_multi"}:
                communication_time = time.time() - start_time
                self.communication_times.append(communication_time)
                self.request_count += 1

                if self.request_count >= self.log_interval:
                    self._log_performance_stats("periodic")

            return response

        except Exception as e:
            logger.error(f"Error handling request {request_type}: {e}")
            return {"error": f"Request processing failed: {e!s}"}

    def _handle_initialize_multi(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize pose estimation for multiple objects."""
        objects_data = request.get("objects", [])
        if not objects_data:
            return {
                "success": False,
                "message": "No objects provided for initialization",
                "poses": {},
                "initialized_objects": [],
            }

        poses = {}
        initialized_objects = []
        failed_objects = []

        inference_start = time.time()

        for obj_data in objects_data:
            object_name = obj_data.get("object_name")
            if not object_name or object_name not in self.available_objects:
                failed_objects.append(f"Invalid object: {object_name}")
                continue

            try:
                # Decode images for this object
                rgb_image = self._decode_image(obj_data["rgb"], cv2.IMREAD_COLOR)
                depth_image = self._decode_image(obj_data["depth"], cv2.IMREAD_ANYDEPTH)
                mask_image = self._decode_image(obj_data["mask"], cv2.IMREAD_GRAYSCALE)

                # Convert depth from mm to meters
                depth_image = depth_image.astype(np.float32) / 1000.0

                # Camera intrinsics
                camera_intrinsics = obj_data.get("camera_intrinsics", [])
                if camera_intrinsics and len(camera_intrinsics) == 9:
                    K = np.array(camera_intrinsics).reshape(3, 3)
                else:
                    K = self.default_K.copy()

                # Save debug images only if debug level is high enough
                if self.debug_level >= 2:
                    self._save_debug_image(rgb_image, f"init_{object_name}_rgb.png")
                    self._save_debug_image(
                        (depth_image * 1000).astype(np.uint16),
                        f"init_{object_name}_depth.png",
                    )
                    self._save_debug_image(mask_image, f"init_{object_name}_mask.png")

                # Load mesh
                mesh_data = self.mesh_data[object_name]

                # Create FoundationPose estimator (shared components)
                try:
                    estimator = FoundationPose(
                        model_pts=mesh_data["vertices"],
                        model_normals=mesh_data["vertex_normals"],
                        mesh=mesh_data["mesh"],
                        scorer=self.scorer,
                        refiner=self.refiner,
                        debug_dir=str(self.debug_dir),
                        debug=0,
                        glctx=self.glctx,
                    )
                except Exception as cuda_error:
                    # Log GPU memory usage on CUDA errors
                    self._log_gpu_memory_usage(f"cuda_error_{object_name}")

                    # Check if it's a CUDA memory error
                    if "cudaMalloc" in str(cuda_error) or "CUDA" in str(cuda_error):
                        logger.error(
                            f"CUDA memory error initializing {object_name}: {cuda_error}. "
                            f"Try reducing --max-vertices (currently {self.max_vertices}) "
                            f"or increasing --decimation-factor (currently {self.decimation_factor})"
                        )
                    raise cuda_error

                # Run initial pose estimation
                mask_bool = mask_image.astype(bool)
                pose = estimator.register(
                    K=K,
                    rgb=rgb_image,
                    depth=depth_image,
                    ob_mask=mask_bool,
                    iteration=self.register_iterations,
                )

                # Store estimator and pose
                self.estimators[object_name] = estimator
                poses[object_name] = self._matrix_to_pose_7d(pose)
                initialized_objects.append(object_name)

                # Log GPU memory usage after each object initialization
                if self.debug_level >= 2:
                    self._log_gpu_memory_usage(f"after_init_{object_name}")

                # Create visualization only if enabled (expensive operation)
                if self.save_visualizations:
                    center_pose = pose @ mesh_data["to_origin_inv"]
                    vis = draw_posed_3d_box(
                        K, img=rgb_image, ob_in_cam=center_pose, bbox=mesh_data["bbox"]
                    )
                    vis = draw_xyz_axis(
                        vis,
                        ob_in_cam=center_pose,
                        scale=0.1,
                        K=K,
                        thickness=3,
                        transparency=0,
                        is_input_rgb=True,
                    )
                    self._save_debug_image(vis, f"init_{object_name}_vis.png")

                logger.info(f"Successfully initialized {object_name}")

            except Exception as e:
                logger.error(f"Failed to initialize {object_name}: {e}")
                failed_objects.append(f"{object_name}: {e}")

        initialization_time = time.time() - inference_start
        self.initialization_times.append(initialization_time)

        self.active_objects = initialized_objects
        self.session_initialized = len(initialized_objects) > 0
        self.frame_count = 0

        success_msg = f"Initialized {len(initialized_objects)} objects"
        if failed_objects:
            success_msg += f", {len(failed_objects)} failed: {failed_objects}"

        return {
            "success": len(initialized_objects) > 0,
            "message": success_msg,
            "poses": poses,
            "initialized_objects": initialized_objects,
        }

    def _handle_predict_multi(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Predict poses for all active objects (highly optimized)."""
        if not self.session_initialized or not self.active_objects:
            return {
                "poses": {},
                "confidences": {},
                "frame_id": -1,
                "error": "No objects initialized. Call 'initialize_multi' first.",
            }

        try:
            # Decode shared images ONCE (major optimization)
            rgb_image = self._decode_image(request["rgb"], cv2.IMREAD_COLOR)
            depth_image = self._decode_image(request["depth"], cv2.IMREAD_ANYDEPTH)

            # Optimized depth conversion (avoid unnecessary copying)
            if depth_image.dtype != np.float32:
                depth_image = depth_image.astype(np.float32, copy=False)
            depth_image *= 0.001  # Convert mm to meters in-place

            # Camera intrinsics (optimized array creation)
            camera_intrinsics = request.get("camera_intrinsics", [])
            if camera_intrinsics and len(camera_intrinsics) == 9:
                K = np.array(camera_intrinsics, dtype=np.float64).reshape(3, 3)
            else:
                K = self.default_K  # Reuse without copying if possible

            # Optional per-object masks (decode only if debug level requires it)
            decoded_masks = {}
            if self.save_individual_masks and self.debug_level >= 2:
                masks = request.get("masks", {})
                for obj_name, mask_bytes in masks.items():
                    if (
                        obj_name in self.active_objects
                    ):  # Only decode masks for active objects
                        decoded_masks[obj_name] = self._decode_image(
                            mask_bytes, cv2.IMREAD_GRAYSCALE
                        )

            # Pre-allocate result dictionaries for better performance
            poses = {}
            confidences = {}
            failed_objects = []

            inference_start = time.time()

            # Optimized tracking loop with early error handling
            for object_name in self.active_objects:
                estimator = self.estimators.get(object_name)
                if estimator is None:
                    failed_objects.append(object_name)
                    continue

                try:
                    mesh_data = self.mesh_data[object_name]

                    # Save individual mask debug image only if enabled and available
                    if (
                        object_name in decoded_masks
                        and self.debug_level >= 2
                        and self.save_individual_masks
                    ):
                        self._save_debug_image(
                            decoded_masks[object_name], f"track_{object_name}_mask.png"
                        )

                    # Core inference (track pose) - optimized parameters
                    pose = estimator.track_one(
                        rgb=rgb_image,
                        depth=depth_image,
                        K=K,
                        iteration=self.track_iterations,
                    )

                    # Fast pose conversion and storage
                    poses[object_name] = self._matrix_to_pose_7d(pose)
                    confidences[object_name] = 0.8  # Placeholder confidence

                    # Create visualization only if explicitly enabled (expensive operation)
                    if self.save_visualizations and self.debug_level >= 1:
                        center_pose = pose @ mesh_data["to_origin_inv"]

                        # Optimized visualization creation (reuse image array)
                        vis = draw_posed_3d_box(
                            K,
                            img=rgb_image,
                            ob_in_cam=center_pose,
                            bbox=mesh_data["bbox"],
                        )
                        vis = draw_xyz_axis(
                            vis,
                            ob_in_cam=center_pose,
                            scale=0.1,
                            K=K,
                            thickness=3,
                            transparency=0,
                            is_input_rgb=True,
                        )
                        self._save_debug_image(vis, f"track_{object_name}_vis.png")

                except Exception as e:
                    logger.error(f"Failed to track {object_name}: {e}")
                    poses[object_name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    confidences[object_name] = 0.0
                    failed_objects.append(object_name)

            prediction_time = time.time() - inference_start
            self.prediction_times.append(prediction_time)

            self.frame_count += 1

            # Save combined debug images only if debug level requires it (minimal I/O)
            if self.debug_level >= 1:
                # Only save RGB for level 1+ (most useful for debugging)
                self._save_debug_image(rgb_image, "track_combined_rgb.png")

                # Only save depth for level 2+ (less frequently needed)
                if self.debug_level >= 2:
                    # Convert back to uint16 for saving (more efficient storage)
                    depth_save = (depth_image * 1000).astype(np.uint16)
                    self._save_debug_image(depth_save, "track_combined_depth.png")

            # Optimized logging (only log failures if any)
            if failed_objects:
                logger.warning(
                    f"Failed to track {len(failed_objects)} objects: {failed_objects}"
                )
            elif self.frame_count % 100 == 0:  # Periodic success logging
                logger.debug(
                    f"Successfully tracked {len(poses)} objects, frame {self.frame_count}"
                )

            return {
                "poses": poses,
                "confidences": confidences,
                "frame_id": self.frame_count,
            }

        except Exception as e:
            logger.error(f"Multi-object pose prediction failed: {e}")
            return {
                "poses": {
                    obj: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    for obj in self.active_objects
                },
                "confidences": {obj: 0.0 for obj in self.active_objects},
                "frame_id": self.frame_count,
                "error": f"Pose prediction failed: {e!s}",
            }

    def _handle_reset_session(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Reset the multi-object tracking session."""
        # Clear session-specific data but preserve cached mesh data
        self.estimators.clear()
        # DO NOT clear self.mesh_data - it contains pre-loaded mesh cache!
        self.active_objects.clear()
        self.session_initialized = False
        self.frame_count = 0

        if self.communication_times:
            self._log_performance_stats("session_reset")

        logger.info("Multi-object session reset successfully")

        return {
            "success": True,
            "message": "Multi-object session reset successfully",
        }

    @staticmethod
    def _decode_image(image_bytes: bytes, flags: int) -> np.ndarray:
        """Decode image from bytes."""
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError(f"Expected bytes, got {type(image_bytes)}")

        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, flags)

        if image is None:
            raise ValueError("Failed to decode image")

        return image

    def _save_debug_image(self, image: np.ndarray, filename: str) -> None:
        """Save debug image to the debug directory (optimized with early exit)."""
        # Early exit if debug is disabled
        if self.debug_level == 0:
            return

        try:
            debug_path = self.debug_dir / filename

            # Optimized image saving based on type and size
            if image.dtype == np.uint16:
                # Use cv2 for depth images (faster for uint16)
                cv2.imwrite(str(debug_path), image)
            else:
                # Use imageio for RGB images (better quality)
                imageio.imwrite(str(debug_path), image)
        except Exception as e:
            # Only log warning if debug level is high enough
            if self.debug_level >= 1:
                logger.warning(f"Failed to save debug image {filename}: {e}")

    def on_client_disconnect(self) -> None:
        """Called when a client disconnects."""
        if self.communication_times:
            self._log_performance_stats("client_disconnect")

    def finalize(self) -> None:
        """Called when server is shutting down."""
        if self.communication_times:
            self._log_performance_stats("server_shutdown")

    def _log_gpu_memory_usage(self, stage: str = "unknown") -> None:
        """Log GPU memory usage for debugging CUDA memory issues."""
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
                total_memory = (
                    torch.cuda.get_device_properties(device).total_memory / 1024**3
                )  # GB

                logger.info(
                    f"GPU Memory Usage ({stage}): "
                    f"Allocated: {allocated:.2f}GB, "
                    f"Cached: {cached:.2f}GB, "
                    f"Total: {total_memory:.2f}GB, "
                    f"Free: {total_memory - allocated:.2f}GB"
                )
            else:
                logger.warning("CUDA not available for memory monitoring")
        except ImportError:
            logger.warning("PyTorch not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Failed to get GPU memory usage: {e}")


async def serve_multi_object_foundationpose(
    host: str,
    port: int,
    objects_config: Dict[str, str],
    debug_images: int = 1,
    save_visualizations: bool = False,
    max_vertices: int = 50000,
    decimation_factor: float = 0.5,
    enable_decimation: bool = True,
):
    """Start the Multi-Object FoundationPose WebSocket server."""
    handler = MultiObjectFoundationPoseServer(objects_config)

    # Apply performance settings
    handler.debug_level = debug_images
    handler.save_visualizations = save_visualizations
    handler.save_individual_masks = debug_images >= 2

    # Apply mesh decimation settings
    handler.max_vertices = max_vertices
    handler.decimation_factor = decimation_factor
    handler.enable_mesh_decimation = enable_decimation

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, finalizing server...")
        handler.finalize()
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        async with DirectServer(host, port, handler):
            logger.info(
                f"Multi-Object FoundationPose server ready at ws://{host}:{port}"
            )
            logger.info(f"Available objects: {list(objects_config.keys())}")
            logger.info(f"Debug images will be saved to: {handler.debug_dir}")

            await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Server shutdown complete")
    finally:
        handler.finalize()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Object FoundationPose WebSocket Server"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=10014, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--debug-images",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Debug image level: 0=none, 1=minimal, 2=full (default: 1)",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save pose visualization images (slower but helpful for debugging)",
    )
    parser.add_argument(
        "--max-vertices",
        type=int,
        default=50000,
        help="Maximum vertices per mesh (reduce for lower VRAM, default: 50000)",
    )
    parser.add_argument(
        "--decimation-factor",
        type=float,
        default=0.5,
        help="Decimation factor when mesh is too large (0.1-0.9, default: 0.5)",
    )
    parser.add_argument(
        "--disable-decimation",
        action="store_true",
        help="Disable automatic mesh decimation (use original mesh sizes)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(message)s",
        force=True,
    )

    logging.getLogger(__name__).setLevel(log_level)
    logging.getLogger("websockets").setLevel(log_level)

    # Define available objects
    code_dir = os.path.dirname(os.path.realpath(__file__))
    objects_config = {
        "b5box": f"{code_dir}/demo_data/b5box/mesh/b5box.obj",
        "basket": f"{code_dir}/demo_data/basket/mesh/basket.obj",
    }

    # Validate mesh files
    for obj_name, mesh_path in objects_config.items():
        if not os.path.exists(mesh_path):
            logger.error(f"Mesh file not found for {obj_name}: {mesh_path}")
            return

    logger.info("Starting Multi-Object FoundationPose WebSocket server...")
    asyncio.run(
        serve_multi_object_foundationpose(
            args.host,
            args.port,
            objects_config,
            debug_images=args.debug_images,
            save_visualizations=args.save_visualizations,
            max_vertices=args.max_vertices,
            decimation_factor=args.decimation_factor,
            enable_decimation=not args.disable_decimation,
        )
    )


if __name__ == "__main__":
    main()
