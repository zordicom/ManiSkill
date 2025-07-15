#!/usr/bin/env python3
"""
RGB Dataset Generator for PickCube-v1 with panda_wristcam robot.

This script generates datasets using a pre-trained PPO RGB fast agent.
Specifically designed for panda_wristcam robot configuration.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append("examples/baselines/ppo")
import mani_skill.envs
from mani_skill.utils.visualization.misc import images_to_video
from mani_skill.utils.wrappers.flatten import (
    FlattenActionSpaceWrapper,
    FlattenRGBDObservationWrapper,
)
from ppo_rgb_fast import Agent as RGBFastAgent


class RGBDatasetGenerator:
    """Dataset generator for RGB observations with panda_wristcam."""

    def __init__(
        self,
        env,
        output_root_dir: str = "training_data/panda_cube_pick_rgb/",
        save_video: bool = True,
        video_fps: int = 30,
    ):
        self.env = env
        self.output_root_dir = Path(output_root_dir)
        self.save_video = save_video
        self.video_fps = video_fps

        # Episode tracking
        self.episode_name = ""
        self.episode_dir = None
        self.observations: List[Dict[str, Any]] = []
        self.video_frames = []
        self.base_camera_video_frames = []
        self.hand_camera_video_frames = []
        self.frame_counter = 0
        self.start_time = None
        self.last_timestamp = None

        # Target image resolution (width, height)
        self._img_size = (224, 224)

    def start_episode(self, episode_name: str = None):
        """Start a new episode and create directory structure."""
        if episode_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = timestamp

        self.episode_name = episode_name
        self.observations = []
        self.frame_counter = 0
        self.start_time = datetime.now()
        self.last_timestamp = None
        self.video_frames = []
        self.base_camera_video_frames = []
        self.hand_camera_video_frames = []

        # Create episode directory structure (match state generator format)
        date_str = self.start_time.strftime("%Y/%m/%d")
        self.episode_dir = (
            self.output_root_dir
            / "panda_cube_pick"
            / "cube_pick"
            / "data"
            / date_str
            / episode_name
        )

        # Create camera directories
        (self.episode_dir / "base_camera_rgb").mkdir(parents=True, exist_ok=True)
        (self.episode_dir / "hand_camera_rgb").mkdir(parents=True, exist_ok=True)
        if self.save_video:
            (self.episode_dir / "default_camera_video").mkdir(
                parents=True, exist_ok=True
            )

        print(f"Started episode: {episode_name}")
        print(f"Episode directory: {self.episode_dir}")
        if self.save_video:
            print("Video recording enabled - frames will be saved from default camera")

    def save_frame_data(self, obs, action=None, info=None, phase="step"):
        """Save frame data including RGB images and metadata."""
        if self.episode_dir is None:
            raise ValueError("Episode not started. Call start_episode() first.")

        # Calculate timing
        current_time = datetime.now()
        elapsed_ms = (current_time - self.start_time).total_seconds() * 1000

        # Calculate delta_ms from last frame
        if self.last_timestamp is None:
            delta_ms = 0.0
        else:
            delta_ms = (current_time - self.last_timestamp).total_seconds() * 1000
        self.last_timestamp = current_time

        # Save RGB images
        rgb_data = obs["rgb"]  # Shape: (batch, H, W, C*num_cameras)

        # Convert to numpy if it's a tensor
        if isinstance(rgb_data, torch.Tensor):
            rgb_data = rgb_data.cpu().numpy()

        # Remove batch dimensions if present
        if rgb_data.ndim == 4:
            rgb_data = rgb_data.squeeze(0)  # Remove batch dimension

        # Split the concatenated RGB channels (6 channels = 3 base + 3 hand)
        if rgb_data.shape[-1] == 6:
            base_rgb = rgb_data[:, :, :3]  # First 3 channels
            hand_rgb = rgb_data[:, :, 3:]  # Last 3 channels
        else:
            # Fallback if channels are different
            base_rgb = rgb_data[:, :, :3]
            hand_rgb = rgb_data[:, :, :3]

        # Ensure images are uint8
        if base_rgb.dtype != np.uint8:
            base_rgb = (base_rgb * 255).astype(np.uint8)
        if hand_rgb.dtype != np.uint8:
            hand_rgb = (hand_rgb * 255).astype(np.uint8)

        # Resize images to target resolution
        base_rgb = cv2.resize(base_rgb, self._img_size, interpolation=cv2.INTER_LINEAR)
        hand_rgb = cv2.resize(hand_rgb, self._img_size, interpolation=cv2.INTER_LINEAR)

        # Save as JPEG images
        frame_str = f"{self.frame_counter:06d}"

        # Convert RGB to BGR for OpenCV
        base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        hand_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR)

        # Save images
        base_rgb_path = self.episode_dir / "base_camera_rgb" / f"{frame_str}.jpg"
        hand_rgb_path = self.episode_dir / "hand_camera_rgb" / f"{frame_str}.jpg"

        cv2.imwrite(str(base_rgb_path), base_bgr)
        cv2.imwrite(str(hand_rgb_path), hand_bgr)

        # Store frames for video compilation
        if self.save_video:
            self.base_camera_video_frames.append(base_bgr)
            self.hand_camera_video_frames.append(hand_bgr)

        # Extract state information and structure it properly
        state_data = obs["state"]
        if isinstance(state_data, torch.Tensor):
            state_data = state_data.cpu().numpy()

        # Remove batch dimension if present
        if state_data.ndim > 1:
            state_data = state_data[0]

        # Structure state data (for panda_wristcam, this should be the flattened state)
        structured_state = {"flattened_state": state_data.tolist()}

        # Structure action data
        action_data = {}
        if action is not None:
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if hasattr(action, "ndim") and action.ndim > 1:
                action = action[0]  # Remove batch dimension
            action_data["action"] = action.tolist()

        # Process extra info
        extra_info = {}
        if info:
            for key, value in info.items():
                if isinstance(value, torch.Tensor):
                    extra_info[key] = value.cpu().numpy().tolist()
                elif isinstance(value, np.ndarray):
                    extra_info[key] = value.tolist()
                else:
                    extra_info[key] = value

        # Create observation entry (match state generator format)
        observation = {
            "frame_index": len(self.observations),
            "delta_ms": delta_ms,
            "elapsed_ms": elapsed_ms,
            "phase": phase,
            "state": structured_state,
            "action": action_data,
            "extra": extra_info,
            "base_camera_rgb": f"base_camera_rgb/{frame_str}.jpg",
            "hand_camera_rgb": f"hand_camera_rgb/{frame_str}.jpg",
        }

        # Capture video frame from default human render camera
        if self.save_video:
            try:
                # Get default human render camera image
                video_frame = self.env.render()

                if video_frame is not None:
                    # Convert to numpy if it's a tensor
                    if hasattr(video_frame, "cpu"):
                        video_frame = video_frame.cpu().numpy()

                    # Handle different shapes
                    if video_frame.ndim == 4:
                        video_frame = video_frame[0]  # Remove batch dimension

                    # Ensure it's uint8
                    if video_frame.dtype != np.uint8:
                        video_frame = (video_frame * 255).astype(np.uint8)

                    # Convert RGB to BGR for OpenCV image saving
                    if video_frame.shape[-1] == 3:
                        bgr_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                    else:
                        bgr_frame = video_frame

                    # Save individual video frame
                    video_frame_path = (
                        self.episode_dir / "default_camera_video" / f"{frame_str}.jpg"
                    )
                    cv2.imwrite(str(video_frame_path), bgr_frame)
                    observation["default_camera_video"] = (
                        f"default_camera_video/{frame_str}.jpg"
                    )

                    # Store frame for video compilation
                    self.video_frames.append(bgr_frame)

            except Exception as e:
                print(f"Warning: Could not capture video frame: {e}")

        self.observations.append(observation)
        self.frame_counter += 1

        if len(self.observations) % 10 == 0:
            print(f"Saved frame {len(self.observations)}: {phase}")

    def _get_camera_metadata(self) -> Dict:
        """Extract real camera intrinsic matrices and parameters from the environment."""
        # Get sensor parameters from the environment
        sensor_params = self.env.get_sensor_params()

        # Extract camera dimensions and intrinsics
        camera_metadata = {
            "k_mats": {},
            "camera_image_dimensions": {},
            "image_processing": {
                "resize_method": "cv2.INTER_LINEAR",
                "crop_method": "center_crop",
                "normalization": "0_to_1",
            },
        }

        # Get camera configurations and sensor data
        for sensor_name, sensor in self.env.scene.sensors.items():
            if hasattr(sensor, "camera") and hasattr(sensor, "config"):
                # Get intrinsic matrix from sensor parameters
                if sensor_name in sensor_params:
                    intrinsic_cv = sensor_params[sensor_name]["intrinsic_cv"]
                    # Convert from tensor to list (flatten 3x3 matrix row-wise)
                    if hasattr(intrinsic_cv, "cpu"):
                        intrinsic_cv = intrinsic_cv.cpu().numpy()
                    k_matrix = (
                        intrinsic_cv[0].flatten().tolist()
                    )  # Take first batch element

                    camera_metadata["k_mats"][sensor_name] = k_matrix

                    # Get camera dimensions
                    camera_metadata["camera_image_dimensions"][sensor_name] = {
                        "width": sensor.config.width,
                        "height": sensor.config.height,
                    }

                    print(f"✅ Extracted camera {sensor_name}: K={k_matrix[:3]}...")
                else:
                    print(f"⚠️ No sensor params found for {sensor_name}")

        return camera_metadata

    def _compile_video(self):
        """Compile video frames into MP4 video files for all cameras using ManiSkill's approach."""
        videos_to_compile = [
            ("default_camera", self.video_frames),
            ("base_camera", self.base_camera_video_frames),
            ("hand_camera", self.hand_camera_video_frames),
        ]

        for video_name, frames in videos_to_compile:
            if not frames:
                print(f"No {video_name} frames to compile")
                continue

            try:
                # Convert BGR frames back to RGB for imageio (since imageio expects RGB)
                rgb_frames = []
                for frame in frames:
                    # Convert BGR to RGB for imageio
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frames.append(rgb_frame)

                # Use ManiSkill's images_to_video function
                images_to_video(
                    rgb_frames,
                    output_dir=str(self.episode_dir),
                    video_name=f"{self.episode_name}_{video_name}",
                    fps=self.video_fps,
                    quality=5,  # Same quality as ManiSkill default
                    verbose=True,
                )

                print(
                    f"{video_name.title()} video compiled successfully with {len(frames)} frames at {self.video_fps} FPS"
                )

            except Exception as e:
                print(f"Error compiling {video_name} video: {e}")

    def finish_episode(self):
        """Finish episode and save all data."""
        if self.episode_dir is None:
            raise ValueError("Episode not started.")

        # Get real camera metadata from environment
        camera_metadata = self._get_camera_metadata()

        # Create the format with metadata and observations (match state generator)
        metadata = {
            "episode_name": self.episode_name,
            "robot_id": "panda_wristcam",
            "data_structure": {
                "description": "Streamlined robot state, action, and observation data",
                "observation_format": {
                    "frame_index": "Sequential frame number",
                    "delta_ms": "Time since last frame in milliseconds",
                    "elapsed_ms": "Total elapsed time since episode start in milliseconds",
                    "phase": "Description of current phase (e.g., 'initial', 'step_N')",
                    "state": {
                        "description": "Robot state observations from environment",
                        "flattened_state": "Flattened state for RGB environments",
                        "other_keys": "Additional state components from environment",
                    },
                    "action": {
                        "description": "Action data executed by the agent",
                        "action": "Full action vector executed",
                    },
                    "extra": {
                        "description": "Task-specific observations",
                        "success": "Whether the task was completed successfully",
                        "is_obj_placed": "Whether object was placed in target location",
                        "is_robot_static": "Whether robot is in static state",
                        "is_grasped": "Whether object is grasped by robot",
                        "other_task_info": "Additional task-specific information",
                    },
                    "image_paths": {
                        "base_camera_rgb": "Path to base camera RGB image",
                        "hand_camera_rgb": "Path to hand camera RGB image",
                        "default_camera_video": "Path to default camera video frame",
                    },
                },
            },
            **camera_metadata,
        }

        metadata["coordinate_system"] = {
            "type": "world",
            "description": "All poses are in world coordinates",
        }

        episode_data = {
            "metadata": metadata,
            "observations": self.observations,
        }

        # Save episode data JSON file
        json_path = self.episode_dir / "observations.json"
        with open(json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Episode finished: {self.episode_name}")
        print(f"Total frames: {len(self.observations)}")
        print(f"Episode data saved to: {json_path}")

        # Print summary of data structure
        if len(self.observations) > 0:
            sample_obs = self.observations[0]
            print("\nData structure summary:")
            print(f"  State keys: {list(sample_obs.get('state', {}).keys())}")
            print(f"  Action keys: {list(sample_obs.get('action', {}).keys())}")
            print(f"  Extra keys: {list(sample_obs.get('extra', {}).keys())}")

            # Print dimensions
            state_data = sample_obs.get("state", {})
            if "flattened_state" in state_data:
                print(
                    f"  flattened_state dimension: {len(state_data['flattened_state'])}"
                )

            action_data = sample_obs.get("action", {})
            if "action" in action_data:
                print(f"  action dimension: {len(action_data['action'])}")

        # Compile video if video recording is enabled
        if self.save_video and len(self.base_camera_video_frames) > 0:
            self._compile_video()

        # Reset for next episode
        self.observations = []
        self.episode_dir = None
        self.frame_counter = 0
        self.last_timestamp = None
        self.video_frames = []
        self.base_camera_video_frames = []
        self.hand_camera_video_frames = []


def load_rgb_agent(env, checkpoint_path: str, device: str = "cuda") -> RGBFastAgent:
    """Load PPO RGB fast agent from checkpoint."""
    print(f"Loading RGB agent from: {checkpoint_path}")

    # Get action dimensions - use math.prod for robustness with multi-dimensional action spaces
    n_act = math.prod(env.action_space.shape)

    # Get sample observation
    sample_obs = env.reset()[0]

    # Create agent
    agent = RGBFastAgent(n_act, sample_obs, device=device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()

    print(f"Agent loaded successfully with {n_act} action dimensions")
    return agent


def generate_rgb_dataset(
    checkpoint_path: str,
    num_episodes: int = 10,
    output_dir: str = "training_data/panda_cube_pick_rgb/",
    seed_start: int = 42,
    max_steps: int = 50,
):
    """Generate RGB dataset using PPO RGB fast agent."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment configuration - match ppo_rgb_fast.py exactly
    env_config = {
        "id": "PickCube-v1",
        "robot_uids": "panda_wristcam",
        "control_mode": "pd_ee_delta_pos",
        "obs_mode": "rgb",
        "render_mode": "all",  # Match PPO training default
        "sim_backend": "physx_cuda",  # Match PPO training explicit setting
    }

    success_count = 0

    for episode_idx in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"GENERATING EPISODE {episode_idx + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # Create fresh environment
        env = gym.make(**env_config)

        # Apply wrappers
        env = FlattenRGBDObservationWrapper(env, rgb=True, state=True)
        if isinstance(env.action_space, gym.spaces.Dict):
            env = FlattenActionSpaceWrapper(env)

        # Load agent
        agent = load_rgb_agent(env, checkpoint_path, device)

        # Create dataset generator
        generator = RGBDatasetGenerator(
            env=env,
            output_root_dir=output_dir,
            save_video=True,
            video_fps=30,
        )

        # Reset environment
        obs, info = env.reset(seed=seed_start + episode_idx)

        # Start episode
        generator.start_episode()

        # Save initial frame
        generator.save_frame_data(obs, action=None, info=info, phase="initial")

        # Episode loop
        done = False
        step = 0
        episode_success = False

        while not done and step < max_steps:
            with torch.no_grad():
                # Prepare observation for agent
                obs_device = {}
                for key, value in obs.items():
                    if isinstance(value, torch.Tensor):
                        obs_device[key] = value.to(device)
                    else:
                        obs_device[key] = torch.from_numpy(value).to(device)

                    # Convert RGB to float
                    if "rgb" in key:
                        obs_device[key] = obs_device[key].float()

                # Get action from agent
                features = agent.get_features(obs_device)
                action = agent.actor_mean(features)  # Deterministic action
                action = action.squeeze(0).cpu().numpy()

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                # Save frame data
                generator.save_frame_data(obs, action, info, phase=f"step_{step}")

                # Check success
                if info.get("success", False):
                    episode_success = True
                    print(f"✅ Episode {episode_idx + 1} SUCCESS at step {step}!")
                    break

        # Finish episode
        generator.finish_episode()

        if episode_success:
            success_count += 1

        print(
            f"Episode {episode_idx + 1} completed: {'SUCCESS' if episode_success else 'FAILED'}"
        )
        print(
            f"Current success rate: {success_count}/{episode_idx + 1} ({100 * success_count / (episode_idx + 1):.1f}%)"
        )

        # Clean up
        env.close()

    # Final summary
    final_success_rate = 100 * success_count / num_episodes if num_episodes > 0 else 0
    print(f"\n{'=' * 60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes: {num_episodes}")
    print(f"Successful episodes: {success_count}")
    print(f"Success rate: {final_success_rate:.1f}%")
    print(f"{'=' * 60}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate RGB dataset for PickCube-v1")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/gilwoo/workspace/ManiSkill/runs/PickCube-v1__ppo_rgb_fast__1__1752546740/ckpt_326.pt",
        help="Path to PPO RGB fast checkpoint",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes to generate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data/panda_cube_pick_rgb/",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--seed-start", type=int, default=42, help="Starting seed for episodes"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Maximum steps per episode"
    )

    args = parser.parse_args()

    # Generate dataset
    generate_rgb_dataset(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
