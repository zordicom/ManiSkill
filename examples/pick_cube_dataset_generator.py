"""
Copyright 2025 Zordi, Inc. All rights reserved.
"""

import json
import os

# Import Agent directly from PPO training script
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import cv2
import gymnasium as gym
import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv

# depth/segmentation removed; Actor/Link no longer needed
from torch import nn
from torch.distributions import Normal

sys.path.append(str(Path(__file__).parent / "baselines" / "ppo"))
from ppo import Agent


class PandaDatasetGenerator:
    """Panda motion planner that generates training dataset in the proper format."""

    def __init__(
        self,
        env,
        output_root_dir="training_data/panda_cube_pick_sim/",
        save_video=True,
        video_fps=30,
        **kwargs,
    ):
        # No motion-planner superclass; store env/robot explicitly
        self.env = env
        self.robot = env.agent.robot
        self.output_root_dir = Path(output_root_dir)
        self.observations: List[Dict[str, Any]] = []
        self.episode_name = ""
        self.episode_dir = None
        self.frame_counter = 0
        self.start_time = None
        self.last_timestamp = None

        # Video recording settings
        self.save_video = save_video
        self.video_fps = video_fps
        self.video_frames = []  # Store frames for default camera video
        self.base_camera_video_frames = []  # Store frames for base camera video

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
        self.video_frames = []  # Reset video frames for new episode
        self.base_camera_video_frames = []  # Reset base camera video frames

        # Create episode directory structure
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

        print(f"Started episode: {episode_name}")
        print(f"Episode directory: {self.episode_dir}")
        if self.save_video:
            print("Video recording enabled - frames will be saved from default camera")

    def save_frame_data(self, action: np.ndarray = None, phase: str = ""):
        """Save frame data including images and observations."""
        if self.episode_dir is None:
            raise ValueError("Episode not started. Call start_episode() first.")

        # Get current observations
        obs = self.env.get_obs()

        # Calculate timing
        current_time = datetime.now()
        elapsed_ms = (current_time - self.start_time).total_seconds() * 1000

        # Calculate delta_ms from last frame
        if self.last_timestamp is None:
            delta_ms = 0.0
        else:
            delta_ms = (current_time - self.last_timestamp).total_seconds() * 1000
        self.last_timestamp = current_time

        # Get task-specific extra observations (obj_pose, tcp_to_obj_pos, obj_to_goal_pos)
        extra_obs = obs.get("extra", {})

        # Frame numbering (6-digit format)
        frame_str = f"{self.frame_counter:06d}"

        # Save camera images
        image_paths = {}

        # Create observation entry
        observation = {
            "frame_index": len(self.observations),
            "delta_ms": delta_ms,
            "elapsed_ms": elapsed_ms,
            "extra": {
                k: (v.tolist() if hasattr(v, "tolist") else v)
                for k, v in extra_obs.items()
            },
        }

        if "sensor_data" in obs:
            sensor_data = obs["sensor_data"]

            # Debug: Print available cameras (only for first few frames)
            if len(self.observations) < 3:
                print(f"Available cameras: {list(sensor_data.keys())}")
                for cam_name, cam_data in sensor_data.items():
                    print(f"  {cam_name}: {list(cam_data.keys())}")

            # Save base camera RGB
            if "base_camera" in sensor_data and "rgb" in sensor_data["base_camera"]:
                rgb_image = sensor_data["base_camera"]["rgb"].cpu().numpy()
                if rgb_image.ndim == 4:
                    rgb_image = rgb_image[0]

                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                # Resize to target resolution
                bgr_image = cv2.resize(
                    bgr_image, self._img_size, interpolation=cv2.INTER_LINEAR
                )

                base_rgb_path = (
                    self.episode_dir / "base_camera_rgb" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(base_rgb_path), bgr_image)
                image_paths["base_camera_rgb"] = f"base_camera_rgb/{frame_str}.jpg"

                # Store frame for base camera video compilation
                if self.save_video:
                    self.base_camera_video_frames.append(bgr_image)

            # depth and segmentation skipped for base camera

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

                    # Convert RGB to BGR for OpenCV
                    if video_frame.shape[-1] == 3:
                        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)

                    # Save individual video frame
                    video_frame_path = (
                        self.episode_dir / "default_camera_video" / f"{frame_str}.jpg"
                    )
                    cv2.imwrite(str(video_frame_path), video_frame)
                    image_paths["default_camera_video"] = (
                        f"default_camera_video/{frame_str}.jpg"
                    )

                    # Store frame for video compilation
                    self.video_frames.append(video_frame)

            except Exception as e:
                print(f"Warning: Could not capture video frame: {e}")

        # Add image paths to observation
        observation.update(image_paths)

        # Add the observation to the list
        self.observations.append(observation)
        self.frame_counter += 1

        if len(self.observations) % 10 == 0:
            print(f"Saved frame {len(self.observations)}: {phase}")

    def _colorize_segmentation(self, seg_image: np.ndarray) -> np.ndarray:
        """Convert segmentation mask to colorized visualization using predetermined colors.

        Args:
            seg_image: Segmentation mask with integer labels

        Returns:
            Colorized BGR image for visualization
        """
        # Get unique segment IDs
        unique_ids = np.unique(seg_image)

        # Create a colorized version
        height, width = seg_image.shape
        colorized = np.zeros((height, width, 3), dtype=np.uint8)

        # Apply predetermined colors to each segment
        for seg_id in unique_ids:
            mask = seg_image == seg_id
            if seg_id in self.color_mapping:
                # Use predetermined color
                colorized[mask] = self.color_mapping[seg_id]
            else:
                # Fallback: use hash-based color for unknown segments
                np.random.seed(int(seg_id))
                color = np.random.randint(50, 255, 3)
                colorized[mask] = color

        return colorized

    def _convert_to_simple_ids(self, seg_image: np.ndarray) -> np.ndarray:
        """Convert segmentation image to simplified IDs (1-5) as UINT8 PNG.

        Args:
            seg_image: Original segmentation mask with arbitrary integer labels

        Returns:
            Simplified segmentation mask with IDs 0-5 as UINT8
        """
        # Create output image
        height, width = seg_image.shape
        simple_seg = np.zeros((height, width), dtype=np.uint8)

        # Map each pixel to simplified ID
        for orig_id, simple_id in self.id_remapping.items():
            mask = seg_image == orig_id
            simple_seg[mask] = simple_id

        return simple_seg

    def _compile_video(self):
        """Compile video frames into MP4 video files for all cameras."""
        videos_to_compile = [
            ("default_camera", self.video_frames),
            ("base_camera", self.base_camera_video_frames),
        ]

        for video_name, frames in videos_to_compile:
            if not frames:
                print(f"No {video_name} frames to compile")
                continue

            video_path = self.episode_dir / f"{self.episode_name}_{video_name}.mp4"

            try:
                # Get video dimensions from first frame
                height, width = frames[0].shape[:2]

                # Define codec and create VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(
                    str(video_path), fourcc, self.video_fps, (width, height)
                )

                # Write frames to video
                for frame in frames:
                    video_writer.write(frame)

                # Release video writer
                video_writer.release()

                print(f"{video_name.title()} video compiled successfully: {video_path}")
                print(f"Video contains {len(frames)} frames at {self.video_fps} FPS")

            except Exception as e:
                print(f"Error compiling {video_name} video: {e}")

    def follow_path(self, result, refine_steps: int = 0):
        """Override follow_path to save dataset at every control frame."""
        n_step = result["position"].shape[0]
        print(f"Following path with {n_step} steps")

        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]
            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Save frame data
            self.save_frame_data(action, f"trajectory_step_{i}")

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def open_gripper(self):
        """Override open_gripper to save dataset at every control frame."""
        print(f"Opening gripper from {self.gripper_state} to {self.OPEN}")
        self.gripper_state = self.OPEN
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()

        for step in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Save frame data
            self.save_frame_data(action, f"open_gripper_step_{step}")

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

    def close_gripper(self, t: int = 6, gripper_state: float = None):
        """Override close_gripper to save dataset at every control frame."""
        old_gripper_state = self.gripper_state
        self.gripper_state = self.CLOSED if gripper_state is None else gripper_state
        print(f"Closing gripper from {old_gripper_state} to {self.gripper_state}")
        qpos = self.robot.get_qpos()[0, :-2].cpu().numpy()

        for step in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])

            obs, reward, terminated, truncated, info = self.env.step(action)

            # Save frame data
            self.save_frame_data(action, f"close_gripper_step_{step}")

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()

        return obs, reward, terminated, truncated, info

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

        # Add segmentation legends if they exist
        if hasattr(self, "segmentation_mapping") and self.segmentation_mapping:
            camera_metadata["segmentation_legend"] = {
                "segmentation_mapping": self.segmentation_mapping,
                "color_mapping": self.color_mapping,
                "id_remapping": self.id_remapping,
                "description": {
                    "segmentation_mapping": "Maps segmentation IDs to entity information (type, name, description)",
                    "color_mapping": "Maps segmentation IDs to BGR color values for visualization",
                    "id_remapping": "Maps original segmentation IDs to simplified IDs (0=background, 1-5=key objects)",
                },
            }
            print(
                f"✅ Added segmentation legend with {len(self.segmentation_mapping)} entities"
            )

        return camera_metadata

    def finish_episode(self):
        """Finish the episode and save the JSON file."""
        if self.episode_dir is None:
            raise ValueError("Episode not started.")

        # Get real camera metadata from environment
        camera_metadata = self._get_camera_metadata()

        # Create the format with metadata and observations
        metadata = {
            "episode_name": self.episode_name,
            "robot_id": "panda",
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
        json_path = self.episode_dir / f"{self.episode_name}.json"
        with open(json_path, "w") as f:
            json.dump(episode_data, f, indent=2)

        print(f"Episode finished: {self.episode_name}")
        print(f"Total frames: {len(self.observations)}")
        print(f"Episode data saved to: {json_path}")

        # Compile video if video recording is enabled
        if self.save_video and len(self.base_camera_video_frames) > 0:
            self._compile_video()

        # Reset for next episode
        self.observations = []
        self.episode_dir = None
        self.frame_counter = 0
        self.last_timestamp = None
        self.segmentation_mapping = {}
        self.color_mapping = {}
        self.id_remapping = {}
        self.video_frames = []
        self.base_camera_video_frames = []


# =====================================================
# PPO AGENT LOADING
# =====================================================


class DummyVectorEnv:
    """Minimal wrapper to make single env compatible with Agent constructor"""

    def __init__(self, env):
        # PPO was trained with state observations, so we need to create a proper Box space
        # The state observation is typically a flat vector
        if (
            hasattr(env.observation_space, "shape")
            and env.observation_space.shape is not None
        ):
            self.single_observation_space = env.observation_space
        else:
            # For complex observation spaces, we need to flatten them
            import gymnasium as gym

            sample_obs = env.observation_space.sample()
            flattened_obs = gym.spaces.flatten(env.observation_space, sample_obs)
            self.single_observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=flattened_obs.shape, dtype=np.float32
            )
        self.single_action_space = env.action_space


def load_ppo_agent(env, ckpt_path: str, device: str | torch.device = "cpu") -> Agent:
    """Load the exact Agent from ppo.py checkpoint."""
    # Create dummy vector env wrapper for Agent constructor
    dummy_envs = DummyVectorEnv(env)

    # Create agent with exact same architecture
    agent = Agent(dummy_envs).to(device)

    # Load checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict, strict=True)
    agent.eval()

    return agent


def generate_pick_cube_dataset(
    env_config: dict,
    num_episodes: int = 1,
    output_root_dir: str = "",
    seed_start: int = 42,
    debug: bool = False,
    vis: bool = False,
    save_video: bool = True,
    video_fps: int = 30,
    use_table_origin: bool = True,
    ppo_checkpoint_path: str | None = None,
    max_steps_per_episode: int | None = None,
):
    """Generate multiple episodes of pick cube dataset.

    Args:
        env_config: Environment configuration dictionary for gym.make()
        num_episodes: Number of episodes to generate
        output_root_dir: Root directory for dataset
        seed_start: Starting seed for episodes
        debug: Enable debug output
        vis: Enable visualization
        save_video: Enable video recording from default camera (default: True)
        video_fps: Frame rate for compiled video (default: 30)
        use_table_origin: Use table origin coordinate system (default: True)
    """
    # Track success statistics
    success_count = 0
    total_episodes = 0

    for episode_idx in range(num_episodes):
        print(f"\n{'=' * 60}")
        print(f"GENERATING EPISODE {episode_idx + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # Create fresh environment for each episode to ensure proper randomization
        env = gym.make(**env_config)

        # Reset environment with unique seed and capture initial observation
        initial_obs, _ = env.reset(seed=seed_start + episode_idx)

        # -------------------------------------------------
        # Make the goal sphere (green target) visible to cameras
        # -------------------------------------------------
        u = env.unwrapped
        print(f"u: {u}")
        if hasattr(u, "goal_site"):
            if hasattr(u, "_hidden_objects") and u.goal_site in u._hidden_objects:
                u._hidden_objects.remove(u.goal_site)
            # Ensure its visuals are enabled
            if hasattr(u.goal_site, "show_visual"):
                u.goal_site.show_visual()
            elif hasattr(u.goal_site, "set_visibility"):
                # Fall back to directly toggling visibility if the helper is missing
                try:
                    u.goal_site.set_visibility(True)
                except Exception as e:
                    print(
                        f"[WARN] Failed to unhide goal sphere via set_visibility: {e}"
                    )

        # Create dataset generator
        generator = PandaDatasetGenerator(
            env,
            output_root_dir=output_root_dir,
            save_video=save_video,
            video_fps=video_fps,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=debug,
        )

        if ppo_checkpoint_path is None:
            raise ValueError(
                "ppo_checkpoint_path must be provided when using PPO rollout."
            )

        # -------------------------------------------------
        # PPO-based rollout (replaces motion-planning logic)
        # -------------------------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a temporary state-only env just for agent initialization
        temp_state_env_config = env_config.copy()
        temp_state_env_config["obs_mode"] = "state"
        temp_state_env = gym.make(**temp_state_env_config)
        agent = load_ppo_agent(temp_state_env, ppo_checkpoint_path, device)

        # Keep the state environment open for getting state observations
        # Reset it with the same seed to keep it synchronized
        temp_state_env.reset(seed=seed_start + episode_idx)

        # Start episode and capture the initial frame
        generator.start_episode()
        generator.save_frame_data(phase="initial")

        obs = initial_obs
        done = False
        step = 0
        _max_steps = max_steps_per_episode or getattr(env, "_max_episode_steps", 100)
        episode_success = False

        while not done and step < _max_steps:
            # Get state observation from the synchronized state environment
            state_obs = temp_state_env.get_obs()

            # Convert to numpy if it's a tensor
            if isinstance(state_obs, torch.Tensor):
                state_obs = state_obs.cpu().numpy().flatten()
            else:
                state_obs = np.array(state_obs).flatten()

            # Convert to tensor
            state_obs_tensor = (
                torch.from_numpy(state_obs).float().to(device).unsqueeze(0)
            )

            with torch.no_grad():
                action = agent.get_action(state_obs_tensor, deterministic=True)
                action = action.squeeze(0).cpu().numpy()

            # Step both environments with the same action to keep them synchronized
            obs, reward, terminated, truncated, info = env.step(action)
            temp_state_env.step(action)  # Keep state env synchronized

            # Check for success
            if info.get("success"):
                episode_success = True
                print(f"✅ Episode {episode_idx + 1} SUCCESS at step {step}!")

            # Save data for this frame
            generator.save_frame_data(action, phase=f"step_{step}")

            done = terminated or truncated
            step += 1

        generator.finish_episode()

        # Update success statistics
        total_episodes += 1
        if episode_success:
            success_count += 1

        print(
            f"Episode {episode_idx + 1} completed: {'SUCCESS' if episode_success else 'FAILED'}"
        )
        print(
            f"Current success rate: {success_count}/{total_episodes} ({100 * success_count / total_episodes:.1f}%)"
        )
        print("Episode completed successfully!")

        # Close both environments to free resources
        env.close()
        temp_state_env.close()

    # Final success rate summary
    final_success_rate = (
        100 * success_count / total_episodes if total_episodes > 0 else 0
    )
    print(f"\n{'=' * 60}")
    print("DATASET GENERATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total episodes: {total_episodes}")
    print(f"Successful episodes: {success_count}")
    print(f"Success rate: {final_success_rate:.1f}%")
    print(f"{'=' * 60}")

    print(f"\nDataset generation complete! Generated {num_episodes} episodes.")


if __name__ == "__main__":
    import argparse
    import os

    import gymnasium as gym
    import mani_skill

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate Panda pick cube dataset")
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=3,
        help="Number of episodes to generate (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data/panda_cube_pick_sim/",
        help="Root directory for dataset output (default: ~/training_data/panda_cube_pick_sim/)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting seed for episodes (default: 42)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Enable visualization",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=30,
        help="Frame rate for compiled video (default: 30)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="PickCube-v1",
        help="Environment ID to use (default: PickCube-v1)",
    )
    parser.add_argument(
        "--robot-uids",
        type=str,
        default="panda",
        help="Robot UIDs to use (default: panda)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="rgb",
    )
    parser.add_argument(
        "--control-mode",
        type=str,
        default="pd_joint_delta_pos",
        help="Control mode (default: pd_joint_pos)",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default="/home/gilwoo/workspace/ManiSkill/runs/PickCube-v1__ppo__1__1752538773/final_ckpt.pt",
        help="Path to the trained PPO checkpoint (*.pt) used to drive the policy",
    )

    args = parser.parse_args()

    # Create environment configuration dictionary
    env_config = {
        "id": args.env_id,
        "robot_uids": args.robot_uids,
        "obs_mode": args.obs_mode,
        "render_mode": "rgb_array",
        "control_mode": args.control_mode,
    }

    # Generate dataset
    generate_pick_cube_dataset(
        env_config,
        num_episodes=args.n_episodes,
        output_root_dir=os.path.expanduser(args.output_dir),
        seed_start=args.seed_start,
        vis=args.vis,
        save_video=not args.no_video,
        video_fps=args.video_fps,
        ppo_checkpoint_path=os.path.expanduser(args.ppo_checkpoint),
    )
