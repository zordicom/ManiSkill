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
from mani_skill.utils.visualization.misc import images_to_video

# depth/segmentation removed; Actor/Link no longer needed
from torch import nn
from torch.distributions import Normal

sys.path.append(str(Path(__file__).parent / "baselines" / "ppo"))
from ppo import Agent

# Add import for ppo_rgb_fast
sys.path.append(str(Path(__file__).parent / "baselines" / "ppo"))
try:
    from ppo_rgb_fast import Agent as RGBFastAgent
    from ppo_rgb_fast import NatureCNN, layer_init
except ImportError:
    RGBFastAgent = None
    NatureCNN = None
    layer_init = None


class PandaDatasetGenerator:
    """Panda motion planner that generates training dataset in the proper format."""

    def __init__(
        self,
        env,
        output_root_dir="training_data/panda_cube_pick_sim/",
        save_video=True,
        video_fps=30,
        use_wrist_cam=False,
        debug=False,
        vis=False,
        base_pose=None,
        visualize_target_grasp_pose=False,
        print_env_info=False,
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
        self.use_wrist_cam = use_wrist_cam
        self.debug = debug
        self.vis = vis
        self.base_pose = base_pose
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.print_env_info = print_env_info

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
        if self.use_wrist_cam:
            (self.episode_dir / "hand_camera_rgb").mkdir(parents=True, exist_ok=True)
        if self.save_video:
            (self.episode_dir / "default_camera_video").mkdir(
                parents=True, exist_ok=True
            )

        print(f"Started episode: {episode_name}")
        print(f"Episode directory: {self.episode_dir}")
        if self.save_video:
            print("Video recording enabled - frames will be saved from default camera")

    def save_frame_data(self, action: np.ndarray = None, phase: str = ""):
        """Save frame data including images, robot state, and action observations."""
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

        # Handle different observation formats
        if isinstance(obs, dict):
            # Dictionary observations (RGB mode)
            extra_obs = obs.get("extra", {})
            obs_dict = obs
        else:
            # Tensor observations (state mode)
            extra_obs = {}
            obs_dict = {"state_tensor": obs}

        # Get state observation (unified for both PPO and PPO_rgb_fast)
        state_obs = {}

        # Handle different observation modes
        if "agent" in obs_dict:
            # Extract agent state (qpos, qvel, etc.) - this is the main state data
            agent_obs = obs_dict["agent"]
            if isinstance(agent_obs, dict):
                for key, value in agent_obs.items():
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().numpy()
                    if hasattr(value, "ndim") and value.ndim > 1:
                        value = value[0]  # Take first batch element
                    state_obs[key] = (
                        value.tolist() if hasattr(value, "tolist") else value
                    )

        # For RGB environments, we might have a flattened state observation
        if "state" in obs_dict:
            state_value = obs_dict["state"]
            if isinstance(state_value, torch.Tensor):
                state_value = state_value.cpu().numpy()
            if hasattr(state_value, "ndim") and state_value.ndim > 1:
                state_value = state_value[0]  # Take first batch element
            state_obs["flattened_state"] = (
                state_value.tolist() if hasattr(state_value, "tolist") else state_value
            )

        # For state-only environments, save the raw state tensor
        if "state_tensor" in obs_dict:
            state_tensor = obs_dict["state_tensor"]
            if isinstance(state_tensor, torch.Tensor):
                state_tensor = state_tensor.cpu().numpy()
            if hasattr(state_tensor, "ndim") and state_tensor.ndim > 1:
                state_tensor = state_tensor[0]  # Take first batch element
            state_obs["raw_state"] = (
                state_tensor.tolist()
                if hasattr(state_tensor, "tolist")
                else state_tensor
            )

        # Extract other relevant observation components (excluding sensor_param)
        for key, value in obs_dict.items():
            if key not in [
                "sensor_data",
                "extra",
                "agent",
                "state",
                "state_tensor",
                "sensor_param",
            ]:
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                if hasattr(value, "ndim") and value.ndim > 1:
                    value = value[0]  # Take first batch element
                state_obs[key] = value.tolist() if hasattr(value, "tolist") else value

        # Process action data - this is the actual action performed by the agent
        action_data = {}
        if action is not None:
            # Convert action to numpy array
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy()
            else:
                action_np = np.array(action)

            # Handle batch dimension
            if hasattr(action_np, "ndim") and action_np.ndim > 1:
                action_np = action_np[0]  # Take first batch element

            # Ensure it's a 1D array
            if action_np.ndim == 0:
                action_np = np.array([action_np])

            action_data["action"] = action_np.tolist()

        # Helper function to ensure JSON serialization
        def convert_to_json_serializable(obj):
            """Recursively convert all torch tensors and numpy arrays to JSON-serializable formats."""
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj

        # Frame numbering (6-digit format)
        frame_str = f"{self.frame_counter:06d}"

        # Save camera images
        image_paths = {}

        # Create streamlined observation entry
        observation = {
            "frame_index": len(self.observations),
            "delta_ms": delta_ms,
            "elapsed_ms": elapsed_ms,
            "phase": phase,
            "state": convert_to_json_serializable(state_obs),
            "action": convert_to_json_serializable(action_data),
            "extra": convert_to_json_serializable(extra_obs),
        }

        # Handle sensor data if available (RGB mode)
        if isinstance(obs, dict) and "sensor_data" in obs:
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

            # Save hand camera RGB if enabled (wrist camera)
            if (
                self.use_wrist_cam
                and "hand_camera" in sensor_data
                and "rgb" in sensor_data["hand_camera"]
            ):
                rgb_image = sensor_data["hand_camera"]["rgb"].cpu().numpy()
                if rgb_image.ndim == 4:
                    rgb_image = rgb_image[0]

                if rgb_image.dtype != np.uint8:
                    rgb_image = (rgb_image * 255).astype(np.uint8)

                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

                # Resize to target resolution
                bgr_image = cv2.resize(
                    bgr_image, self._img_size, interpolation=cv2.INTER_LINEAR
                )

                hand_rgb_path = (
                    self.episode_dir / "hand_camera_rgb" / f"{frame_str}.jpg"
                )
                cv2.imwrite(str(hand_rgb_path), bgr_image)
                image_paths["hand_camera_rgb"] = f"hand_camera_rgb/{frame_str}.jpg"

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

                    # Save individual video frame (BGR format for cv2.imwrite)
                    video_frame_path = (
                        self.episode_dir / "default_camera_video" / f"{frame_str}.jpg"
                    )
                    cv2.imwrite(str(video_frame_path), bgr_frame)
                    image_paths["default_camera_video"] = (
                        f"default_camera_video/{frame_str}.jpg"
                    )

                    # Store frame for video compilation (keep as BGR for consistency)
                    self.video_frames.append(bgr_frame)

            except Exception as e:
                print(f"Warning: Could not capture video frame: {e}")

        # Add image paths to observation
        observation.update(convert_to_json_serializable(image_paths))

        # Add the observation to the list
        self.observations.append(observation)
        self.frame_counter += 1

        if len(self.observations) % 10 == 0:
            print(f"Saved frame {len(self.observations)}: {phase}")
            print(f"  State obs keys: {list(state_obs.keys())}")
            if action_data:
                print(f"  Action shape: {len(action_data.get('action', []))}")
            else:
                print("  No action data")

    def _compile_video(self):
        """Compile video frames into MP4 video files for all cameras using ManiSkill's approach."""
        videos_to_compile = [
            ("default_camera", self.video_frames),
            ("base_camera", self.base_camera_video_frames),
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
            "robot_id": "panda_wristcam" if self.use_wrist_cam else "panda",
            "data_structure": {
                "description": "Streamlined robot state, action, and observation data",
                "observation_format": {
                    "frame_index": "Sequential frame number",
                    "delta_ms": "Time since last frame in milliseconds",
                    "elapsed_ms": "Total elapsed time since episode start in milliseconds",
                    "phase": "Description of current phase (e.g., 'initial', 'step_N')",
                    "state": {
                        "description": "Robot state observations from environment",
                        "qpos": "Joint positions (if in agent observations)",
                        "qvel": "Joint velocities (if in agent observations)",
                        "flattened_state": "Flattened state for RGB environments",
                        "raw_state": "Raw state tensor for state-only environments",
                        "other_keys": "Additional state components from environment",
                    },
                    "action": {
                        "description": "Action data executed by the agent",
                        "action": "Full action vector executed",
                    },
                    "extra": {
                        "description": "Task-specific observations",
                        "obj_pose": "Object pose in world coordinates",
                        "tcp_to_obj_pos": "Vector from TCP to object",
                        "obj_to_goal_pos": "Vector from object to goal",
                        "other_task_info": "Additional task-specific information",
                    },
                    "image_paths": {
                        "base_camera_rgb": "Path to base camera RGB image",
                        "hand_camera_rgb": "Path to hand camera RGB image (if wrist cam enabled)",
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
            if "qpos" in state_data:
                print(f"  qpos dimension: {len(state_data['qpos'])}")
            if "qvel" in state_data:
                print(f"  qvel dimension: {len(state_data['qvel'])}")

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


def load_ppo_rgb_fast_agent(
    env, ckpt_path: str, device: str | torch.device = "cpu"
) -> RGBFastAgent:
    """Load the exact Agent from ppo_rgb_fast.py checkpoint."""
    if RGBFastAgent is None:
        raise ImportError(
            "ppo_rgb_fast.py Agent not available. Please ensure it's importable."
        )

    # Import the flattening wrapper
    # Get action dimensions
    import math

    from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

    # Check if environment is already wrapped - if so, use it directly
    if (
        hasattr(env, "_is_wrapped")
        or not hasattr(env.unwrapped, "_init_raw_obs")
        or "sensor_data" not in env.unwrapped._init_raw_obs
    ):
        # Environment is already wrapped, use it directly
        temp_env = env
    else:
        # Create a temporary wrapped environment to get the right observation structure
        temp_env = FlattenRGBDObservationWrapper(env, rgb=True, state=True)

        # Apply action space wrapper if needed (same as in ppo_rgb_fast.py)
        if isinstance(temp_env.action_space, gym.spaces.Dict):
            from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

            temp_env = FlattenActionSpaceWrapper(temp_env)

    # Get action dimensions from properly wrapped environment
    n_act = math.prod(temp_env.action_space.shape)
    sample_obs = temp_env.reset()[0]

    # Create agent with exact same architecture
    agent = RGBFastAgent(n_act, sample_obs, device=device)

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
    use_wrist_cam: bool = False,
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
        use_wrist_cam: Use panda_wristcam robot configuration with ppo_rgb_fast agent (saves both base and hand camera images) (default: False)
    """
    # Track success statistics
    success_count = 0
    total_episodes = 0

    for episode_idx in range(num_episodes):  # noqa: PLR1702
        print(f"\n{'=' * 60}")
        print(f"GENERATING EPISODE {episode_idx + 1}/{num_episodes}")
        print(f"{'=' * 60}")

        # Create fresh environment for each episode to ensure proper randomization
        env = gym.make(**env_config)

        # Apply wrappers if using wrist cam BEFORE creating PandaDatasetGenerator
        if use_wrist_cam:
            from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

            # Apply wrappers to the raw environment
            env = FlattenRGBDObservationWrapper(env, rgb=True, state=True)

            # Apply action space wrapper if needed (same as in ppo_rgb_fast.py)
            if isinstance(env.action_space, gym.spaces.Dict):
                from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

                env = FlattenActionSpaceWrapper(env)

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
            use_wrist_cam=use_wrist_cam,
        )

        if ppo_checkpoint_path is None:
            raise ValueError(
                "ppo_checkpoint_path must be provided when using PPO rollout."
            )

        # -------------------------------------------------
        # PPO-based rollout (replaces motion-planning logic)
        # -------------------------------------------------

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create agent based on whether wrist cam is enabled
        if use_wrist_cam:
            # For wrist cam, use RGB fast agent with RGB environment
            # The agent expects the dual camera setup (base_camera + hand_camera)
            agent = load_ppo_rgb_fast_agent(env, ppo_checkpoint_path, device)
            # No need for separate state environment since main env provides flattened obs
            temp_state_env = None
        else:
            # Create a temporary state-only env just for agent initialization
            temp_state_env_config = env_config.copy()
            temp_state_env_config["obs_mode"] = "state"
            temp_state_env = gym.make(**temp_state_env_config)
            agent = load_ppo_agent(temp_state_env, ppo_checkpoint_path, device)
            wrapped_env = None

        # Keep the state environment open for getting state observations
        # Reset it with the same seed to keep it synchronized
        if temp_state_env is not None:
            temp_state_env.reset(seed=seed_start + episode_idx)

        # Start episode and capture the initial frame
        generator.start_episode()
        generator.save_frame_data(phase="initial")

        obs = initial_obs
        done = False
        step = 0
        _max_steps = max_steps_per_episode or getattr(env, "_max_episode_steps", 100)
        episode_success = False

        # Initialize wrapped_obs for wrist cam case
        if use_wrist_cam:
            wrapped_obs = obs  # Use the observation from the main environment

        while not done and step < _max_steps:
            with torch.no_grad():
                if use_wrist_cam:
                    # For RGB fast agent, use flattened observations from wrapped environment
                    # Move observation to device and ensure proper format
                    obs_device = {}
                    for key, value in wrapped_obs.items():
                        if isinstance(value, torch.Tensor):
                            obs_device[key] = value.to(device)
                        else:
                            obs_device[key] = torch.from_numpy(value).to(device)

                        # Convert RGB from uint8 to float as expected by the model
                        # (The model will handle the /255.0 normalization)
                        if "rgb" in key:
                            obs_device[key] = obs_device[key].float()

                    # Get action using the actor_mean for deterministic action
                    features = agent.get_features(obs_device)
                    action = agent.actor_mean(features)
                    action = action.squeeze(0).cpu().numpy()
                else:
                    # For basic PPO agent, use state observations
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

                    action = agent.get_action(state_obs_tensor, deterministic=True)
                    action = action.squeeze(0).cpu().numpy()

            # Step both environments with the same action to keep them synchronized
            obs, reward, terminated, truncated, info = env.step(action)
            if not use_wrist_cam:
                temp_state_env.step(action)  # Keep state env synchronized
            if use_wrist_cam:
                wrapped_obs = obs  # Use the observation from the main environment

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

        # Close environments to free resources
        env.close()
        if temp_state_env is not None:
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
        default="panda_wristcam",
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
        default="auto",
        help="Control mode (default: auto - pd_joint_delta_pos for PPO, pd_ee_delta_pos for PPO RGB fast)",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default="/home/gilwoo/workspace/ManiSkill/runs/PickCube-v1__ppo__1__1752538773/final_ckpt.pt",
        help="Path to the trained PPO checkpoint (*.pt) used to drive the policy",
    )
    parser.add_argument(
        "--use-wrist-cam",
        action="store_true",
        help="Use panda_wristcam robot configuration with ppo_rgb_fast agent (saves both base and hand camera images)",
    )

    args = parser.parse_args()

    # Update checkpoint path for wrist cam if not explicitly provided
    if (
        args.use_wrist_cam
        and args.ppo_checkpoint
        == "/home/gilwoo/workspace/ManiSkill/runs/PickCube-v1__ppo__1__1752538773/final_ckpt.pt"
    ):
        args.ppo_checkpoint = "/home/gilwoo/workspace/ManiSkill/runs/PickCube-v1__ppo_rgb_fast__1__1752546740/ckpt_176.pt"
        print(f"Using RGB fast checkpoint for wrist cam: {args.ppo_checkpoint}")
    elif args.use_wrist_cam:
        print(f"Using custom checkpoint for wrist cam: {args.ppo_checkpoint}")
    else:
        print(f"Using state-based PPO checkpoint: {args.ppo_checkpoint}")

    # Create environment configuration dictionary
    env_config = {
        "id": args.env_id,
        "robot_uids": args.robot_uids,
        "obs_mode": args.obs_mode,
        "render_mode": "rgb_array",
        "control_mode": args.control_mode,
    }

    # Update robot configuration and control mode for wrist cam
    if args.use_wrist_cam:
        env_config["robot_uids"] = "panda_wristcam"
        # PPO RGB fast agent uses ee_delta_pos control mode
        env_config["control_mode"] = (
            "pd_ee_delta_pos" if args.control_mode == "auto" else args.control_mode
        )
    else:
        # PPO state agent uses joint_delta_pos control mode
        env_config["control_mode"] = (
            "pd_joint_delta_pos" if args.control_mode == "auto" else args.control_mode
        )

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
        use_wrist_cam=args.use_wrist_cam,
    )
