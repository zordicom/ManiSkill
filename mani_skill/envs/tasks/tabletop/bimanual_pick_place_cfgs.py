"""
BimanualPickPlace-v1 configuration file for two-robot bimanual setups.
This file contains task-specific configurations including robot poses, camera positions,
and task parameters optimized for different robot types (Panda, A1 Galaxea, etc.).
"""

import numpy as np
import sapien

from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.structs.pose import Pose

# Robot-specific configurations
ROBOT_CONFIGS = {
    "panda": {
        "left_arm": {
            "pose": Pose.create_from_pq(
                p=[0, 0.4, 0.82], q=[1, 0, 0, 0]
            ),  # Left arm position (Y-axis layout, facing +X)
            "home_qpos": np.array([
                # Mirrored Panda home pose for left arm
                np.deg2rad(-20.0),  # panda_joint1 (mirrored)
                np.deg2rad(-45.0),  # panda_joint2
                np.deg2rad(0.0),  # panda_joint3
                np.deg2rad(-135.0),  # panda_joint4
                np.deg2rad(0.0),  # panda_joint5
                np.deg2rad(90.0),  # panda_joint6
                np.deg2rad(45.0),  # panda_joint7
                0.04,  # panda_finger_joint1 (open)
                0.04,  # panda_finger_joint2 (open)
            ]),
        },
        "right_arm": {
            "pose": Pose.create_from_pq(
                p=[0, -0.4, 0.82], q=[1, 0, 0, 0]
            ),  # Right arm position (Y-axis layout, facing +X)
            "home_qpos": np.array([
                # Standard Panda home pose for right arm
                np.deg2rad(20.0),  # panda_joint1
                np.deg2rad(-45.0),  # panda_joint2
                np.deg2rad(0.0),  # panda_joint3
                np.deg2rad(-135.0),  # panda_joint4
                np.deg2rad(0.0),  # panda_joint5
                np.deg2rad(90.0),  # panda_joint6
                np.deg2rad(45.0),  # panda_joint7
                0.04,  # panda_finger_joint1 (open)
                0.04,  # panda_finger_joint2 (open)
            ]),
        },
    },
    "panda_wristcam": {
        "left_arm": {
            "pose": Pose.create_from_pq(
                p=[0, 0.4, 0.82], q=[1, 0, 0, 0]
            ),  # Left arm position (Y-axis layout, facing +X)
            "home_qpos": np.array([
                # Mirrored Panda home pose for left arm
                np.deg2rad(-20.0),  # panda_joint1 (mirrored)
                np.deg2rad(-45.0),  # panda_joint2
                np.deg2rad(0.0),  # panda_joint3
                np.deg2rad(-135.0),  # panda_joint4
                np.deg2rad(0.0),  # panda_joint5
                np.deg2rad(90.0),  # panda_joint6
                np.deg2rad(45.0),  # panda_joint7
                0.04,  # panda_finger_joint1 (open)
                0.04,  # panda_finger_joint2 (open)
            ]),
        },
        "right_arm": {
            "pose": Pose.create_from_pq(
                p=[0, -0.4, 0.82], q=[1, 0, 0, 0]
            ),  # Right arm position (Y-axis layout, facing +X)
            "home_qpos": np.array([
                # Standard Panda home pose for right arm
                np.deg2rad(20.0),  # panda_joint1
                np.deg2rad(-45.0),  # panda_joint2
                np.deg2rad(0.0),  # panda_joint3
                np.deg2rad(-135.0),  # panda_joint4
                np.deg2rad(0.0),  # panda_joint5
                np.deg2rad(90.0),  # panda_joint6
                np.deg2rad(45.0),  # panda_joint7
                0.04,  # panda_finger_joint1 (open)
                0.04,  # panda_finger_joint2 (open)
            ]),
        },
    },
    "a1_galaxea": {
        "left_arm": {
            "pose": Pose.create_from_pq(
                p=[0, 0.355, 0.82], q=[1, 0, 0, 0]
            ),  # Left arm position (matches XML: Y=+0.355, facing +X)
            "home_qpos": np.array([
                np.deg2rad(-41.4),  # arm_joint1 (mirrored from right)
                np.deg2rad(0.11),  # arm_joint2
                np.deg2rad(-51.7),  # arm_joint3
                np.deg2rad(88.9),  # arm_joint4
                np.deg2rad(-65.4),  # arm_joint5
                np.deg2rad(-2.1),  # arm_joint6
                0.02,  # gripper1_axis (open)
                0.02,  # gripper2_axis (mimic)
            ]),
        },
        "right_arm": {
            "pose": Pose.create_from_pq(
                p=[0, -0.355, 0.82], q=[1, 0, 0, 0]
            ),  # Right arm position (matches XML: Y=-0.355, facing +X)
            "home_qpos": np.array([
                np.deg2rad(42.0),  # arm_joint1
                np.deg2rad(45.0),  # arm_joint2
                np.deg2rad(-48.0),  # arm_joint3
                np.deg2rad(80.0),  # arm_joint4
                np.deg2rad(-35.0),  # arm_joint5
                np.deg2rad(15.0),  # arm_joint6
                0.02,  # gripper1_axis (open)
                0.02,  # gripper2_axis (mimic)
            ]),
        },
    },
    "xarm6_robotiq": {
        "left_arm": {
            "pose": Pose.create_from_pq(
                p=[0, 0.4, 0.82], q=[1, 0, 0, 0]
            ),  # Left arm position (Y-axis layout, facing +X)
            "home_qpos": np.array([
                # XArm6 home pose for left arm (mirrored from right)
                0.0,  # joint1 (mirrored)
                0.22,  # joint2
                -1.23,  # joint3
                0.0,  # joint4
                1.01,  # joint5
                0.0,  # joint6
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # 6 gripper joints (open)
            ]),
        },
        "right_arm": {
            "pose": Pose.create_from_pq(
                p=[0, -0.4, 0.82], q=[1, 0, 0, 0]
            ),  # Right arm position (Y-axis layout, facing +X)
            "home_qpos": np.array([
                # XArm6 home pose for right arm - using rest keyframe values
                0.0,  # joint1
                0.22,  # joint2
                -1.23,  # joint3
                0.0,  # joint4
                1.01,  # joint5
                0.0,  # joint6
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # 6 gripper joints (open)
            ]),
        },
    },
}

BIMANUAL_PICK_PLACE_CONFIG = {
    # Task parameters
    "task": {
        "b5box_half_size": [
            0.025,
            0.025,
            0.025,
        ],  # Approximate dimensions (now using STL collision file)
        "basket_internal_size": [
            0.24,
            0.18,
            0.10,
        ],  # Internal basket dimensions (24cm x 18cm x 10cm) based on primitive collision boxes
        "success_thresh": 0.08,  # Distance threshold for success (b5box center to basket center)
        "max_episode_steps": 200,
        "reward_scale": 1.0,
    },
    # Scene configuration (matches bimanual_scene.xml)
    "scene": {
        "table": {
            "pose": Pose.create_from_pq(
                p=[0.27, 0, 0.766], q=[1, 0, 0, 0]
            ),  # Table position from XML
            "size": [0.34, 0.61, 0.05],  # Half-sizes from XML
        },
        "basket": {
            "spawn_region": {
                "center": [0.3, 0.15],  # Position from XML (plastic bin)
                "half_size": [0.05, 0.05],  # Small randomization around basket
            },
        },
        "b5box": {
            "spawn_region": {
                "center": [0.3, -0.15],  # Position from XML (target object)
                "half_size": [0.04, 0.04],  # Randomization area
                "height_offset": 0.059,  # Height above table surface (0.825 - 0.766 = 0.059)
            },
        },
    },
    # Camera configuration - using look_at style for intuitive camera positioning
    "cameras": {
        "static_top": {
            # Camera position and orientation from XML
            "pose": Pose.create_from_pq(
                p=[-0.23, 0, 1.3],  # Position from XML
                q=[0.5, 0.5, -0.5, 0.5],  # Looking down at workspace
            ),
            "fov": 60 * np.pi / 180,  # 60 degrees from XML
            "width": 128,
            "height": 128,
        },
        "human_render": {
            "pose": Pose.create_from_pq(
                p=[0.844194, -0.0453854, 1.37297],
                q=[0.00106472, 0.193666, 0.000210285, -0.981067],
            ),
            "fov": 60 * np.pi / 180,
            "width": 512,
            "height": 512,
            # Alternative human render camera viewpoints using look_at(camera_pos, target_pos):
            #
            # Side view from left:
            # "pose": look_at([0.27, -1.2, 1.0], [0.27, 0, 0.8]),
            #
            # High angled view (cinematographic):
            # "pose": look_at([1.0, -0.5, 1.5], [0.27, 0, 0.8]),
            #
            # Close-up view focused on workspace:
            # "pose": look_at([0.5, 0, 0.9], [0.27, 0, 0.8]),
            #
            # Wide view from back:
            # "pose": look_at([-0.8, 0, 1.2], [0.27, 0, 0.8]),
        },
    },
    # Reward configuration
    "rewards": {
        "reach_reward_scale": 1.0,
        "grasp_reward_scale": 2.0,
        "place_reward_scale": 5.0,
        "success_reward": 10.0,
        "distance_threshold": {
            "reach": 0.1,  # Distance to object for reach reward
            "grasp": 0.05,  # Distance to object for grasp reward
            "place": 0.05,  # Distance to basket for place reward
        },
    },
}
