"""
PickBox-v1 configuration file for different robots.
Based on PickCube configurations but adapted for the pick-and-place task with baskets.
"""

import numpy as np

from .bimanual_pick_place_cfgs import ROBOT_CONFIGS as BIMANUAL_ROBOT_CONFIGS

PICK_BOX_CONFIGS = {
    "panda": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        # Add bimanual arm configurations
        "left_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["panda"]["left_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["panda"]["left_arm"]["home_qpos"],
        },
        "right_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["panda"]["right_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["panda"]["right_arm"]["home_qpos"],
        },
    },
    "panda_wristcam": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        # Add bimanual arm configurations
        "left_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["panda_wristcam"]["left_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["panda_wristcam"]["left_arm"][
                "home_qpos"
            ],
        },
        "right_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["panda_wristcam"]["right_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["panda_wristcam"]["right_arm"][
                "home_qpos"
            ],
        },
    },
    "a1_galaxea": {
        # Use same tight thresholds as PickCube for A1 Galaxea
        "cube_half_size": 0.012,
        "goal_thresh": 0.015,  # Tight threshold like PickCube
        "cube_spawn_half_size": 0.06,
        "cube_spawn_center": (-0.05, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.25, 0, 0.5],
        "sensor_cam_target_pos": [-0.08, 0, 0.08],
        "human_cam_eye_pos": [0.5, 0.6, 0.5],
        "human_cam_target_pos": [-0.05, 0.0, 0.3],
        # Override bimanual spawn positions for single-arm use
        "object_spawn_center": [0.1, -0.2],  # Closer to A1 robot
        "goal_spawn_center": [0.1, 0.0],  # Reasonable distance from object
        # Static top camera configuration (user-specified)
        "static_top_cam_pos": [-0.228881, 0.0366335, 1.41154],
        "static_top_cam_quat": [
            0.931246,
            0.000830829,
            0.364383,
            -0.00212252,
        ],  # w,x,y,z format
        "static_top_cam_width": 224,
        "static_top_cam_height": 224,
        "static_top_cam_fovy": 1.06,  # Radians
        "static_top_cam_near": 0.1,
        "static_top_cam_far": 1000.0,
        # Add bimanual arm configurations
        "left_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["a1_galaxea"]["left_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["a1_galaxea"]["left_arm"]["home_qpos"],
        },
        "right_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["a1_galaxea"]["right_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["a1_galaxea"]["right_arm"]["home_qpos"],
        },
    },
    "xarm6_robotiq": {
        "cube_half_size": 0.02,
        "goal_thresh": 0.025,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "max_goal_height": 0.3,
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
        # Add bimanual arm configurations
        "left_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["xarm6_robotiq"]["left_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["xarm6_robotiq"]["left_arm"][
                "home_qpos"
            ],
        },
        "right_arm": {
            "pose": BIMANUAL_ROBOT_CONFIGS["xarm6_robotiq"]["right_arm"]["pose"].p,
            "home_qpos": BIMANUAL_ROBOT_CONFIGS["xarm6_robotiq"]["right_arm"][
                "home_qpos"
            ],
        },
    },
}
