"""
Copyright 2025 Zordi, Inc. All rights reserved.

Custom Panda robot with conservative action bounds for precise manipulation tasks.
"""

from copy import deepcopy

import numpy as np

from mani_skill.agents.controllers.pd_ee_pose import PDEEPoseControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda


@register_agent()
class PandaSmallBounds(Panda):
    """Panda robot with conservative action bounds for PickCube task.

    Action bounds (pd_ee_delta_pose):
        - Position: ±0.03m (±3cm) per step
        - Rotation: ±0.06 rad (±3.44°) per step

    This provides precise control for small object manipulation while allowing
    sufficient range over 200 steps:
        - Total position: 6m
        - Total rotation: 687°
    """

    uid = "panda_small_bounds"

    @property
    def _controller_configs(self):
        # Get parent configs (includes all default controllers)
        configs = super()._controller_configs

        # Create custom pd_ee_delta_pose with conservative bounds
        arm_pd_ee_delta_pose_small = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.03,  # ±3cm position delta per step
            pos_upper=0.03,
            rot_lower=-0.06,  # ±3.44° rotation delta per step (in radians)
            rot_upper=0.06,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        # Also create target delta version
        arm_pd_ee_target_delta_pose_small = deepcopy(arm_pd_ee_delta_pose_small)
        arm_pd_ee_target_delta_pose_small.use_target = True

        # Get gripper config from existing setup
        gripper_config = configs["pd_ee_delta_pose"]["gripper"]

        # Override pd_ee_delta_pose with custom bounds
        configs["pd_ee_delta_pose"] = dict(
            arm=arm_pd_ee_delta_pose_small,
            gripper=gripper_config,
        )

        # Also override target delta pose version
        configs["pd_ee_target_delta_pose"] = dict(
            arm=arm_pd_ee_target_delta_pose_small,
            gripper=gripper_config,
        )

        return configs
