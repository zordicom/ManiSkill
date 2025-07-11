# pick_approach_joint_env.py
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .ik_controller import BimanualIKController, load_bimanual_model

SUCCESS_POS_EPS = 0.012  # ≤1.2 cm in xyz
SUCCESS_ANG_EPS = 10 * np.pi / 180  # ≤10°


class PickApproachJointEnv(gym.Env):
    """
    Gym‑compatible wrapper around the Phase‑1 logic of `AdvancedIKPickAndPlace`
    (ik_example.py). Similar to PickApproachEnv but uses joint position deltas
    instead of Cartesian pose deltas for actions. Episode terminates once the EE
    is suitably aligned w.r.t. the target object.

    Simulation modes:
    - Kinematic mode (default): Full kinematic simulation, no physics stepping
      - Robot arms: Kinematic control (direct joint position setting)
      - Objects: Static (no physics simulation)
      - Much faster and more stable for learning
    - Dynamic mode: Mixed kinematic+dynamic simulation
      - Robot arms: Kinematic control (direct joint position setting)
      - Objects: Dynamic simulation (gravity, collisions, contacts)
      - Control frequency: 25Hz (0.04s per env.step)
      - Physics frequency: 500Hz (0.002s timestep)
      - 20 physics substeps per control step

    OBSERVATION SPACE (45D):
    [0:6]   Joint positions (6) - right arm joint angles
    [6:7]   Gripper state (1) - 1.0 for open, 0.0 for closed
    [7:10]  EE position (3) - end-effector position in world coordinates
    [10:14] EE quaternion (4) - end-effector orientation (w,x,y,z)
    [14:17] Target position (3) - target object position in world coordinates
    [17:21] Target quaternion (4) - target object orientation (w,x,y,z)
    [21:24] Bin position (3) - bin position in world coordinates
    [24:28] Bin quaternion (4) - bin orientation (w,x,y,z)
    [28:31] EE x-axis (3) - end-effector x-axis unit vector
    [31:34] Target x-axis (3) - target object x-axis unit vector (world-aligned)
    [34:37] EE y-axis (3) - end-effector y-axis unit vector
    [37:40] Target y-axis (3) - target object y-axis unit vector
    [40:43] Delta XYZ (3) - position difference from EE to target
    [43:44] Delta angle x-axis (1) - angle between EE and target x-axes
    [44:45] Delta angle z-axis (1) - angle between EE z-axis and downward

    EXPERT BEHAVIOR MATCHING:
    The expert (ik_expert_pick_place.py) uses consistent world x-axis alignment:
    1. Chooses target x-axis direction within 90° of world x-axis (1,0,0)
    2. Computes gripper orientation to align with this world-aligned target x-axis
    3. Maintains downward z-axis orientation throughout the approach

    This environment matches the expert behavior through:
    - Target x-axis selection that stays within 90° of world x-axis
    - Primary reward terms for position, x-axis alignment, and z-axis downward
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        xml_path: str | Path = None,
        max_steps: int = 250,
        render_mode: str | None = None,
        control_freq: float = 25.0,  # Hz - frequency of env.step()
        kinematic_mode: bool = True,  # Default to kinematic mode for better learning
    ):
        super().__init__()
        # Load model and data directly using the utility function
        self.model, self.data = load_bimanual_model(xml_path)
        # Create controller with existing model and data
        self.controller = BimanualIKController(self.model, self.data)
        self.viewer = None
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.kinematic_mode = kinematic_mode
        self.step_ctr = 0

        # Physics and control timing (only used in dynamic mode)
        self.control_freq = control_freq  # 25 Hz
        self.control_timestep = 1.0 / control_freq  # 0.04 seconds
        self.physics_timestep = self.model.opt.timestep  # 0.002 seconds from XML
        self.n_substeps = int(
            self.control_timestep / self.physics_timestep
        )  # 20 substeps

        print("Environment initialized:")
        print(f"  Simulation mode: {'Kinematic' if self.kinematic_mode else 'Dynamic'}")
        if self.kinematic_mode:
            print("  Full kinematic simulation - fast and stable for learning")
        else:
            print(
                f"  Control frequency: {self.control_freq} Hz ({self.control_timestep:.3f}s per step)"
            )
            print(
                f"  Physics frequency: {1.0 / self.physics_timestep:.1f} Hz ({self.physics_timestep:.3f}s per substep)"
            )
            print(f"  Substeps per control step: {self.n_substeps}")

        # -------- Observation --------------
        #  ▸ Joint positions (6) – helps exploration
        #  ▸ Gripper State (1) – 1.0 for open, 0.0 for closed
        #  ▸ EE pose  (position 3, orientation 4)
        #  ▸ Target pose (3 + 4)
        #  ▸ Bin pose (3 + 4)
        #  ▸ EE x-axis unit vector (3)
        #  ▸ Target x-axis unit vector (world-aligned) (3)
        #  ▸ EE y-axis unit vector (3)
        #  ▸ Target y-axis unit vector (3)
        #  ▸ Delta XYZ to target (3)
        #  ▸ Delta Radians Angle X-axis to target's X-axis (1)
        #  ▸ Delta Radians Angle Z-axis to (0,0, -1) (1)
        #     = 6 + 1 + 7 + 7 + 7 + 3 + 3 + 3+ 3+ 3 + 1 + 1 = 45 dims total

        high = np.ones(45, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # -------- Action --------------------
        # Joint position deltas for the right arm (6 joints)
        # scaled inside _apply_action()
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], np.float32),
            dtype=np.float32,
        )

        # Joint control state
        self.target_joint_positions = None

    # ------------------------------------------------------------------ #
    # API                                                                #
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._randomize_scene()
        self.step_ctr = 0

        # Initialize target joint positions to current positions
        self.target_joint_positions = self.controller.get_joint_positions(
            "right"
        ).copy()

        obs = self._get_obs()
        if self.render_mode == "human":
            self._maybe_render()
        return obs, {"success": False}

    def step(self, action):
        self.step_ctr += 1
        self._apply_action(action)  # Set target joint positions and run physics
        reward, done, success = self._compute_reward()

        obs = self._get_obs()
        info = {"success": success}

        if self.render_mode == "human":
            self._maybe_render()

        time_out = self.step_ctr >= self.max_steps
        done = done or time_out
        return obs, reward, done, False, info  # Gymnasium v1 API

    def render(self):
        self._maybe_render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _randomize_scene(self):
        """
        Reset robot to home. Keep target near its original XML position with
        small offsets (±3cm in x,y) and random orientation.
        """
        # Set both arms to home positions using the controller
        self.controller._set_home_positions()
        mujoco.mj_forward(self.model, self.data)

        # Sample pose for 'target' body with freejoint
        # Find the freejoint for the target body
        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )

        # Find the joint associated with the target body
        target_joint_id = None
        for joint_id in range(self.model.njnt):
            if self.model.jnt_bodyid[joint_id] == target_body_id:
                target_joint_id = joint_id
                break

        if target_joint_id is not None:
            # Get original position from model (qpos0 contains initial joint positions)
            joint_qpos_start = self.model.jnt_qposadr[target_joint_id]
            original_pos = self.model.qpos0[
                joint_qpos_start : joint_qpos_start + 3
            ].copy()
            original_quat = self.model.qpos0[
                joint_qpos_start + 3 : joint_qpos_start + 7
            ].copy()

            # Sample small offsets (±3cm = ±0.03m)
            xy_offset = self.np_random.uniform(-0.03, 0.03, size=2)
            new_pos = original_pos.copy()
            new_pos[0] += xy_offset[0]  # x offset
            new_pos[1] += xy_offset[1]  # y offset
            # Keep z the same

            # Random orientation (yaw only for simplicity)
            yaw = self.np_random.uniform(-np.pi, np.pi)
            new_quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # w,x,y,z

            # Set the freejoint qpos values (position + quaternion)
            self.data.qpos[joint_qpos_start : joint_qpos_start + 3] = new_pos
            self.data.qpos[joint_qpos_start + 3 : joint_qpos_start + 7] = new_quat

        mujoco.mj_forward(self.model, self.data)

    def _get_obs(self):
        ee_pos, ee_quat = self.controller.get_current_pose("right")
        tgt_pos, tgt_quat = self._get_target_pose()
        bin_pos, bin_quat = self._get_bin_pose()

        # Get joint positions
        joints = self.controller.get_joint_positions("right")

        # Get gripper state (assuming 1.0 for open, 0.0 for closed - simplified for now)
        gripper_state = np.array([1.0])  # TODO: Implement actual gripper state reading

        # Extract rotation matrices
        ee_rot_matrix = self.data.site_xmat[self.controller.right_site_id].reshape(3, 3)
        tgt_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_pose"
        )
        tgt_rot_matrix = self.data.site_xmat[tgt_site_id].reshape(3, 3)

        # Extract axis vectors
        ee_x = ee_rot_matrix[:, 0].copy()
        ee_y = ee_rot_matrix[:, 1].copy()
        ee_z = ee_rot_matrix[:, 2].copy()

        tgt_x_raw = tgt_rot_matrix[:, 0].copy()
        tgt_y = tgt_rot_matrix[:, 1].copy()

        # Choose target X direction that has angle < 90 degrees with world x-axis (1,0,0)
        world_x = np.array([1, 0, 0])
        tgt_x = tgt_x_raw if np.dot(tgt_x_raw, world_x) >= 0 else -tgt_x_raw

        # Normalise (should already be unit, but numerical safety)
        ee_x /= np.linalg.norm(ee_x) + 1e-8
        ee_y /= np.linalg.norm(ee_y) + 1e-8
        ee_z /= np.linalg.norm(ee_z) + 1e-8
        tgt_x /= np.linalg.norm(tgt_x) + 1e-8
        tgt_y /= np.linalg.norm(tgt_y) + 1e-8

        # Delta XYZ to target
        delta_xyz = tgt_pos - ee_pos

        # Delta angle between EE x-axis and target x-axis
        delta_angle_x = np.array([np.arccos(np.clip(np.dot(ee_x, tgt_x), -1, 1))])

        # Delta angle between EE z-axis and downward direction (0,0,-1)
        downward = np.array([0, 0, -1])
        delta_angle_z = np.array([np.arccos(np.clip(np.dot(ee_z, downward), -1, 1))])

        # Concatenate in the order specified in the comment:
        # Joint positions (6) + Gripper State (1) + EE pose (3+4) + Target pose (3+4) + Bin pose (3+4) +
        # EE x-axis (3) + Target x-axis (3) + EE y-axis (3) + Target y-axis (3) +
        # Delta XYZ (3) + Delta angle x-axis (1) + Delta angle z-axis (1)
        return np.concatenate([
            joints,  # 6
            gripper_state,  # 1
            ee_pos,  # 3
            ee_quat,  # 4
            tgt_pos,  # 3
            tgt_quat,  # 4
            bin_pos,  # 3
            bin_quat,  # 4
            ee_x,  # 3
            tgt_x,  # 3
            ee_y,  # 3
            tgt_y,  # 3
            delta_xyz,  # 3
            delta_angle_x,  # 1
            delta_angle_z,  # 1
        ]).astype(np.float32)

    def _apply_action(self, action):
        """
        Apply joint position deltas. In kinematic mode, just updates positions and
        computes forward kinematics. In dynamic mode, runs physics simulation.
        """
        # Scale joint deltas: 0.016 radians (~0.9 degrees) per action unit
        # Reduced from 0.12 to slow down movement for ~100 step optimal policies
        joint_delta = 0.016 * action

        # Update target joint positions
        self.target_joint_positions += joint_delta

        # Apply joint limits to targets
        if hasattr(self.model, "jnt_range") and self.model.jnt_range is not None:
            right_joint_ids = self.controller.right_joint_ids
            for i, joint_id in enumerate(right_joint_ids):
                if joint_id < len(self.model.jnt_range):
                    joint_range = self.model.jnt_range[joint_id]
                    if joint_range[0] < joint_range[1]:  # Valid range
                        self.target_joint_positions[i] = np.clip(
                            self.target_joint_positions[i],
                            joint_range[0],
                            joint_range[1],
                        )

        if self.kinematic_mode:
            # Kinematic mode: Set positions and compute forward kinematics only
            self.controller.set_joint_positions("right", self.target_joint_positions)
            # Compute forward kinematics to update all derived quantities
            mujoco.mj_forward(self.model, self.data)
        else:
            # Dynamic mode: Run physics simulation for control_timestep duration
            # Robot arms move kinematically, objects move dynamically
            for substep in range(self.n_substeps):
                # Set robot joint positions directly (kinematic control)
                self.controller.set_joint_positions(
                    "right", self.target_joint_positions
                )

                # Step physics (objects fall under gravity, collisions, etc.)
                mujoco.mj_step(self.model, self.data)

                # Maintain kinematic control for both arms (zeros velocities/accelerations)
                # This ensures robot stays kinematic while objects remain dynamic
                self.controller.maintain_kinematic_control()

    def _compute_reward(self):
        ee_pos, ee_quat = self.controller.get_current_pose("right")
        tgt_pos, tgt_quat = self._get_target_pose()

        # --- position error (m) & orientation error (rad) ----------
        d_pos = np.linalg.norm(ee_pos - tgt_pos)
        ee_rot_matrix = self.data.site_xmat[self.controller.right_site_id].reshape(3, 3)
        ee_x = ee_rot_matrix[:, 0]
        ee_z = ee_rot_matrix[:, 2]  # z-axis of end-effector

        tgt_x_raw = self.data.site_xmat[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_pose")
        ].reshape(3, 3)[:, 0]

        # Choose target X direction that has angle < 90 degrees with world x-axis (1,0,0)
        world_x = np.array([1, 0, 0])
        tgt_x = tgt_x_raw if np.dot(tgt_x_raw, world_x) >= 0 else -tgt_x_raw

        ang = np.arccos(np.clip(np.dot(ee_x, tgt_x), -1, 1))

        # --- z-axis alignment with downward direction (0,0,-1) ----
        downward = np.array([0, 0, -1])
        z_alignment = np.dot(
            ee_z, downward
        )  # Ranges from -1 to 1, 1 is perfect alignment

        # --- Properly scaled reward with three primary objectives ---------
        # 1. Position error penalty (scaled appropriately)
        # 2. X-axis orientation error penalty (scaled appropriately)
        # 3. Z-axis downward alignment reward
        pos_reward = -1.0 * d_pos  # Scale by distance in meters (10x penalty per meter)
        ang_reward = -0.5 * ang  # Scale by angle in radians (2x penalty per radian)
        z_reward = 0.1 * z_alignment  # Reward for downward orientation

        shaped = pos_reward + ang_reward + z_reward

        done = (d_pos < SUCCESS_POS_EPS) and (ang < SUCCESS_ANG_EPS)
        success = bool(done)

        # Add small step penalty to encourage efficiency
        if not done:
            shaped -= 0.01  # Small step penalty
        if success:
            shaped += 100.0  # Large success bonus

        return shaped * 0.1, done, success  # Scale by 0.1 to make it more stable

    # ---------------- utilities ----------------
    def _maybe_render(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()

    def _get_target_pose(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_pose")
        pos = self.data.site_xpos[site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return pos, quat

    def _get_bin_pose(self):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_pose")
        pos = self.data.site_xpos[site_id].copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[site_id])
        return pos, quat

    def get_alignment_diagnostics(self):
        """Get diagnostic information about world alignment behavior (matching expert focus).

        Returns:
            dict: Diagnostic information including world alignment metrics
        """
        ee_rot_matrix = self.data.site_xmat[self.controller.right_site_id].reshape(3, 3)
        ee_x = ee_rot_matrix[:, 0]
        ee_z = ee_rot_matrix[:, 2]

        tgt_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "target_pose"
        )
        tgt_x_raw = self.data.site_xmat[tgt_site_id].reshape(3, 3)[:, 0]

        world_x = np.array([1, 0, 0])
        downward = np.array([0, 0, -1])

        # Choose world-aligned target x-axis (same as expert)
        tgt_x = tgt_x_raw if np.dot(tgt_x_raw, world_x) >= 0 else -tgt_x_raw

        return {
            "ee_world_x_alignment": float(np.dot(ee_x, world_x)),
            "tgt_world_x_alignment": float(np.dot(tgt_x, world_x)),
            "ee_world_x_angle_deg": float(
                np.degrees(np.arccos(np.clip(np.abs(np.dot(ee_x, world_x)), 0, 1)))
            ),
            "tgt_world_x_angle_deg": float(
                np.degrees(np.arccos(np.clip(np.abs(np.dot(tgt_x, world_x)), 0, 1)))
            ),
            "ee_world_x_sign": float(1.0 if np.dot(ee_x, world_x) >= 0 else -1.0),
            "tgt_world_x_sign": float(1.0 if np.dot(tgt_x, world_x) >= 0 else -1.0),
            "ee_z_downward_alignment": float(np.dot(ee_z, downward)),
            "ee_tgt_x_alignment": float(np.dot(ee_x, tgt_x)),
            "ee_tgt_x_angle_deg": float(
                np.degrees(np.arccos(np.clip(np.dot(ee_x, tgt_x), -1, 1)))
            ),
            "world_direction_consistent": bool(
                np.sign(np.dot(ee_x, world_x)) == np.sign(np.dot(tgt_x, world_x))
            ),
        }
