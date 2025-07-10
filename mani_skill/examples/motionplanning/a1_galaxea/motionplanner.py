import glob
import os
import shutil
from typing import Dict

import mplib
import numpy as np
import sapien.core as sapien
import torch

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.utils.structs.pose import to_sapien_pose


class A1GalaxeaMotionPlanningSolver(PandaArmMotionPlanningSolver):
    """Motion-planning helper tailored to the A1 Galaxea single-arm configuration.

    The solver re-uses all generic logic from :class:`~PandaArmMotionPlanningSolver` and
    only customises the following aspects:

    1. **Move-group name** – the SRDF defines a group called ``tcp`` for the end-effector.
    2. **DOF counts** – the arm has 6 joints, followed by a single scalar gripper joint
       (the second finger is handled via mimic).
    3. **Gripper command range** – the prismatic gripper opens at ``≈0.03 m`` and closes
       at ``0.0 m``.
    """

    # Prismatic finger joint values (metres) – sign is reversed on Galaxea
    OPEN: float = -0.03  # fully open (fingers retracted)
    CLOSED: float = 0.025  # fully closed (fingers gripping)

    def __init__(
        self,
        env: BaseEnv,
        debug: bool = False,
        vis: bool = True,
        base_pose: sapien.Pose | None = None,
        visualize_target_grasp_pose: bool = True,
        print_env_info: bool = True,
        joint_vel_limits: float = 0.9,
        joint_acc_limits: float = 0.9,
    ) -> None:
        # Handle bimanual mode by setting up the correct robot reference
        self.env = env
        self.base_env: BaseEnv = env.unwrapped
        self.env_agent: BaseAgent = self.base_env.agent

        # Handle both single-arm and bimanual modes
        if hasattr(self.env_agent, "robot"):
            # Single robot mode
            self.robot = self.env_agent.robot
            self.is_bimanual = False
        elif hasattr(self.env_agent, "agents"):
            # Bimanual mode - use right arm (active agent, index 1)
            self.robot = self.env_agent.agents[1].robot
            self.is_bimanual = True
            # Also update env_agent to point to the right arm for other operations
            self.env_agent = self.env_agent.agents[1]
        else:
            raise ValueError(
                f"Unable to determine robot configuration from agent: {type(self.env_agent)}"
            )

        # Manually initialize parent class attributes to avoid robot access issues
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        self.base_pose = to_sapien_pose(base_pose)
        self.planner = self.setup_planner()
        self.control_mode = self.base_env.control_mode
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.visualize_target_grasp_pose = visualize_target_grasp_pose
        self.elapsed_steps = 0
        self.use_point_cloud = False
        self.collision_pts_changed = False
        self.all_collision_pts = None

        # Set up grasp pose visual (simplified version without panda-specific logic)
        self.grasp_pose_visual = None
        if self.vis and self.visualize_target_grasp_pose:
            # Skip grasp pose visual for A1 since it's panda-specific
            pass
        print(
            f"🔍 [A1 DEBUG] A1GalaxeaMotionPlanningSolver.__init__() - setting gripper_state to OPEN: {self.OPEN}"
        )
        self.gripper_state = self.OPEN

        # Initialize robot to home keyframe so gripper starts properly open
        print(
            "🔍 [A1 DEBUG] Initializing robot to home keyframe for proper gripper position"
        )
        home_qpos = self.env_agent.keyframes["home"].qpos[:7]
        print(f"🔍 [A1 DEBUG] Setting robot to home qpos: {home_qpos}")

        # The robot has 8 joints but keyframe only has 7 (master gripper joint)
        # We need to expand to include the mimic joint
        full_qpos = np.zeros(8)
        full_qpos[:7] = home_qpos  # 6 arm joints + 1 gripper joint
        full_qpos[7] = home_qpos[6]  # Set gripper2_axis to same as gripper1_axis

        print(f"🔍 [A1 DEBUG] Expanded qpos for robot: {full_qpos}")
        self.robot.set_qpos(full_qpos)

        # Verify the robot is in the correct position
        actual_qpos = self.robot.get_qpos()[0].cpu().numpy()
        actual_gripper = actual_qpos[6:8]
        print(
            f"🔍 [A1 DEBUG] After home initialization - actual gripper joints: {actual_gripper}"
        )

        # Give the robot a moment to settle in the new position
        for _ in range(3):
            if self.is_bimanual:
                action = self._format_action_for_bimanual(actual_qpos[:7])
            else:
                action = actual_qpos[:7]  # Single-arm mode
            obs, reward, terminated, truncated, info = self.env.step(action)

    def _format_action_for_bimanual(
        self, right_arm_action: np.ndarray
    ) -> Dict[str, torch.Tensor]:
        """Return action dict required by MultiAgent.

        right_arm_action: np.ndarray of shape (7,) generated by planner for right arm.
        Left arm will be held fixed by sending its current qpos each control step.
        """
        # Reference agents
        left_agent = self.base_env.agent.agents[0]
        right_agent = self.base_env.agent.agents[1]
        device = self.base_env.device

        # --- right arm (convert & batch) ---
        # Ensure we only send the DOF-sized part (ignore velocities if provided)
        right_dof = right_agent.controller.single_action_space.shape[0]
        right_action_clipped = right_arm_action[:right_dof]
        right_tensor = torch.as_tensor(
            right_action_clipped, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # --- left arm: keep at current qpos, slice to controller DOF ---
        left_dof = left_agent.controller.single_action_space.shape[0]
        left_qpos = left_agent.robot.get_qpos()[0, :left_dof]
        left_tensor = left_qpos.to(torch.float32).unsqueeze(0)

        return {
            f"{left_agent.uid}-0": left_tensor,
            f"{right_agent.uid}-1": right_tensor,
        }

    # ---------------------------------------------------------------------
    # Planner setup
    # ---------------------------------------------------------------------
    def setup_planner(self):
        """Instantiate an *mplib* planner for the A1 arm with minimal collision checking."""
        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]

        # Try different approaches to work around the broken mesh files
        planner = None
        error_msg = ""

        print("🔧 Attempting to work around A1 Galaxea broken mesh files...")

        # Approach 1: Try completely disabling collision checking
        try:
            print("   Trying approach 1: Disable collision checking...")
            planner = mplib.Planner(
                urdf=self.env_agent.urdf_path,
                srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                user_link_names=link_names,
                user_joint_names=joint_names,
                move_group="tcp",
                joint_vel_limits=np.ones(6) * self.joint_vel_limits,
                joint_acc_limits=np.ones(6) * self.joint_acc_limits,
                convex=False,
                use_collision=False,  # Disable collision checking completely
            )
            print("   ✅ Successfully created planner with collision checking disabled")
        except Exception as e:
            error_msg += f"Approach 1 (use_collision=False): {e}\n"
            print(f"   ❌ Failed: {e}")

        # Approach 2: Try with point clouds instead of meshes
        if planner is None:
            try:
                print("   Trying approach 2: Point cloud collision...")
                planner = mplib.Planner(
                    urdf=self.env_agent.urdf_path,
                    srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
                    user_link_names=link_names,
                    user_joint_names=joint_names,
                    move_group="tcp",
                    joint_vel_limits=np.ones(6) * self.joint_vel_limits,
                    joint_acc_limits=np.ones(6) * self.joint_acc_limits,
                    convex=False,
                    use_point_cloud=True,  # Use point clouds instead of meshes
                )
                print("   ✅ Successfully created planner with point cloud collision")
            except Exception as e:
                error_msg += f"Approach 2 (use_point_cloud=True): {e}\n"
                print(f"   ❌ Failed: {e}")

        if planner is None:
            print(
                "\n❌ All approaches failed. The A1 Galaxea robot has irreparably broken mesh files."
            )
            print(f"Errors encountered:\n{error_msg}")
            print(
                "💡 **RECOMMENDATION:** Use the Panda robot instead, which works perfectly:"
            )
            print(
                "   python mani_skill/examples/motionplanning/panda/run.py -e PickCube-v1 --save-video"
            )
            print(
                "   python mani_skill/examples/motionplanning/panda/run.py -e PickBox-v1 --save-video"
            )
            raise RuntimeError(
                "A1 Galaxea robot mesh files are too damaged for motion planning. Use Panda robot instead."
            )

        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    # ------------------------------------------------------------------
    # Convenience gripper helpers
    # ------------------------------------------------------------------
    def open_gripper(self):
        """Command the gripper to the *open* configuration."""
        print(
            f"🔍 [A1 DEBUG] open_gripper() called - setting gripper_state from {self.gripper_state} to {self.OPEN}"
        )
        self.gripper_state = self.OPEN
        qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
        print(f"🔍 [A1 DEBUG] open_gripper() - current arm qpos: {qpos}")
        for step in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:  # pd_joint_pos_vel, etc.
                action = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])
            print(f"🔍 [A1 DEBUG] open_gripper() step {step}: action={action}")
            if self.is_bimanual:
                action = self._format_action_for_bimanual(action)
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Check actual gripper state after step
            actual_qpos = self.robot.get_qpos()[0].cpu().numpy()
            actual_gripper = actual_qpos[6:8]  # Both gripper joints
            print(
                f"🔍 [A1 DEBUG] open_gripper() step {step}: ACTUAL gripper joints={actual_gripper}"
            )

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    def close_gripper(self, t: int = 6, gripper_state: float | None = None):
        """Command the gripper to the *closed* configuration.

        Args:
            t: Number of control cycles to hold the command.
            gripper_state: Desired prismatic joint value. Defaults to
                :pyattr:`CLOSED`.
        """
        old_gripper_state = self.gripper_state
        self.gripper_state = self.CLOSED if gripper_state is None else gripper_state
        print(
            f"🔍 [A1 DEBUG] close_gripper() called - setting gripper_state from {old_gripper_state} to {self.gripper_state}"
        )
        qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
        print(f"🔍 [A1 DEBUG] close_gripper() - current arm qpos: {qpos}")
        for step in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])
            print(f"🔍 [A1 DEBUG] close_gripper() step {step}: action={action}")
            if self.is_bimanual:
                action = self._format_action_for_bimanual(action)
            obs, reward, terminated, truncated, info = self.env.step(action)

            # Check actual gripper state after step
            actual_qpos = self.robot.get_qpos()[0].cpu().numpy()
            actual_gripper = actual_qpos[6:8]  # Both gripper joints
            print(
                f"🔍 [A1 DEBUG] close_gripper() step {step}: ACTUAL gripper joints={actual_gripper}"
            )

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Debug trajectory execution
    # ------------------------------------------------------------------

    def move_to_pose_with_screw(
        self, pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Override move_to_pose_with_screw to debug gripper state."""
        print(
            f"🔍 [A1 DEBUG] move_to_pose_with_screw() called with gripper_state: {self.gripper_state}"
        )
        return super().move_to_pose_with_screw(pose, dry_run, refine_steps)

    def move_to_pose_with_RRTConnect(
        self, pose, dry_run: bool = False, refine_steps: int = 0
    ):
        """Override move_to_pose_with_RRTConnect to debug gripper state."""
        print(
            f"🔍 [A1 DEBUG] move_to_pose_with_RRTConnect() called with gripper_state: {self.gripper_state}"
        )
        return super().move_to_pose_with_RRTConnect(pose, dry_run, refine_steps)

    def follow_path(self, result, refine_steps: int = 0):
        """Override follow_path to handle bimanual action format and debug gripper values."""
        n_step = result["position"].shape[0]

        # Debug: Print trajectory information
        print(f"🔍 [A1 DEBUG] Following path with {n_step} steps")
        print(f"🔍 [A1 DEBUG] Current gripper_state: {self.gripper_state}")

        for i in range(n_step + refine_steps):
            qpos = result["position"][min(i, n_step - 1)]

            if self.control_mode == "pd_joint_pos_vel":
                qvel = result["velocity"][min(i, n_step - 1)]
                action = np.hstack([qpos, qvel, self.gripper_state])
            else:
                action = np.hstack([qpos, self.gripper_state])

            # Format action for bimanual mode if needed
            if self.is_bimanual:
                action = self._format_action_for_bimanual(action)
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.elapsed_steps += 1
            if self.print_env_info:
                print(
                    f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}"
                )
            if self.vis:
                self.base_env.render_human()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _ensure_convex_meshes(self) -> None:
        """Create `<mesh>.convex.stl` copies if mplib requires them.

        mplib's FCL loader appends ``.convex.stl`` to every mesh filename when
        ``convex=True`` (its default).  If those files are missing the planner
        raises a *ValueError*.  We sidestep that by duplicating each original
        mesh on-the-fly the first time we initialise the planner.
        """
        urdf_dir = os.path.dirname(self.env_agent.urdf_path)
        mesh_globs = [
            os.path.join(urdf_dir, "meshes", "*.STL"),
            os.path.join(urdf_dir, "meshes", "*.stl"),
        ]
        for pattern in mesh_globs:
            for mesh_path in glob.glob(pattern):
                convex_path = mesh_path + ".convex.stl"
                if not os.path.exists(convex_path):
                    try:
                        shutil.copy(mesh_path, convex_path)
                    except OSError as exc:  # permissions or other IO issues
                        print(
                            f"[warn] Could not create convex copy for {mesh_path}: {exc}"
                        )
