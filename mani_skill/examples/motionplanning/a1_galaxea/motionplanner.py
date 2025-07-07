import glob
import os
import shutil

import mplib
import numpy as np
import sapien.core as sapien

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

    # Prismatic finger joint value (metres) for open/closed states
    OPEN: float = 0.03
    CLOSED: float = 0.0

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
        super().__init__(
            env,
            debug,
            vis,
            base_pose,
            visualize_target_grasp_pose,
            print_env_info,
            joint_vel_limits,
            joint_acc_limits,
        )
        self.gripper_state = self.OPEN

    # ---------------------------------------------------------------------
    # Planner setup
    # ---------------------------------------------------------------------
    def setup_planner(self):
        """Instantiate an *mplib* planner for the A1 arm."""
        # Ensure convex mesh copies exist; mplib looks for \*.convex.stl files
        self._ensure_convex_meshes()

        link_names = [link.get_name() for link in self.robot.get_links()]
        joint_names = [joint.get_name() for joint in self.robot.get_active_joints()]
        planner = mplib.Planner(
            urdf=self.env_agent.urdf_path,
            srdf=self.env_agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group="tcp",  # defined in the URDF as fixed joint "tcp_joint"
            joint_vel_limits=np.ones(6) * self.joint_vel_limits,
            joint_acc_limits=np.ones(6) * self.joint_acc_limits,
        )
        planner.set_base_pose(np.hstack([self.base_pose.p, self.base_pose.q]))
        return planner

    # ------------------------------------------------------------------
    # Convenience gripper helpers
    # ------------------------------------------------------------------
    def open_gripper(self):
        """Command the gripper to the *open* configuration."""
        self.gripper_state = self.OPEN
        qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
        for _ in range(6):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:  # pd_joint_pos_vel, etc.
                action = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])
            obs, reward, terminated, truncated, info = self.env.step(action)
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
        self.gripper_state = self.CLOSED if gripper_state is None else gripper_state
        qpos = self.robot.get_qpos()[0, :6].cpu().numpy()
        for _ in range(t):
            if self.control_mode == "pd_joint_pos":
                action = np.hstack([qpos, self.gripper_state])
            else:
                action = np.hstack([qpos, np.zeros_like(qpos), self.gripper_state])
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
