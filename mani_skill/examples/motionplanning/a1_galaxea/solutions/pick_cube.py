import numpy as np
import sapien

from mani_skill.envs.tasks import PickCubeEnv
from mani_skill.examples.motionplanning.a1_galaxea.motionplanner import (
    A1GalaxeaMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve(env: PickCubeEnv, seed=None, debug: bool = False, vis: bool = False):
    """Solve *PickCube-v1* for an **A1 Galaxea** arm via classic motion planning.

    The high-level strategy mirrors the official Panda/XArm6 examples:

    1. Move to a *pre-grasp* pose 5 cm above the computed grasp.
    2. Descend to the grasp pose and close the fingers.
    3. Lift and carry the cube to the green goal site.

    Args:
        env: A ManiSkill ``PickCube-v1`` environment initialised with
            ``robot_uids="a1_galaxea"``.
        seed: Optional environment seed.
        debug: Enable extra printouts and wait-for-key pauses.
        vis: Render a GUI while planning/executing.

    Returns:
        The last tuple returned by :py:meth:`env.step`, making it consistent
        with other official examples. If motion planning fails, *-1* is
        returned.
    """
    env.reset(seed=seed)

    # Support both single-arm and bimanual A1 Galaxea
    robot_uids = env.unwrapped.robot_uids
    if robot_uids != "a1_galaxea" and robot_uids != ("a1_galaxea", "a1_galaxea"):
        raise ValueError(
            f"This solver only supports 'a1_galaxea' (single-arm) or ('a1_galaxea', 'a1_galaxea') (bimanual), but got {robot_uids}."
        )

    # Tweak velocity/acceleration limits for more precise trajectories
    # Get robot pose - handle both single-arm and bimanual modes
    if hasattr(env.unwrapped.agent, "robot"):
        # Single-arm mode
        base_pose = env.unwrapped.agent.robot.pose
    else:
        # Bimanual mode - use right arm (active agent, index 1)
        base_pose = env.unwrapped.agent.agents[1].robot.pose

    planner = A1GalaxeaMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=base_pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
        joint_vel_limits=0.5,
        joint_acc_limits=0.5,
    )

    # Slightly increase finger depth so the cube sits deeper between the pads
    FINGER_LENGTH = 0.035  # 3.5 cm
    env_unwrapped = env.unwrapped

    # ------------------------------------------------------------------
    # 1) Compute grasp pose
    # ------------------------------------------------------------------
    obb = get_actor_obb(env_unwrapped.cube)
    approaching = np.array([0, 0, -1])  # approach from +Z world direction
    # Get TCP pose - handle both single-arm and bimanual modes
    if hasattr(env.unwrapped.agent, "tcp"):
        # Single-arm mode
        tcp_pose = env.agent.tcp.pose
    else:
        # Bimanual mode - use right arm (active agent, index 1)
        tcp_pose = env.unwrapped.agent.agents[1].tcp.pose

    target_closing = tcp_pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()

    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, _ = grasp_info["closing"], grasp_info["center"]
    # Build grasp pose - handle both single-arm and bimanual modes
    if hasattr(env.unwrapped.agent, "build_grasp_pose"):
        # Single-arm mode
        grasp_pose = env.agent.build_grasp_pose(
            approaching, closing, env.cube.pose.sp.p
        )
    else:
        # Bimanual mode - use right arm (active agent, index 1)
        grasp_pose = env.unwrapped.agent.agents[1].build_grasp_pose(
            approaching, closing, env.cube.pose.sp.p
        )

    # ------------------------------------------------------------------
    # 2) Reach pre-grasp (3 cm above)
    #    A1 has shorter reach; a smaller drop distance yields better IK
    # ------------------------------------------------------------------
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.03])
    planner.move_to_pose_with_screw(reach_pose)

    # ------------------------------------------------------------------
    # 3) Descend and grasp
    # ------------------------------------------------------------------
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # ------------------------------------------------------------------
    # 4) Transport to goal
    # ------------------------------------------------------------------
    goal_pose = sapien.Pose(env.goal_site.pose.sp.p, grasp_pose.q)
    res = planner.move_to_pose_with_screw(goal_pose)

    planner.close()
    return res
