import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_box import PickBoxEnv
from mani_skill.examples.motionplanning.a1_galaxea.motionplanner import (
    A1GalaxeaMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve(env: PickBoxEnv, seed=None, debug: bool = False, vis: bool = False):
    """Solve *PickBox-v1* for an **A1 Galaxea** arm via classic motion planning.

    The high-level strategy mirrors the official Panda/XArm6 examples:

    1. Move to a *pre-grasp* pose 5 cm above the computed grasp.
    2. Descend to the grasp pose and close the fingers.
    3. Lift the box 15 cm straight up to avoid collisions.
    4. Move horizontally to above the basket (maintaining height).
    5. Lower into the basket and release.

    Args:
        env: A ManiSkill ``PickBox-v1`` environment initialised with
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
    )

    # Gripper state is initialized to OPEN (-0.03) in the planner constructor
    print(
        f"Initial gripper state: {planner.gripper_state} (should be -0.03 for fully open)"
    )

    FINGER_LENGTH = 0.025  # 2.5 cm effective finger depth
    env_unwrapped = env.unwrapped

    # ------------------------------------------------------------------
    # 1) Compute grasp pose for the b5box
    # ------------------------------------------------------------------
    obb = get_actor_obb(env_unwrapped.b5box)
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
            approaching, closing, env.b5box.pose.sp.p
        )
    else:
        # Bimanual mode - use right arm (active agent, index 1)
        grasp_pose = env.unwrapped.agent.agents[1].build_grasp_pose(
            approaching, closing, env.b5box.pose.sp.p
        )

    # ------------------------------------------------------------------
    # 2) Reach pre-grasp (5 cm above) with gripper fully open
    # ------------------------------------------------------------------
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # ------------------------------------------------------------------
    # 3) Descend and grasp
    # ------------------------------------------------------------------
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()

    # ------------------------------------------------------------------
    # 4) Move to basket - three stage approach: lift → move → drop
    # ------------------------------------------------------------------
    # Stage 1: Lift the box 15cm straight up from current position (reduced for better reachability)
    # Get current TCP position - handle both single-arm and bimanual modes
    if hasattr(env.unwrapped.agent, "tcp"):
        # Single-arm mode
        current_pos = env.agent.tcp.pose.p.cpu().numpy().flatten()
    else:
        # Bimanual mode - use right arm (active agent, index 1)
        current_pos = env.unwrapped.agent.agents[1].tcp.pose.p.cpu().numpy().flatten()
    lift_pose = sapien.Pose(current_pos + np.array([0, 0, 0.15]), grasp_pose.q)
    result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if result == -1:
        print("Failed to lift the box")
        planner.close()
        return -1
    print(f"Lift pose: {lift_pose}")
    # Stage 2: Move horizontally to above the basket (maintaining same height as lift)
    basket_center = env.basket.pose.sp.p
    lifted_height = lift_pose.p[2]  # Use the same Z-height from the lift pose
    print(f"Lifted height: {lifted_height}")
    above_basket_pose = sapien.Pose(
        np.array([basket_center[0], basket_center[1], lifted_height]),
        grasp_pose.q,
    )
    print(f"Above basket pose: {above_basket_pose}")
    result = planner.move_to_pose_with_RRTConnect(above_basket_pose)
    if result == -1:
        print("Failed to move to above basket position")
        planner.close()
        return -1

    # Stage 3: Lower into the basket and release
    # Place the object just above the basket bottom (basket_center is already at basket bottom + some height)
    basket_drop_height = (
        basket_center[2] + 0.15
    )  # 8cm above basket center for safe release
    lower_pose = sapien.Pose(
        np.array([basket_center[0], basket_center[1], basket_drop_height]), grasp_pose.q
    )
    print(f"Lower pose: {lower_pose}")
    res = planner.move_to_pose_with_RRTConnect(lower_pose)
    # Release the object - ensure gripper is fully open (-0.03)
    planner.open_gripper()
    print(
        f"Release gripper state: {planner.gripper_state} (should be -0.03 for fully open)"
    )
    # input("Press Enter to continue...")

    planner.close()
    return res
