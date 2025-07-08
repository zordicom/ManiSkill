import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_box import PickBoxEnv
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)


def solve(env: PickBoxEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)
    planner = PandaArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )

    FINGER_LENGTH = 0.025
    env = env.unwrapped

    # retrieves the object oriented bounding box (trimesh box object)
    obb = get_actor_obb(env.b5box)  # Use b5box instead of cube

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    # we can build a simple grasp pose using this information for Panda
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(
        approaching, closing, env.b5box.pose.sp.p
    )  # Use b5box

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTConnect(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_RRTConnect(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to basket - three stage approach: lift → move → drop
    # -------------------------------------------------------------------------- #
    # Stage 1: Lift the box 15cm straight up from current position
    current_pos = env.agent.tcp.pose.p.cpu().numpy().flatten()
    lift_pose = sapien.Pose(current_pos + np.array([0, 0, 0.15]), grasp_pose.q)
    planner.move_to_pose_with_RRTConnect(lift_pose)

    # Stage 2: Move horizontally to above the basket (maintaining 15cm height)
    basket_center = env.basket.pose.sp.p
    above_basket_pose = sapien.Pose(
        basket_center + np.array([0, 0, 0.15]), grasp_pose.q
    )
    planner.move_to_pose_with_RRTConnect(above_basket_pose)

    # Stage 3: Drop down into the basket and release
    place_pose = sapien.Pose(basket_center + np.array([0, 0, 0.05]), grasp_pose.q)
    res = planner.move_to_pose_with_RRTConnect(place_pose)
    planner.open_gripper()

    planner.close()
    return res
