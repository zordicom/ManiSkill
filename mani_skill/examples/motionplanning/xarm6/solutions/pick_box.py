import numpy as np
import sapien

from mani_skill.envs.tasks.tabletop.pick_box import PickBoxEnv
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6PandaGripperMotionPlanningSolver,
    XArm6RobotiqMotionPlanningSolver,
)


def solve(env: PickBoxEnv, seed=None, debug=False, vis=False):
    env.reset(seed=seed)

    if env.unwrapped.robot_uids == "xarm6_robotiq":
        planner_cls = XArm6RobotiqMotionPlanningSolver
    elif env.unwrapped.robot_uids == "xarm6_pandagripper":
        planner_cls = XArm6PandaGripperMotionPlanningSolver
    else:
        raise ValueError(f"Unsupported robot uid: {env.robot_uid}")
    planner = planner_cls(
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
    obb = get_actor_obb(env.b5box)

    approaching = np.array([0, 0, -1])
    # get transformation matrix of the tcp pose, is default batched and on torch
    target_closing = (
        env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    )
    # we can build a simple grasp pose using this information
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    closing, center = grasp_info["closing"], grasp_info["center"]
    grasp_pose = env.agent.build_grasp_pose(approaching, closing, env.b5box.pose.sp.p)

    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    planner.move_to_pose_with_RRTStar(reach_pose)

    # -------------------------------------------------------------------------- #
    # Grasp
    # -------------------------------------------------------------------------- #
    planner.move_to_pose_with_screw(grasp_pose)
    planner.close_gripper()

    # -------------------------------------------------------------------------- #
    # Move to basket - three stage approach: lift → move → drop
    # -------------------------------------------------------------------------- #
    # Stage 1: Lift the box 15cm straight up from current position
    current_pos = env.agent.tcp.pose.p.cpu().numpy().flatten()
    lift_pose = sapien.Pose(current_pos + np.array([0, 0, 0.15]), grasp_pose.q)
    planner.move_to_pose_with_RRTStar(lift_pose)

    # Stage 2: Move horizontally to above the basket (maintaining 15cm height)
    basket_pose = env.basket.pose
    basket_pos = (
        basket_pose.p.cpu().numpy().flatten()
        if hasattr(basket_pose.p, "cpu")
        else basket_pose.p
    )
    above_basket_pose = sapien.Pose(basket_pos + np.array([0, 0, 0.15]), grasp_pose.q)
    planner.move_to_pose_with_RRTStar(above_basket_pose)

    # Stage 3: Drop down into the basket and release
    place_pose = sapien.Pose(basket_pos + np.array([0, 0, 0.05]), grasp_pose.q)
    res = planner.move_to_pose_with_RRTStar(place_pose)
    planner.open_gripper()

    planner.close()
    return res
