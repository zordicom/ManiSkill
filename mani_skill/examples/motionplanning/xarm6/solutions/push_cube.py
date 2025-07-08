import numpy as np
import sapien

from mani_skill.envs.tasks import PushCubeEnv
from mani_skill.examples.motionplanning.xarm6.motionplanner import (
    XArm6PandaGripperMotionPlanningSolver,
    XArm6RobotiqMotionPlanningSolver,
)


def solve(env: PushCubeEnv, seed=None, debug=False, vis=False):
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

    env = env.unwrapped

    # -------------------------------------------------------------------------- #
    # Close gripper and reach behind the cube
    # -------------------------------------------------------------------------- #
    planner.close_gripper()

    # Get cube and goal positions
    cube_pos = env.obj.pose.sp.p
    goal_pos = env.goal_region.pose.sp.p

    # Position behind the cube, at a good height for pushing
    push_height = cube_pos[2] + 0.01  # Slightly above cube center
    reach_pose = sapien.Pose(
        p=np.array([cube_pos[0] - 0.06, cube_pos[1], push_height]),
        q=env.agent.tcp.pose.sp.q,
    )
    planner.move_to_pose_with_RRTStar(reach_pose)

    # -------------------------------------------------------------------------- #
    # Push cube to goal by moving to position behind goal region
    # -------------------------------------------------------------------------- #
    # Calculate push target: position behind the goal region to push the cube into it
    push_target = np.array([goal_pos[0] - 0.08, goal_pos[1], push_height])
    goal_pose = sapien.Pose(p=push_target, q=env.agent.tcp.pose.sp.q)
    res = planner.move_to_pose_with_RRTStar(goal_pose)

    planner.close()
    return res
