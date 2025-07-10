import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from copy import deepcopy

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from mani_skill.examples.motionplanning.a1_galaxea.solutions import (
    solvePickBox,
    solvePickCube,
)
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.utils.wrappers.record import RecordEpisode

MP_SOLUTIONS = {
    "PickCube-v1": solvePickCube,
    "PickBox-v1": solvePickBox,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Motion-planning demo for A1 Galaxea")
    parser.add_argument(
        "-e",
        "--env-id",
        type=str,
        default="PickCube-v1",
        choices=list(MP_SOLUTIONS.keys()),
        help="Environment to run the motion-planning solver on.",
    )
    parser.add_argument(
        "-o",
        "--obs-mode",
        type=str,
        default="none",
        help="Observation mode (usually 'none' for planning demos).",
    )
    parser.add_argument(
        "-n",
        "--num-traj",
        type=int,
        default=10,
        help="Number of trajectories to generate.",
    )
    parser.add_argument(
        "--reward-mode",
        type=str,
        help="Reward mode override if the task supports multiple modes.",
    )
    parser.add_argument(
        "--sim-backend",
        type=str,
        default="auto",
        help="Simulation backend: 'auto', 'cpu', or 'gpu'.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="rgb_array",
        help="Sensor pack for video recording (sensors | rgb_array)",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Show a GUI while planning/executing.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save RGB-array videos alongside the trajectory files.",
    )
    parser.add_argument(
        "--traj-name",
        type=str,
        help="Custom trajectory basename (timestamp used by default).",
    )
    parser.add_argument(
        "--shader",
        default="default",
        type=str,
        help="Shader pack for rendering (default | rt | rt-fast)",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="demos",
        help="Directory to store trajectories/videos.",
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="CPU processes for parallel generation (CPU backend only).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helper for one process
# -----------------------------------------------------------------------------


def _main(args, proc_id: int = 0, start_seed: int = 0) -> str:
    # Use bimanual environment for A1 Galaxea where supported
    if args.env_id == "PickBox-v1":
        env_id = "PickBoxBimanual-v1"
        bimanual = True
    else:
        env_id = args.env_id
        bimanual = False

    # Create environment with appropriate parameters
    env_kwargs = {
        "robot_uids": "a1_galaxea",
        "obs_mode": args.obs_mode,
        "reward_mode": args.reward_mode,
        "control_mode": "pd_joint_pos",
        "render_mode": args.render_mode,
        "sensor_configs": dict(shader_pack=args.shader),
        "human_render_camera_configs": dict(shader_pack=args.shader),
        "viewer_camera_configs": dict(shader_pack=args.shader),
        "sim_backend": args.sim_backend,
    }

    # Only add bimanual parameter for environments that support it
    if bimanual:
        env_kwargs["bimanual"] = bimanual

    env = gym.make(env_id, **env_kwargs)

    # Use the same solver for both single-arm and bimanual versions
    solve_fn = MP_SOLUTIONS[args.env_id]

    if not args.traj_name:
        traj_basename = time.strftime("%Y%m%d_%H%M%S")
    else:
        traj_basename = args.traj_name
    if args.num_procs > 1:
        traj_basename = f"{traj_basename}.{proc_id}"

        # Use appropriate directory name for bimanual vs single-arm
    output_env_name = env_id if bimanual else args.env_id
    source_desc = f"Official A1 Galaxea motion-planning solution ({'Bimanual' if bimanual else 'Single-arm'}) (ManiSkill)"

    # Skip recording for bimanual mode due to dict action space compatibility issues
    if not bimanual:
        env = RecordEpisode(
            env,
            output_dir=osp.join(args.record_dir, output_env_name, "motionplanning"),
            trajectory_name=traj_basename,
            save_video=args.save_video,
            source_type="motionplanning",
            source_desc=source_desc,
            video_fps=30,
            record_reward=False,
            save_on_reset=False,
        )

    output_h5 = env._h5_file.filename if hasattr(env, "_h5_file") else None
    successes = []
    episode_lengths = []

    for i in tqdm(range(args.num_traj), desc=f"proc_id: {proc_id}"):
        res = solve_fn(env, seed=start_seed + i, debug=False, vis=args.vis)
        if res == -1:
            success = False
        else:
            success = res[-1]["success"].item()
            episode_lengths.append(res[-1]["elapsed_steps"].item())
        successes.append(success)

        # Only flush if recording is enabled (single-arm mode)
        if hasattr(env, "flush_trajectory"):
            env.flush_trajectory()
        if args.save_video and hasattr(env, "flush_video"):
            env.flush_video()

    print(
        f"Success-rate: {np.mean(successes):.2f}; "
        f"Avg-episode-len: {np.mean(episode_lengths) if episode_lengths else 0:.1f}"
    )
    env.close()
    return output_h5


# -----------------------------------------------------------------------------
# Entry-point
# -----------------------------------------------------------------------------


def main(args):
    if args.num_procs > 1 and args.num_traj >= args.num_procs:
        # parallel generation (CPU backend only)
        seeds = [i * (args.num_traj // args.num_procs) for i in range(args.num_procs)]
        pool = mp.Pool(args.num_procs)
        proc_args = [(deepcopy(args), idx, seeds[idx]) for idx in range(args.num_procs)]
        out_paths = pool.starmap(_main, proc_args)
        pool.close()

        merged = out_paths[0].rsplit(".", 1)[0] + ".h5"
        merge_trajectories(merged, out_paths)
        for p in out_paths:
            os.remove(p)
            json_p = p.replace(".h5", ".json")
            os.remove(json_p)
    else:
        _main(args)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main(parse_args())
