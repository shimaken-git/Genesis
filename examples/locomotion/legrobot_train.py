import argparse
import os
import pickle
import shutil

from legrobot_env import LegRobotEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import xyz_to_quat
import numpy as np


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 3,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "knee_joint": 0.75,
            "ankle_y_joint": -0.0937,
            "ankle_x_joint": 0.0,
        },
        "max_joint_angles": {  # [rad]
            "knee_joint": 2.5,
            "ankle_y_joint": 1.0,
            "ankle_x_joint": 1.0,
        },
        "min_joint_angles": {  # [rad]
            "knee_joint": 0.0,
            "ankle_y_joint": -1.0,
            "ankle_x_joint": -1.0,
        },
        "dof_names": [
            "knee_joint",
            "ankle_y_joint",
            "ankle_x_joint",
        ],
        # PD
        "kp": [30.0, 20.0, 20.0],
        "kd": [0.5, 0.795, 0.795],
        # "damping": [0.1, 0.1, 0.1],
        # "armature": [0.00567, 0.001, 0.001],
        "damping": [1.0, 1.0, 1.0],
        "armature": [0.1, 0.1, 0.1],
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 60,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.6],
        # "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "base_init_quat": np.array(xyz_to_quat(np.array([0.0, -32.6, 0.0])), dtype='float32'),
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 18,    # dof_num
        "obs_scales": {
            "lin_vel": 2.0,    # 3
            "ang_vel": 0.25,   # 3
            "dof_pos": 1.0,    # dof_num
            "dof_vel": 0.05,   # dof_num
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.07, #-0.005,
            "similar_to_default": -0.1,
            "dof_pos_range": -1.0,
            "feet_air_time": 0.001,
            "lateral_moving": 1.0,
            "lateral_periodic": 5.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 0.5],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="legrobot-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=500)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = LegRobotEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
