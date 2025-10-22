import argparse
import os
import pickle
import shutil

from clala_stand_env import ClalaEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


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
        "num_actions": 10,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "L_calf_joint": 1.15,
            "L_hip2_joint": 0.0,
            "L_hip_joint": 0.0,
            "L_thigh_joint": -0.57,
            "L_toe_joint": -0.60,
            "R_calf_joint": 1.15,
            "R_hip2_joint": 0.0,
            "R_hip_joint": 0.0,
            "R_thigh_joint": -0.57,
            "R_toe_joint": -0.60,
        },
        "max_joint_angles": {  # [rad]
            "L_calf_joint": 2.5,
            "L_hip2_joint": 0.7,
            "L_hip_joint": 0.4,
            "L_thigh_joint": 1.5,
            "L_toe_joint": 1.5,
            "R_calf_joint": 2.5,
            "R_hip2_joint": 0.7,
            "R_hip_joint": 0.4,
            "R_thigh_joint": 1.5,
            "R_toe_joint": 1.5,
        },
        "min_joint_angles": {  # [rad]
            "L_calf_joint": -0.05,
            "L_hip2_joint": -0.7,
            "L_hip_joint": -0.4,
            "L_thigh_joint": -1.5,
            "L_toe_joint": -1.5,
            "R_calf_joint": -0.05,
            "R_hip2_joint": -0.7,
            "R_hip_joint": -0.4,
            "R_thigh_joint": -1.5,
            "R_toe_joint": -1.5,
        },
        "dof_names": [
            "L_calf_joint",
            "L_hip2_joint",
            "L_hip_joint",
            "L_thigh_joint",
            "L_toe_joint",
            "R_calf_joint",
            "R_hip2_joint",
            "R_hip_joint",
            "R_thigh_joint",
            "R_toe_joint",
        ],
        # PD
        "kp": [15.3, 9.1, 30.0, 29.0, 30.0, 15.3, 9.1, 30.0, 29.0, 30.0],
        # "kp": [40.0, 9.1, 10.0, 29.0, 10.0, 40.0, 9.1, 10.0, 29.0, 10.0],
        "kd": [0.934, 0.758, 0.5, 0.795, 0.5, 0.934, 0.758, 0.5, 0.795, 0.5],
        # "kp": [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        # "kd": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        # "damping": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "armature": [0.0173, 0.021, 0.001, 0.00567, 0.001, 0.0173, 0.021, 0.001, 0.00567, 0.001],
        "damping": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # "armature": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        # termination
        "termination_if_roll_greater_than": 20,  # degree
        "termination_if_pitch_greater_than": 20,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.35],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 39,    # dof_num * 3 + 9 かな？    下の26に加えて　dof_effort(dof_num) robot_angle(3) 合計39　って感じかな。
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
            # "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.07, #-0.005,
            "similar_to_default": -0.1,
            "dof_pos_range": -1.0,
            # "dof_motion": -1.0,
            # "thigh_up": 0.0,
            # "feet_air_time": 0.001,
            # "lateral_moving": 1.0,
            # "lateral_periodic": 5.0,
            # "hip_angle_st": -1.0,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "body_angle_x_range": [0.0, 0.0],
        "body_angle_y_range": [0.0, 0.0],
        "body_height_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="clala-stand")
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

    env = ClalaEnv(
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
