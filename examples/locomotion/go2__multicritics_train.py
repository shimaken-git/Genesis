'''
rsl_rl version3.1.1用のgo2_train
'''

import argparse
import os
import pickle
import shutil

from go2__multicritics_env import Go2Env
from rsl_rl.runners import OnPolicyRunner
# from rsl_rl.algorithms.ppo import PPO
# from rsl_rl.modules.actor_critic import ActorCritic

import genesis as gs


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO_MULTI",
            "learning_rate": 0.001,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            "normalize_advantage_per_mini_batch": False,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCriticMultiple",
            "activation": "elu",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "critic2_hidden_dims": [512, 256, 128],
            "init_noise_std": 2.0,
            "noise_std_type": "scalar",
        },
        # "runner": {
        #     "checkpoint": -1,
        #     "load_run": -1,
        #     "log_interval": 1,
        #     "record_interval": -1,
        #     "resume": False,
        #     "resume_path": None,
        #     "runner_class_name": "runner_class_name",
        # },
        "class_name": OnPolicyRunner,
        "num_steps_per_env": 24,
        "max_iterations": max_iterations,
        "seed": 1,
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy", "privileged"],
        },
        "reward_groups": {
            "normal": ["normal"],
            "barrier": ["barrier"],
        },
        "save_interval": 100,
        "experiment_name": exp_name,
        "run_name": "",
        "logger": "tensorboard",
        # "neptune_project": "",
        # "wandb_project": "",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "calf_names": [
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        # "num_obs": 66, # 3+3+12+12+12+12+4*3 = 66
        # standard observation
        # body orientation, body angular velocity, joint positions and velocities, history of joint position errors and joint velocities,
        # relative foot positions in the body frame, previous actions, commanded velocity, cyclic functions, and a stand-mode indicator
        "num_pri_obs": 3,
        # "num_pri_obs": 3, # 3+4+
        # privileged observation
        # body's linear velocity, foot contact state, and terrain information around the feet
        # critic にはstandard observationとprivileged observationが入力される。
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.2,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            # "foot_slip": 0.1,
            # "action_smoothness1": 0.1,
            # "action_smoothness2": 0.1,
            # "orientation_deviation": 0.1,
            # "joint_position_regularization": 0.1,
            # "joint_velocity_regularization": 0.1,
            # "joint_acceleration_regularization": 0.1,
            # "torque_regularization": 0.1,
            # "base_motion_regulation": 0.1,
            # "body_contact": 0.1,
            # "body_com_offset": 0.1,

            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,

        },
        "barrier_reward_parameters": {   # [scale, lower, upper, delta]
            "tracking_lin_vel": [1.0, -0.1, 0.1, 0.1],
            "tracking_ang_vel": [0.2, -0.1, 0.1, 0.1],
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 0.5],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.7, 1.7],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-multicritics-walking")
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

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg, show_viewer=True
    )

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )    # rsl_rl ver 3.0.0以降では、OnPolicyRunnerでtrain_cfgの内容を変更（popしてしまう）するのでここでファイルアウトする

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
