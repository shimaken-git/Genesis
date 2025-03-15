import argparse
import os
import pickle
import rospy
import rosparam
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from geometry_msgs.msg import Twist
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

cmd_vel = Twist()

def imuCb(msg):
    x = msg.orientation.x
    # print(msg.orientation)

def twistCb(msg):
    global cmd_vel
    cmd_vel = msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    rospy.init_node('genesis_eval_node')
    rospy.Subscriber("/wit/imu", Imu, imuCb)
    rospy.Subscriber("/cmd_vel", Twist, twistCb)
    pub = rospy.Publisher("/genesis_angles", Float32MultiArray, queue_size=1)

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    obs, _ = env.reset()
    with torch.no_grad():
        # while True:
        while not rospy.is_shutdown():
            angle_msg = Float32MultiArray()
            # lay = MultiArrayDimension()
            # lay.label = "angles"
            # lay.size = 12
            # lay.stride = 12 * 4
            # angle_msg.layout.dim.append(lay)
            # angle_msg.layout.data_offset = 0
            actions = policy(obs)
            for i in range(12):
                angle_msg.data.append(actions[0, i].item())
            pub.publish(angle_msg)
            obs, _, rews, dones, infos = env.step(actions)
            print(obs[0, 6:9])
            # obs[0,6] = cmd_vel.linear.x * 2.0
            # obs[0,7] = 0.0
            # obs[0,8] = cmd_vel.angular.z
            obs[0,6] = 0.8
            obs[0,7] = 0.0
            obs[0,8] = 0.0
            print(obs[0, 6:9])
            print(cmd_vel.linear, cmd_vel.angular)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
