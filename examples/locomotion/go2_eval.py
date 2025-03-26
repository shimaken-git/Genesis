import argparse
import os
import pickle
import rospy
import rosparam
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

cmd_vel = Twist()
joint = JointState()
imu = Imu()

def imuCb(msg):
    global imu
    imu = msg

def twistCb(msg):
    global cmd_vel
    cmd_vel = msg

def jointCb(msg):
    global joint
    joint = msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    rospy.init_node('genesis_eval_node')
    rospy.Subscriber("/realRobot/imu", Imu, imuCb)
    rospy.Subscriber("/cmd_vel", Twist, twistCb)
    rospy.Subscriber("/realRobot/joint_states", JointState, jointCb)
    pub = rospy.Publisher("/genesis_angles", Float32MultiArray, queue_size=1)

    gs.init(backend=gs.cpu)

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
        device="cpu"
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cpu")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cpu")

    obs, _ = env.reset()
    # for i in range(12):
    #     joint.position.append(0)
    #     joint.velocity.append(0)
    angle_msg = Float32MultiArray()
    projected_gravity = torch.zeros((1, 3), device="cpu", dtype=gs.tc_float)
    with torch.no_grad():
        while not rospy.is_shutdown():
            ## make obs
            inv_base_quat = inv_quat(torch.tensor([[imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z]]))
            projected_gravity = transform_by_quat(env.global_gravity, inv_base_quat)
            # obs[0, 0:3] = torch.tensor([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]) * env.obs_scales["ang_vel"]
            # obs[0, 3:6] = projected_gravity
            obs[0, 6:9] = torch.tensor([cmd_vel.linear.x * 2.0, 0.0, cmd_vel.angular.z * 1.5]) * env.commands_scale
            if len(joint.position) == 12:
                obs[0, 9:21] = (torch.tensor(joint.position) - env.default_dof_pos) * env.obs_scales["dof_pos"]
                obs[0, 21:33] = torch.tensor(joint.velocity) * env.obs_scales["dof_vel"]
            # obs[0, 33:45] = actions
            # print(obs)
            # commands
            # print(obs[0, 6:9])  #commands
            ## make actions
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            ## make publish data
            angle_msg.data.clear()
            for i in range(12):
                angle_msg.data.append(env.target_dof_pos[0, i].item())
            pub.publish(angle_msg)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
