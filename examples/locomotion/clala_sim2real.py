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
from clala_env import ClalaEnv
from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

cmd_vel = Twist()
joint = JointState()
imu = Imu()
home_imu = Imu()
imu_first = True
imu_ready = False

def imuCb(msg):
    global imu, home_imu, imu_ready, imu_first
    if imu_first:
        home_imu = msg
        imu_first = False
    else:
        imu = msg
        imu_ready = True

def twistCb(msg):
    global cmd_vel
    cmd_vel = msg

def jointCb(msg):
    global joint
    joint = msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="clala-walking")
    parser.add_argument("--ckpt", type=int, default=500)
    args = parser.parse_args()

    rospy.init_node('genesis_eval_node')
    rospy.Subscriber("/wit/imu", Imu, imuCb)
    rospy.Subscriber("/cmd_vel", Twist, twistCb)
    rospy.Subscriber("/joint_states", JointState, jointCb)
    pub = rospy.Publisher("/genesis_angles", Float32MultiArray, queue_size=1)
    pub2 = rospy.Publisher("/genesis_imu", Imu, queue_size=1)

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env_cfg["termination_if_roll_greater_than"] = 60
    env_cfg["termination_if_pitch_greater_than"] = 60

    env = ClalaEnv(
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
    angle_msg = Float32MultiArray()
    projected_gravity = torch.zeros((1, 3), device="cpu", dtype=gs.tc_float)
    with torch.no_grad():
        while not rospy.is_shutdown():
            ## make obs
            if imu_ready:
                home_quat = torch.tensor([[home_imu.orientation.x, home_imu.orientation.y, home_imu.orientation.z, home_imu.orientation.w]])
                print("home_quat", home_quat)
                current_quat = torch.tensor([[imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]])
                print("current_quat", current_quat)
                diff_quat = transform_quat_by_quat(inv_quat(home_quat), current_quat)
                print("diff_quat", diff_quat)
                inv_diff_quat = inv_quat(diff_quat)
                projected_gravity = transform_by_quat(env.global_gravity, inv_diff_quat)
                # obs[0, 0:3] = torch.tensor([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]) * env.obs_scales["ang_vel"]
                # obs[0, 3:6] = projected_gravity
            obs[0, 6:9] = torch.tensor([cmd_vel.linear.x * 2.0, cmd_vel.linear.y * 2.0, cmd_vel.angular.z * 4.0]) * env.commands_scale
            if len(joint.position) == 10:
                # for j in joint.position:
                #     print(j, end=' ')
                # print('')
                # obs[0, 9:19] = (torch.tensor(joint.position) - env.default_dof_pos) * env.obs_scales["dof_pos"]
                # obs[0, 19:29] = torch.tensor(joint.velocity) * env.obs_scales["dof_vel"]
                print((torch.tensor(joint.position) - env.default_dof_pos) * env.obs_scales["dof_pos"])
            ## make actions
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)
            ## make publish data
            angle_msg.data.clear()
            for i in range(10):
                angle_msg.data.append(env.target_dof_pos[0, i].item())
            pub.publish(angle_msg)
            # print("env.base_quat ", env.base_quat)
            # print("rpy ", quat_to_xyz(env.base_quat))

            # gimu = Imu()
            # gimu.header.stamp = rospy.Time.now()
            # gimu.orientation.x = env.base_quat[0, 1]
            # gimu.orientation.y = env.base_quat[0, 2]
            # gimu.orientation.z = env.base_quat[0, 3]
            # gimu.orientation.w = env.base_quat[0, 0]
            # gimu.angular_velocity.x = env.base_ang_vel[0, 0]
            # gimu.angular_velocity.y = env.base_ang_vel[0, 1]
            # gimu.angular_velocity.z = env.base_ang_vel[0, 2]
            # pub2.publish(gimu)

    # original loop
    # obs, _ = env.reset()
    # with torch.no_grad():
    #     while True:
    #         actions = policy(obs)
    #         obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""
