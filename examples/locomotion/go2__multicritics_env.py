'''
rsl_rl version3.1.1用のgo2_train
'''

import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat, quat_to_R
from tensordict import TensorDict
from rsl_rl.utils.barrier import relaxed_barrier_for_interval

import numpy as np



def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda"):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = obs_cfg["num_pri_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.elapsed_time = 0.0
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]
        self.barrier_rew_parameters = reward_cfg["barrier_reward_parameters"]
        self.T = self.env_cfg["cycle"]
        # self.d_lower_gait = self.env_cfg["d_lower_gait"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        # self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add terrain
        hf = np.zeros((40, 40), dtype=np.int16)
        hf[10:30, 10:30] = 200 * np.hanning(20)[:, None] * np.hanning(20)[None, :]
        self.hf_tensor = torch.from_numpy(hf).clone()
        self.hf_tensor = self.hf_tensor.to(self.device)

        self.horizontal_scale = 0.25  # metres between grid points
        self.vertical_scale   = 0.005  # metres per height-field unit
        pos = (-3.0, -3.0, 0.0)
        self.terrain_pos = torch.tensor([pos], device=self.device)
        self.terrain = self.scene.add_entity(
            morph=gs.morphs.Terrain(
                pos = pos,
                height_field=hf,
                horizontal_scale=self.horizontal_scale,
                vertical_scale=self.vertical_scale,
            ),
        )

        self.hf = self.terrain.geoms[0].metadata["height_field"]

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]
        self.foot_idxs = []
        # search foot geom
        for name in self.env_cfg["calf_names"]:
            geoms = self.robot.get_link(name).geoms
            for g in geoms:
                if g.get_pos()[0, 2] < 0.01:
                    self.foot_idxs.append(g.idx)
        print("self.foot_idxs", self.foot_idxs)

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        self.barrier_rew_functions, self.barrier_epi_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        for name in self.barrier_rew_parameters.keys():
            self.barrier_rew_parameters[name][0] *= self.dt
            self.barrier_rew_functions[name] = getattr(self, "_barrier_reward_" + name)
            self.barrier_epi_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # gait function
        phase = torch.tensor([[1, -1, -1, 1]], device=self.device, dtype=gs.tc_float)
        self.phase_tensor = phase.repeat(self.num_envs, 1)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.privileged_obs_buf = torch.zeros((self.num_envs, self.num_privileged_obs), device=self.device, dtype=gs.tc_float)
        self.observations = TensorDict({"policy": self.obs_buf, "privileged": self.privileged_obs_buf})
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.barrier_rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last2_base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.last_base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )
        self.last_foot_pos = torch.zeros((self.num_envs, 4, 3), device=self.device, dtype=gs.tc_float)
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        self.target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(self.target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)
        self.dof_force[:] = self.robot.get_dofs_force(self.motor_dofs)

        #接地検出
        contacts_info = self.robot.get_contacts()
        # print("contacts_info", contacts_info)
        contacts = []
        for foot in self.foot_idxs:
            mask = (contacts_info['geom_a'] == foot)
            # 非ゼロの位置を取得
            row_indices, col_indices = mask.nonzero(as_tuple=True)
            # print("row_indices", row_indices)
            # print("col_indices", col_indices)
            # output用tensorを用意
            result = torch.full((contacts_info['geom_a'].size(0),), -1, dtype=torch.long, device=self.device)
            # 該当要素に書き込み
            result[row_indices] = col_indices[range(len(row_indices))]
            contacts.append(result)
        self.foot_contact = torch.stack(contacts, dim = 1)
        self.foot_contact_float = torch.zeros_like(self.foot_contact, device=self.device, dtype=gs.tc_float)
        self.foot_contact_float = torch.where(self.foot_contact > -1, 1.0, -1.0)
        self.foot_contact_float = torch.flatten(self.foot_contact_float, start_dim=1)

        _p = []
        for g_idx in self.foot_idxs:
            for g in self.robot.geoms:
                if g.idx == g_idx:
                    _p.append(g.get_pos())
        self.foot_pos = torch.stack(_p, dim = 1)
        # print("foot_pos", self.foot_pos[0]) # world coordinate

        # print("robot pos:", self.robot.get_pos()[0])
        # print("robot quat: ", self.robot.get_quat()[0])
        # print("robot euler:", self.base_euler[0])

        # height of foot position
        footpos_height = self.terrain_height_from_tensor()
        # print("height of foot pos ", footpos_height[0])
        # cyclic function
        self.cycle_a, self.cycle_b = (np.sin(2*np.pi*self.elapsed_time/self.T), np.cos(2*np.pi*self.elapsed_time/self.T))
        cyclic_func = torch.tensor([self.cycle_a, self.cycle_b], device=self.device, dtype=gs.tc_float)
        self.cyclic_func_tensor = cyclic_func.repeat(self.num_envs, 1)
        # print("cyclic_func", self.cyclic_func_tensor.shape, self.cyclic_func_tensor)

        # compute foot_vel
        self.foot_vel = (self.foot_pos - self.last_foot_pos) / self.dt

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        self.barrier_rew_buf[:] = 0.0
        for name, barrier_rew_func in self.barrier_rew_functions.items():
            scale = self.barrier_rew_parameters[name][0]
            lower = self.barrier_rew_parameters[name][1]
            upper = self.barrier_rew_parameters[name][2]
            delta = self.barrier_rew_parameters[name][3]
            sum = self.barrier_rew_parameters[name][4]
            rew = barrier_rew_func()
            rew = relaxed_barrier_for_interval(rew, lower=lower, upper=upper, delta_frac=delta) * scale
            if sum > 0:
                rew = torch.sum(rew, dim = 1)
            self.barrier_rew_buf += rew
            self.barrier_epi_sums[name] += rew


        # compute observations
        base_pos_exp = torch.unsqueeze(self.base_pos, 1)
        relative_foot_pos = (self.foot_pos - base_pos_exp)
        relative_foot_pos = torch.flatten(relative_foot_pos, start_dim=1)
        self.obs_buf = torch.cat(
            [
                # self.projected_gravity,  # 3
                self.base_euler * self.obs_scales["ori_vel"], # body orientation 3 euler角を入れる
                self.base_ang_vel * self.obs_scales["ang_vel"],  # body angular velocity 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # joint positions 12
                self.dof_vel * self.obs_scales["dof_vel"],  # joint velocities 12
                (self.target_dof_pos - self.dof_pos) * self.obs_scales["pos_err"], # history of joint position errors 12
                # history of joint velocities 12 dof_velと同じになるので控える
                relative_foot_pos * self.obs_scales["foot_pos"], # relative foot positions in the body frame 12 COMからの距離
                self.actions,  # previous actions 12
                self.commands * self.commands_scale,  # commanded velocity 3
                self.cyclic_func_tensor # cyclic functions 2
                # stand-mode indicator 1
            ],
            axis=-1,
        )
        self.privileged_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["base_lin_vel"],  # body's linear velocity 3
                self.foot_contact_float, # foot contact state 4
                footpos_height # terrain information around the feet 4
            ],
            axis=-1,
        )
        self.observations = TensorDict({"policy": self.obs_buf, "privileged": self.privileged_obs_buf})

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.elapsed_time += self.dt
        self.last_foot_pos[:] = self.foot_pos[:]
        self.last_base_quat[:] = self.base_quat[:]
        self.last2_base_pos[:] = self.last_base_pos[:]
        self.last_base_pos[:] = self.base_pos[:]

        return self.observations, self.rew_buf, self.barrier_rew_buf, self.reset_buf, self.extras

    # 地表高さを取得する関数（テンソル版）（未デバッグ）
    def terrain_height_from_tensor(self):
        """
        foot_pos: ワールド座標 (tensor)
        height_field: torch.tensor shape (H, W)
        horizontal_scale, vertical_scale: float
        terrain_pos: (x0, y0, z0) ワールド原点オフセット
        """
        H, W = self.hf_tensor.shape

        # ローカル座標系に変換
        lpos = (self.foot_pos - self.terrain_pos) / self.horizontal_scale

        # floor / frac
        ipos = torch.floor(lpos).long()
        tpos = lpos - ipos.float()

        # 境界をクランプ
        ipos = torch.clamp(ipos, torch.tensor([0, 0, 0], device=self.device), torch.tensor([H - 2, W - 2, 100], device=self.device))

        # 4近傍を取得（双線形補間）
        h00 = self.hf_tensor[ipos[:, :, 0], ipos[:, :, 1]]
        h10 = self.hf_tensor[ipos[:, :, 0] + 1, ipos[:, :, 1]]
        h01 = self.hf_tensor[ipos[:, :, 0], ipos[:, :, 1] + 1]
        h11 = self.hf_tensor[ipos[:, :, 0] + 1, ipos[:, :, 1] + 1]

        # 双線形補間（ベクトル演算）
        h_interp = (
            h00 * (1 - tpos[:, :, 0]) * (1 - tpos[:, :, 2])
            + h10 * tpos[:, :, 0] * (1 - tpos[:, :, 2])
            + h01 * (1 - tpos[:, :, 0]) * tpos[:, :, 2]
            + h11 * tpos[:, :, 2] * tpos[:, :, 2]
        )

        # スケール＋高さオフセット
        footpos_height = h_interp * self.vertical_scale + self.terrain_pos[:,2]
        return footpos_height

    def get_observations(self):
        # return self.obs_buf
        return self.observations

    # def get_privileged_observations(self):
    #     return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self.extras["barrier_episode"] = {}
        for key in self.barrier_epi_sums.keys():
            self.extras["barrier_episode"]["rew_" + key] = (
                torch.mean(self.barrier_epi_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.barrier_epi_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.observations, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_foot_slip(self):
        # Penalize foot slip
        # 接地している足のワールド速度を見る。（スリップしていなければゼロのはず）
        result = torch.sum(self.foot_vel[:,:,0][self.foot_contact > -1]**2 + self.foot_vel[:,:,0][self.foot_contact > -1]**2)
        return result

    def _reward_action_smoothness1(self):
        # Penalize action smoothness 1st-oder
        return torch.square(torch.norm(self.base_pos - self.last_base_pos, dim=1))

    def _reward_action_smoothness2(self):
        # Penalize action smoothness 2nd-oder
        return torch.square(torch.norm(self.base_pos - 2 * self.last_base_pos + self.last2_base_pos, dim=1))

    def _reward_orientation_deviation(self):
        # Penalize orientation deviation
        base_R = quat_to_R(self.base_quat)
        return torch.sum(torch.square(torch.acos(base_R[:,2,2])))

    def _reward_joint_position_regularization(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_joint_velocity_regularization(self):
        return torch.sum(torch.square(self.dof_vel))

    def _reward_joint_acceleration_regularization(self):
        return torch.sum(torch.square(self.dof_vel - self.last_dof_vel))

    def _reward_torque_regularization(self):
        return torch.sum(torch.square(self.dof_force))

    def _reward_base_motion_regulation(self):
        # quaternionの時間微分を求める
        d_base_quat = (self.base_quat - self.last_base_quat) / self.dt  # tensor(num_envs, 4)
        d_base_quat = d_base_quat.unsqueeze(2)  # tensor(num_envs, 4, 1)
        q0 = self.base_quat[:,0]
        q1 = self.base_quat[:,1]
        q2 = self.base_quat[:,2]
        q3 = self.base_quat[:,3]
        row1 = torch.stack([-q1, q0, q3, -q2], dim=1)
        row2 = torch.stack([-q2, -q3, q0, q1], dim=1)
        row3 = torch.stack([-q3, q2, -q1, q0], dim=1)
        mat_e = torch.stack([row1, row2, row3], dim=1)
        omega = 2 * torch.bmm(mat_e, d_base_quat)
        return torch.sum(0.4 * torch.square(self.base_lin_vel[:, 2]) + 0.2 * torch.abs(omega[:, 0]) + 0.2 * torch.abs(omega[:, 1]))

    # def _reward_body_contact(self):
    #     return torch.sum(self.body_contact)

    # def _reward_body_com_offset(self):
    #     return torch.sum(torch.square(self.com_offset[:, :2]) * self.stand)

    def _barrier_reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _barrier_reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _barrier_reward_gait_timing(self):
        gait_bool = self.phase_tensor * self.foot_contact/torch.abs(self.foot_contact) * self.cycle_a < self.barrier_rew_parameters["gait_timing"][1]
        gait_penalty = torch.where(gait_bool, self.cycle_a, 0.0)
        return gait_penalty