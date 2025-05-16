import numpy as np
from .connection import Connection
from .start_goal import Start, Goal

class ZK_Env:
    def __init__(self, env_args, render_mode=None):
        np.set_printoptions(precision=6, suppress=True, linewidth=80)
        # 顺序不能交换
        self.args = env_args
        self._set_render_mode(render_mode)
        self.connection = Connection(self.args)
        # 是否环境第一次启动
        self.initial = True
        # 提取我方单个飞机的原始观测数据,一个字典{}， reset in reset()
        # need to reset in reset_var()
        self.step_num = 0
        self.max_steps = 600

    def _set_render_mode(self, render_mode):
        # 演示模式，无加速
        if render_mode == 1:
            self.args.play_mode = 1
        # 训练调试模式，可加速查看
        if render_mode == 2:
            self.args.render = 1

    def close(self):
        self.connection.close()

    def reset(self):

        # 回合步数重置
        self.step_num = 0
        # 初始化
        self.goal = Goal()
        self.start = Start()
        init_data = self.start.get_init_data(self.initial, render=self.args.render)
        self.initial = False
        self.connection.send_condition(init_data)
        # 返回 red obs {"":"", .....}
        obs = self.compute_observation(self.connection.accept_from_socket())
        return obs, {}

    def step(self, action_array):

        action_dict = self.compute_action(action_array)

        self.connection.send_condition(action_dict)
        next_obs = self.compute_observation(self.connection.accept_from_socket())
        done_info = self._is_done()
        done = len(done_info) != 0

        self.step_num += 1
        info = {"done_info": done_info}

        return next_obs, done, info


    def _is_done(self):
        done_info = {}

        if self.step_num >= self.max_steps:
            done_info["is_max_steps_reach"] = True

        if abs(self.goal.altitude_err) > 4000:
            done_info["is_out_of_bounds"] = True

        if self.goal.is_reach_goal():
            done_info["is_reach_goal"] = True

        return done_info


    # ============================================================ obs related ============================================================

    def compute_observation(self, full_state):
        control_side = self.args.control_side
        obs = full_state[control_side][f'{control_side}_0']

        GEO_METER = 111319.49
        FOOT_METER = 0.3048

        state = [
            obs['position/lat-geod-deg'] * GEO_METER,
            obs['position/long-gc-deg'] * GEO_METER,
            obs['position/h-sl-ft'] * FOOT_METER,
            obs['velocities/mach'],
            np.radians(np.clip(obs['aero/alpha-deg'], -30, 30)),
            np.radians(np.clip(obs['aero/beta-deg'], -30, 30)),
            obs['velocities/p-rad_sec'],
            obs['velocities/q-rad_sec'],
            obs['velocities/r-rad_sec'],
            obs['attitude/roll-rad'],
            obs['attitude/pitch-rad'],
            np.radians(obs['attitude/psi-deg']),

        ]
        # print(f"{obs['simulation/dt']}")
        # print(f"{obs['simulation/sim-time-sec']}")
        observation = self.goal.update(state)
        return observation


    # ============================================================ action related ============================================================
    def compute_action(self, action_array):

        control_side = self.args.control_side
        action_dict = dict()
        action_dict[control_side] = {
            f'{control_side}_0': {
                'mode': 2,
                # 目标高度ft
                "target_altitude_ft": action_array[0],  # 负数 左下压
                # 目标速度ft
                "target_velocity": action_array[1],  #
                # 目标航向 -180 180
                "target_track_deg": action_array[2],  # 负数 上升

            }}


        return action_dict