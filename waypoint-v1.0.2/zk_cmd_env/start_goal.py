import numpy as np

class Start:
    def __init__(self, random=False):

        self.position = np.zeros(3)
        self.rng = np.random.default_rng()
        self.position[:2] = np.array([0, 30000])
        self.position[2] = 5000  # 起点海拔为最高+最低 除以2， 7500
        self.yaw = 270 + 30



    def get_init_data(self, initial, render=0):

        lat, long = self.position[:2] / 111319.49
        altitude_ft = self.position[2] * 3.28084
        yaw = self.yaw

        # 固定位置范围
        initial_data = {
            # 0.2703
            "red": {
                "red_0": {
                    "ic/h-sl-ft": altitude_ft,
                    "ic/terrain-elevation-ft": 1e-08,
                    "ic/long-gc-deg": long,
                    "ic/lat-geod-deg": lat,
                    "ic/u-fps": 600,
                    "ic/v-fps": 0,
                    "ic/w-fps": 0,
                    "ic/p-rad_sec": 0,
                    "ic/q-rad_sec": 0,
                    "ic/r-rad_sec": 0,
                    "ic/roc-fpm": 0,
                    "ic/psi-true-deg": yaw,
                    "ic/phi-deg": 0,
                    "ic/theta-deg": 0
                    # , "model": 16, "mode": 1.0,"target_longdeg":0.3,"target_latdeg":0.3,"target_altitude_ft":28000.0
                }
            }
        }
        if initial:
            initial_data.update({"flag": {"init": {"save": 0, "SplitScreen": 0, "render": render}}})
        else:
            initial_data.update({"flag": {"reset": {"save": 0, "SplitScreen": 0}}})

        return initial_data


class Goal:
    def __init__(self, random=False):

        # 终点位置
        self.position = np.zeros(3)
        self.position[0] = 0
        self.position[1] = 10000
        # self.position[1] = 0
        self.position[2] = 7000

        self.goal_reach_threshold = 100

        self.horizontal_distance = 0
        self.horizontal_distance_delta = 0

        self.bearing_err = 0
        self.bearing_err_abs_delta = 0

        self.altitude_err = 0
        self.altitude_err_abs_delta = 0


        self.mach_err = 0
        self.mach_err_delta = 0

        self.roll = 0
        self.pitch = 0

    def update_goal(self):
        rng = np.random.default_rng()
        distance_delta = 0
        position = np.zeros(3)
        while not 15000 < distance_delta < 20000 :
            distance = rng.random() * 30000.0 # 0-30000m 是目标点距离原点的范围
            bearing = rng.random() * 2 * np.pi
            position[:2] = np.cos(bearing), np.sin(bearing)
            position[:2] *= distance

            distance_delta = np.linalg.norm(self.position[:2] - position[:2])

        position[2] = (rng.random()*2-1) * 2000 + 7000

        self.position = np.array([0,0,7000])

    def update(self, state):

        # -------------------get state-------------------
        cur_position = state[:3]
        # 与目标位置的坐标偏差 单位米
        displacement = self.position - cur_position
        # 与目标的水平距离

        horizontal_distance = np.linalg.norm(displacement[:2])
        # 航向偏差
        bearing = state[11] if state[11] <= np.pi else state[11] - 2 * np.pi
        absolute_bearing = np.arctan2(displacement[1], displacement[0])
        # print(np.degrees(absolute_bearing))
        bearing_err = ((absolute_bearing - bearing) + np.pi) % (2 * np.pi) - np.pi
        # 高度偏差
        altitude_err = displacement[2]
        # 速度偏差
        mach_err = 1 - state[3]
        # alpha_beta = state[4:6]
        # angular_rates = state[6:9]
        # print(angular_rates[0])
        # 姿态角偏差
        roll, pitch = state[9], state[10]
        # -------------------get state-------------------
        pre_horizontal_distance = self.horizontal_distance
        self.horizontal_distance = horizontal_distance
        self.horizontal_distance_delta = pre_horizontal_distance - horizontal_distance

        pre_bearing_err = self.bearing_err
        self.bearing_err = bearing_err
        self.bearing_err_abs_delta = np.degrees( abs(pre_bearing_err) - abs(self.bearing_err) ) # clip [-2 2]
        # print('-----------------test---------------')
        # print(f"{np.degrees(old_bearing_err)}  {old_bearing_err}" )
        # print(f"{np.degrees(self.bearing_err)}  {self.bearing_err}")
        # print(f"{np.degrees(self.bearing_err_delta)}  {self.bearing_err_delta}")

        pre_altitude_err = self.altitude_err
        self.altitude_err = altitude_err
        self.altitude_err_abs_delta = abs(pre_altitude_err) - abs(self.altitude_err)
        # print("------------test-----------")
        # print(self.altitude_err_abs_delta)

        pre_mach_err = self.mach_err
        self.mach_err = mach_err
        self.mach_err_delta = pre_mach_err - self.mach_err

        self.roll = roll
        self.pitch = pitch

        obs_need_scale = np.array([displacement[0], displacement[1], bearing, roll, pitch])
        # print(obs_need_scale)
        low = np.array([-60000, -60000, -np.pi, -np.pi, -np.pi*0.5])
        high = np.array([60000, 60000, np.pi, np.pi, np.pi*0.5])
        obs_scale = 2 * (obs_need_scale - low) / (high - low) - 1
        return obs_scale

    def is_reach_goal(self):
        # if self.horizontal_distance < 400:
        #     print(self.horizontal_distance)
        # if self.horizontal_distance - self.goal_reach_threshold <= 0 and abs(self.altitude_err) - self.goal_reach_threshold/2 < 0:
        if self.horizontal_distance - self.goal_reach_threshold <= 0 and abs(self.altitude_err) - self.goal_reach_threshold/2 < 0:
            print(f"horizontal_distance: {self.horizontal_distance}")
            print(f"altitude_err_abs: {abs(self.altitude_err)}")
            print(f"roll: {np.degrees(abs(self.roll))}")
            print(f"pitch: {np.degrees(abs(self.pitch))}")
            print(f"mach_err: {abs(self.mach_err) * 340}")
            return True
        else:
            return False


