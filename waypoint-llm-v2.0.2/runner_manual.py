
import numpy as np
from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args
import tyro

def compute_action(obs):
    coef = 3.28084
    target_altitude_ft = obs[0]*coef
    target_velocities_ft = obs[0]*coef
    target_track_deg = obs[0]
    action = np.array([target_altitude_ft, target_velocities_ft, target_track_deg], dtype=np.float64)
    return action



if __name__ == '__main__':
    env_args = tyro.cli(Env_Args)
    env = ZK_Env(env_args, render_mode=2)
    obs, _ = env.reset()
    done = False
    while not done:
        action = compute_action(obs)
        obs, done, info = env.step(action)