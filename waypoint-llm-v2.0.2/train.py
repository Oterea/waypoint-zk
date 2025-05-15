import tyro
from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args
import logging
from zk_cmd_env.llm_api import compute_action_with_obs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    env_args = tyro.cli(Env_Args)

    env = ZK_Env(env_args, render_mode=2)
    for i in range(10):
        next_obs, _ = env.reset()
        next_done = False
        while not next_done:
            # TODO
            action_array = compute_action_with_obs(next_obs)

            next_obs, next_done, info = env.step(action_array)






