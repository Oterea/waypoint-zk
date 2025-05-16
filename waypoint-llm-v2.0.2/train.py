import tyro
from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args
import logging
from zk_cmd_env.llm_api import compute_action_with_obs
import asyncio

async def run(action):
    for i in range(10):
        if action is None:
            action = 1
        print(f"{action}")

async def call_api(prompt):
    return 999

async def runner():
    task = asyncio.create_task(call_api("你好？"))
    action = None
    await run(action)

    action = await task

# if __name__ == "__main__":
#     obs = env.reset()
#
#
#     while not done:
#         # 程序片段B,通过大模型api根据obs计算action
#         action = api(obs)
#         # 程序片段A，根据action执行动作
#         obs, action, done = env.step(action)
#
#     当reset返回的obs，第一次交给程序B的api阻塞同步执行，得到action，然后交给程序A执行，程序A执行一次之后返回了obs
#     然后程序B根据新的obs异步执行，同时程序片段A也并发执行，在程序B没有得到结果的时候，程序A中使用的action一直是上一次的action。当程序B执行完毕的时候，程序A
#     用新的action











