import time
import requests
import numpy as np
import threading
from zk_cmd_env.zk_env import ZK_Env
from config import Env_Args
import tyro

last_action = None
lock = threading.Lock()  # 控制多线程下的变量安全
llm_done = threading.Event()


def call_llm_api(obs, count):
    prompt = """
    任务描述：你需要根据下面的信息输出格式为["目标高度", "目标速度", "目标航向",]的子目标。我有一个底层控制器可以接受这个子目标进而控制f16无人战斗机的飞行导航到目标点，同时避开障碍物。
下面的信息包括无人机当前信息：当前三维坐标、当前速度、当前航向。终点信息：终点三维坐标、终点需要达到的速度。障碍物的三维坐标。
坐标单位是米，速度单位是米每秒，航向单位是度(规定航向范围是(-180,180]，机头朝正y方向，航向为0；机头朝负y方向，航向为180；机头朝正x方向，航向为90；机头朝负x方向，航向为-90)
{"cur":{"cur_x":30000, "cur_y":0, "cur_z":6000, "cur_v":600, "cur_yaw": -90},
"end":{"end_x":10000, "end_y":0, "end_z":7000, "end_v":900},
"barrier":[{"b1_x":0, "b1_y":20000, "b1_z":7000}, {"b2_x":0, "b2_y":25000, "b2_z":6500}]}
输出示例：[7000,700,-90]
    """
    prompt = prompt.strip()
    try:
        response = requests.post(
            # "http://localhost:11434/api/generate",
            "http://192.168.2.62:11435/api/generate",
            json={
                "model": "qwen2:0.5b",
                "prompt": prompt,
                "stream": False
            }
        )
        print(f"[{prompt}] 响应内容：", response.json())
    except Exception as e:
        print("请求失败：", e)
        return

    new_action = np.array([7000 * 3.28084, 600+count*10, -90-count*10], dtype=np.float64)

    with lock:
        global last_action
        last_action = new_action

    llm_done.set()  # 通知主线程，LLM action 更新完成
    print("llm done")

def control_loop(env):
    global last_action
    count = 0
    obs = env.reset()
    done = False

    # 第一次同步获取 action
    last_action = np.array([7000 * 3.28084, 600, -90], dtype=np.float64)

    # 异步发起下一次 action 获取请求
    llm_thread = threading.Thread(target=call_llm_api, args=(obs, count))
    llm_thread.start()


    while not done:
        with lock:
            current_action = last_action.copy()
        print(current_action)
        obs, done, info = env.step(current_action)

        if llm_done.is_set():
            # LLM 返回了结果，启动下一轮请求
            llm_done.clear()
            threading.Thread(target=call_llm_api, args=(obs, count)).start()
        count += 1
        # time.sleep(0.05)  # 控制主循环频率


def main():
    env_args = tyro.cli(Env_Args)
    env = ZK_Env(env_args, render_mode=2)
    for i in range(100):
        control_loop(env)


if __name__ == '__main__':
    main()
