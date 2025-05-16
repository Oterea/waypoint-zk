import time
import torch
import requests


def cuda():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())


def remote(prompt):
    response = requests.post(
        "http://192.168.2.62:11435/api/generate",
        json={
            "model": "qwen2:0.5b",
            "prompt": f"{prompt}",
            "stream": False
        }
    )
    print(response.json()["response"])
def local():
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen:0.5b",
            "prompt": "你好？",
            "stream": False
        }
    )
    print(response.json())


if __name__ == '__main__':
    start = time.time()
    for i in range(10):
        prompt = """
        任务描述：你需要根据下面的信息输出格式为["目标高度", "目标速度", "目标航向",]的子目标。我有一个底层控制器可以接受这个子目标进而控制f16无人战斗机的飞行导航到目标点，同时避开障碍物。
下面的信息包括无人机当前信息：当前三维坐标、当前速度、当前航向。终点信息：终点三维坐标、终点需要达到的速度。障碍物的三维坐标。
坐标单位是米，速度单位是米每秒，航向单位是度(规定航向范围是(-180,180]，机头朝正y方向，航向为0；机头朝负y方向，航向为180；机头朝正x方向，航向为90；机头朝负x方向，航向为-90)
"cur":{"cur_x":30000, "cur_y":0, "cur_z":6000, "cur_v":600, "cur_yaw": -90},
"end":{"end_x":10000, "end_y":0, "end_z":7000, "end_v":900},
"barrier":[{"b1_x":0, "b1_y":20000, "b1_z":7000}, {"b2_x":0, "b2_y":25000, "b2_z":6500}]
        """
        remote(prompt)





