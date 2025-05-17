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
            "model": "qwen2:7b",
            "prompt": f"{prompt}",
            "stream": False
        }
    )
    print(response.json()["response"])


def local():
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2:7b",
            "prompt": "你好？",
            "stream": False
        }
    )
    print(response.json())


def remote_chat(prompt):


    response = requests.post(
        "http://192.168.2.62:11435/api/chat",
        json={
            "model": "qwen2:7b",
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}"
                }
            ],
            "stream": False
        }
    )
    print(response.json())


if __name__ == '__main__':
    start = time.time()
    for i in range(10):

        prompt = """你是一个无人机导航专家。我将提供的信息有：无人机信息：当前三维坐标位置和当前速度和当前航向；目标点信息：目标点三维坐标位置和到达目标点需要达到的速度。
你的任务是根据当前提供的信息对无人机进行全局的路径规划，你需要计算下一个子目标点T1的三维位置坐标和速度，直到到达终点。全局规划过程中的子目标点越少越好，不考虑障碍物。
当前坐标x=30000，y=0，z=6000，当前速度600。终点坐标x=10000，y=0，z=7000，终点速度900。你需要输出子目标点的坐标序列和速度序列。"""
        remote(prompt)
