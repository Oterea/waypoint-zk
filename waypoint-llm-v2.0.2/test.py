import torch
import sys
import os
import numpy as np
import requests
def cuda():
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())

def llm():
    from zk_cmd_env.llm_api import llm_output
    a = llm_output("aa")
    print(a)

def tes():
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [
                {
                    "role": "user",
                    "content": "你好？"
                }
            ],
            "stream": False
        }
    )
    print(response.json())

if __name__ == '__main__':
    tes()




