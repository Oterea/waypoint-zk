import numpy as np
import requests
import json


def compute_action_with_obs(next_obs):
    prompt = ""
    content = llm_output(prompt)
    content = ""
    action_array = compute_action(content)
    return action_array


def llm_output(prompt):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "qwen2.5:0.5b",
            "messages": [
                {
                    "role": "user",
                    "content": "你好"
                }
            ],
            "stream": False
        }
    )

    # raw = response.json()
    # content= raw.get("choices")[0].get("message").get("content")
    content = response.json()
    return content

def compute_action(content):
    # [0, 10000] m, [0, 32808] ft
    target_altitude_ft = 7000*3.28084
    # [0, 340] m/s, [0, 1115] ft/s
    target_velocity = 600
    # [-180, 180] deg
    target_track_deg = -90
    action_array = np.array([target_altitude_ft, target_velocity, target_track_deg], dtype=np.float64)
    return action_array
