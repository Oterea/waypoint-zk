import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_airplane(ax, position, yaw_deg=0, size=800, color='orange'):
    body = np.array([[1.0, 0.0, 0.0], [-1.0, -0.3, 0.0], [-1.0, 0.3, 0.0]]) * size
    wings = np.array([[-0.2, -1.0, 0.0], [-0.2, 1.0, 0.0], [0.0, 0.0, 0.0]]) * size
    yaw = np.radians(yaw_deg)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    body_rot = body @ Rz.T + position
    wings_rot = wings @ Rz.T + position
    ax.add_collection3d(Poly3DCollection([body_rot], color=color, alpha=1))
    ax.add_collection3d(Poly3DCollection([wings_rot], color=color, alpha=0.7))

# 路径点
path_points = np.array([
    [30000, 0, 6000],
    [27000, 0, 6100],
    [24000, 0, 6250],
    [21000, 0, 6400],
    [18000, 0, 6600],
    [15000, 0, 6800],
    [12000, 0, 6950],
    [10000, 0, 7000],
])

# 计算第一段的 yaw（用于起点飞机朝向）
delta = path_points[1] - path_points[0]
yaw_rad = np.arctan2(delta[1], delta[0])
yaw_deg = np.degrees(yaw_rad)

# 障碍物
barriers = [
    {"x": 0, "y": 20000, "z": 7000, "r": 50},
    {"x": 0, "y": 25000, "z": 6500, "r": 50}
]
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 路径线
ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2],
        color='dodgerblue', linestyle='--', linewidth=2, label='Path')

# 起点和终点
ax.scatter(*path_points[0], c='green', s=80, label='Start')
ax.text(*path_points[0] + np.array([0, 0, 150]), "Start", color='green')
ax.scatter(*path_points[-1], c='red', s=80, label='End')
ax.text(*path_points[-1] + np.array([0, 0, 150]), "End", color='red')

# 单架飞机（在起点，朝向第一段路径）
draw_airplane(ax, position=path_points[0], yaw_deg=yaw_deg, size=800, color='orange')

# 障碍物球体
for b in barriers:
    x = b["r"] * np.outer(np.cos(u), np.sin(v)) + b["x"]
    y = b["r"] * np.outer(np.sin(u), np.sin(v)) + b["y"]
    z = b["r"] * np.outer(np.ones(np.size(u)), np.cos(v)) + b["z"]
    ax.plot_surface(x, y, z, color='gray', alpha=0.4)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (Altitude, m)")
ax.set_title("3D Path with Sub-Goals and One Aircraft")
ax.set_box_aspect([1, 1, 0.5])
ax.view_init(elev=30, azim=120)
ax.legend()
plt.tight_layout()
plt.show()
