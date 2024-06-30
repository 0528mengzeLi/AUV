import os
import numpy as np
import random
import pickle
from SeafloorTerrainModel import SeafloorTerrain
from OceanCurrents import OceanCurrents
from CombinedEnvModel import CombinedEnvironment
from PlotEnvModel import EnvironmentPlotter
import matplotlib

# 设置Matplotlib后端
matplotlib.use('TkAgg')


def generate_random_position(space_size, depth_data):
    while True:
        x = random.randint(0, space_size - 1)
        y = random.randint(0, space_size - 1)
        z = random.randint(0, 10)
        if z > depth_data[x, y] / 1000:
            return x, y, z


def main():
    # 设置文件路径
    terrain_data_path = 'envs-data/terrain_data_2.pkl'
    current_data_path = 'envs-data/current_data_2.pkl'
    env_data_path = 'envs-data/environment_data_2.pkl'

    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists('envs-data'):
        os.makedirs('envs-data')

    # 生成地形数据
    terrain = SeafloorTerrain(space_size=11)
    terrain.save_depth_data(terrain_data_path)

    # 生成海流数据
    currents = OceanCurrents(space_size=11)
    currents.save_currents(current_data_path)

    # 加载地形数据以进行位置验证
    with open(terrain_data_path, 'rb') as f:
        depth_data = pickle.load(f)
    space_size = depth_data.shape[0]

    # 设置AUV的初始位置和目标位置
    # # 固定
    # auv_start = [2, 8, 2]
    # auv_target = [8, 2, 3]
    # 随机
    auv_start = generate_random_position(space_size, depth_data)
    auv_target = generate_random_position(space_size, depth_data)

    # 生成并保存综合环境数据
    environment = CombinedEnvironment(terrain, currents, auv_start, auv_target)
    environment.save_environment(env_data_path)

    # 绘制环境模型
    plotter = EnvironmentPlotter(env_data_path)
    plotter.plot_environment_model()


if __name__ == '__main__':
    main()
