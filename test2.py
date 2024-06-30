import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import random
import pickle
import numpy as np
from AUV import AUV_1
from models.dqn_model import DQNAgent
from env.SeafloorTerrainModel import SeafloorTerrain
from env.OceanCurrents import OceanCurrents
from env.CombinedEnvModel import CombinedEnvironment
from utils.visualization import EnvironmentPlotter
from utils.AUV_energy_model import EnergyModel
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
    # 定义环境数据路径和模型路径
    # 设置文件路径
    terrain_data_path = 'envs-data/terrain_data_2.pkl'
    current_data_path = 'envs-data/current_data_2.pkl'
    env_data_path = 'envs-data/environment_data_2.pkl'
    model_path = 'save/dqn_model_2.pth'
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
    success_count = 0
    for _ in range(100):

        # 随机生成AUV的初始位置和目标位置
        auv_start = generate_random_position(space_size, depth_data)
        auv_target = generate_random_position(space_size , depth_data)

        # 生成并保存综合环境数据
        environment = CombinedEnvironment(terrain, currents, auv_start, auv_target)
        environment.save_environment(env_data_path)

        # 创建可视化器
        plotter = EnvironmentPlotter(env_data_path)

        # 创建能耗模型实例
        energy_model = EnergyModel(current_data_path)

        # 创建AUV_1实例
        auv = AUV_1(env_data_path)

        # 定义状态空间大小和动作空间大小
        state_size = auv.state_size
        action_size = auv.action_size

        # 创建DQN代理并加载训练好的模型
        agent = DQNAgent(state_size, action_size)
        agent.model.load_state_dict(torch.load(model_path))
        agent.model.eval()

        # 获取起始状态
        state = np.array(auv.current_position)

        path = [state]  # 初始化路径记录

        done = False
        total_reward = 0

        # 设置epsilon为0以完全利用已经训练好的策略
        agent.epsilon = 0

        while not done:
            # 根据当前状态选择动作
            action = agent.act(state)

            # 执行动作并获取奖励和下一个状态
            reward, next_state, done = auv.move(action)

            # 更新总奖励
            total_reward += reward

            # 更新状态为下一个状态
            state = np.array(next_state)
            path.append(state)  # 记录路径

            # 如果到达目标位置，跳出循环
            if tuple(auv.current_position) == auv.auv_target:
                done = True
                success_count += 1
            # 打印总奖励
        print(f"Total Reward: {total_reward}, Steps: {auv.stepcost}")

        # 可视化路径
        plotter.plot_auv_path(path)

        # 计算路径的能耗
        energy_consumption = energy_model.calculate_energy(path)
        print(f"Energy Consumption: {energy_consumption}")

    # 计算并输出正确率
    success_rate = success_count / 100
    print(f'Success rate: {success_rate:.2%}')


if __name__ == '__main__':
    main()
