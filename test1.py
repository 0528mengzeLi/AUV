import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from AUV import AUV_1
from models.dqn_model import DQNAgent
from utils.visualization import EnvironmentPlotter
from utils.AUV_energy_model import EnergyModel
import matplotlib

# 设置Matplotlib后端
matplotlib.use('TkAgg')


def main():
    # 定义环境数据路径和模型路径
    env_data_path = 'env/envs-data/environment_data.pkl'
    model_path = 'save/dqn_model.pth'
    current_data_file = 'env/envs-data/current_data.pkl'

    # 创建可视化器
    plotter = EnvironmentPlotter(env_data_path)

    # 创建能耗模型实例
    energy_model = EnergyModel(current_data_file)

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

    # 打印总奖励
    print(f"Total Reward: {total_reward}, Steps: {auv.stepcost}")

    # 可视化路径
    plotter.plot_auv_path(path)

    # 计算路径的能耗
    energy_consumption = energy_model.calculate_energy(path)
    print(f"Energy Consumption: {energy_consumption}")


if __name__ == "__main__":
    main()
