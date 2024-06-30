import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from AUV import AUV_1
from models.dqn_model import DQNAgent
from utils.visualization import EnvironmentPlotter
import matplotlib.pyplot as plt
import matplotlib

# 设置Matplotlib后端
matplotlib.use('TkAgg')


def main():
    # 定义环境数据路径
    env_data_path = 'env/envs-data/environment_data_2.pkl'
    # 创建可视化器
    plotter = EnvironmentPlotter(env_data_path)
    # 创建AUV_1实例
    auv = AUV_1(env_data_path)

    # 定义状态空间大小和动作空间大小
    state_size = auv.state_size
    action_size = auv.action_size
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建DQN代理
    agent = DQNAgent(state_size, action_size)
    stepcost = 0
    trace_path = []  # 路径记录
    trace_stepcost = []
    # 定义训练参数
    batch_size = 32
    episodes = 20000
    update_freq = 1000
    visual_freq = 500

    for episode in range(episodes):
        # 重置AUV环境状态
        with open(env_data_path, 'rb') as f:
            auv = AUV_1(env_data_path)

        # 获取起始状态
        state = np.array(auv.current_position)

        total_reward = 0
        done = False

        while not done:
            stepcost += 1
            # 根据当前状态选择动作
            action = agent.act(state)
            # 执行动作并获取奖励和下一个状态
            reward, next_state, done = auv.move(action)
            # 将经验存储到代理的记忆中
            agent.remember(state, action, reward, next_state, done)
            # 进行经验回放（每个步骤都进行一次）
            loss = agent.replay(batch_size)
            # 更新状态为下一个状态
            state = np.array(next_state)
            # 更新总奖励
            total_reward += reward

            # 定期更新目标网络
            if stepcost % update_freq == 0:
                agent.update_target_model()

        if episode >= 1 and agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        if episode and episode % visual_freq == 0:
            plotter.plot_auv_path(auv.path)
        if episode % 500 == 0:
            agent.save(f'save/dqn_model.pth')
        # 更新指标
        trace_stepcost.append(auv.stepcost)
        trace_path.append(auv.path)
        # print(auv.path)
        # 打印每个episode的总奖励
        print(
            f"Episode: {episode}, Total Reward: {total_reward}, Step: {auv.stepcost}, Loss: {loss}, Epsilon: {agent.epsilon}")
        # print(auv.path)
    agent.save(f'save/dqn_model.pth')

    print('Training finished.')


if __name__ == "__main__":
    main()