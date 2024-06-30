import numpy as np
import pickle
from utils.AUV_energy_model import water_resistance, thruster_efficiency
import matplotlib.pyplot as plt


class AUV_1:
    def __init__(self, env_data_path):
        # 初始化函数，加载环境数据并设置AUV初始状态
        with open(env_data_path, 'rb') as f:
            self.depth_data, self.Vx_total, self.Vy_total, self.auv_start, self.auv_target = pickle.load(f)
        self.space_size = self.depth_data.shape[0]  # 确定空间大小
        self.current_position = np.array(self.auv_start)  # 使用np.array转换为可变对象
        self.target_position = np.array(self.auv_target)
        self.state_size = 3
        self.action_size = 6

        self.stepcost = 0
        self.energy = 0
        self.max_search_steps = 800  # 最大搜索步数
        self.path = [self.auv_start]  # 当前路径

        self.trace_path = []  # 路径记录
        self.trace_stepcost = []
        self.trace_energy = []

        # 检查起始位置和目标位置是否有效
        if not self.is_valid_position(self.auv_start):
            raise ValueError("初始位置无效.")
        if not self.is_valid_position(self.auv_target):
            raise ValueError("目标位置无效.")

    def is_valid_position(self, position):
        # 检查位置是否在地形范围内
        x, y, z = position
        if x < 0 or x >= self.space_size or y < 0 or y >= self.space_size or z < 0 or z >= self.space_size:
            print('出界！')
            return False
        elif z <= self.depth_data[x, y] / 1000:
            print('碰到障碍物！')
            return False
        return True

    def move(self, action):

        done = False
        self.stepcost += 1  # 更新步数
        ori_dist = np.sqrt((self.current_position[0] - self.target_position[0]) ** 2 +
                           (self.current_position[1] - self.target_position[1]) ** 2 +
                           (self.current_position[2] - self.target_position[2]) ** 2)
        # 根据动作移动AUV，并更新路径
        next_position = self.current_position.copy()
        if action == 0:  # 上
            next_position[2] += 1
        elif action == 1:  # 下
            next_position[2] -= 1
        elif action == 2:  # 前
            next_position[0] += 1
        elif action == 3:  # 后
            next_position[0] -= 1
        elif action == 4:  # 左
            next_position[1] += 1
        elif action == 5:  # 右
            next_position[1] -= 1
        # 检查新位置是否有效
        if not self.is_valid_position(next_position):
            # 无效位置返回负奖励
            reward = -2000
            done = True

        else:  # 是有效位置

            self.current_position = next_position
            # 输出新的位置
            # print('新的位置:', self.current_position)
            self.path.append(self.current_position.copy())
            now_dist = np.sqrt((self.current_position[0] - self.target_position[0]) ** 2 +
                               (self.current_position[1] - self.target_position[1]) ** 2 +
                               (self.current_position[2] - self.target_position[2]) ** 2)
            # 检查是否到达目标位置
            if np.array_equal(self.current_position, self.target_position):
                done = True
                reward = 2000
                print('Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('Success!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            else:  # 没有到达
                reward = -1
                # 计算基于距离变化的奖励
                if now_dist <= ori_dist:
                    reward += 0  # 距离目标更近，奖励
                else:
                    reward -= 100  # 距离目标更远，惩罚

            # 获取当前位置的海流速度分量
            x, y = self.current_position[0], self.current_position[1]
            flow_speed_x = self.Vx_total[y, x]
            flow_speed_y = self.Vy_total[y, x]

            # 假设AUV以1m/s的速度行驶
            speed = 1

            # 根据AUV的行动方向调整真实速度计算
            if action == 0:  # 上
                true_speed = speed
                resistance = water_resistance(true_speed)
                efficiency = thruster_efficiency(true_speed)
            elif action == 1:  # 下
                true_speed = speed
                resistance = water_resistance(true_speed)
                efficiency = thruster_efficiency(true_speed)
            elif action == 2:  # 前（沿X轴）
                true_speed_x = speed + flow_speed_x
                true_speed_y = -flow_speed_y  # 抵消海流速度
                true_speed = np.sqrt(true_speed_x ** 2 + true_speed_y ** 2)
                resistance = water_resistance(true_speed)
                efficiency = thruster_efficiency(true_speed)
            elif action == 3:  # 后（沿X轴）
                true_speed_x = speed - flow_speed_x
                true_speed_y = flow_speed_y  # 抵消海流速度
                true_speed = np.sqrt(true_speed_x ** 2 + true_speed_y ** 2)
                resistance = water_resistance(true_speed)
                efficiency = thruster_efficiency(true_speed)
            elif action == 4:  # 左（沿Y轴）
                true_speed_x = -flow_speed_x  # 抵消海流速度
                true_speed_y = speed + flow_speed_y
                true_speed = np.sqrt(true_speed_x ** 2 + true_speed_y ** 2)
                resistance = water_resistance(true_speed)
                efficiency = thruster_efficiency(true_speed)
            elif action == 5:  # 右（沿Y轴）
                true_speed_x = flow_speed_x  # 抵消海流速度
                true_speed_y = speed - flow_speed_y
                true_speed = np.sqrt(true_speed_x ** 2 + true_speed_y ** 2)
                resistance = water_resistance(true_speed)
                efficiency = thruster_efficiency(true_speed)

            # 计算能耗
            self.energy = (resistance * 1) / efficiency  # 假设每一步移动的距离为1
            self.trace_energy.append(self.energy)

            reward -= int(self.energy)  # 能耗作为负奖励

        # 检查是否超过最大步数
        if self.stepcost >= self.max_search_steps:
            print('超过最大步数!')
            done = True
            reward = -2000

        return reward, self.current_position.copy(), done

    def output_path(self):
        print('路径:', self.path)

    def visualize_metrics(self):  # 可视化 步数 变化

        epochs = range(len(self.trace_energy))

        fig, axs = plt.subplots(1, 1, figsize=(6, 8))

        # step plot
        axs.plot(epochs, self.trace_stepcost, label='Step', color='blue')
        axs.set_title('Time per Epoch')
        axs.set_xlabel('Epoch')
        axs.set_ylabel('fuel')
        axs.legend()

        plt.tight_layout()
        plt.show()
