import pickle
import numpy as np


# 计算水阻力与航行速度的方程
def water_resistance(speed):
    return (0.05 * 1025 * 0.7 * (speed ** 2)) / 2
    # 1025kg/m^3为海水密度,0.7为阻力系数,AUV横截面积为0.05m^2


# 推进器推力与效率的方程
def thruster_efficiency(speed):
    return 0.9  # 假设推进器效率为90%


class EnergyModel:
    def __init__(self, current_data_file):
        self.load_current_data(current_data_file)

    def load_current_data(self, current_data_file):
        with open(current_data_file, 'rb') as f:
            # 假设 current_data 文件中的数据是一个包含两个数组的元组
            self.Vx_total, self.Vy_total = pickle.load(f)

    def calculate_energy(self, path):
        energy = 0
        speed = 1  # 假设AUV以1m/s的速度行驶
        for i in range(len(path) - 1):
            current_position = path[i]
            next_position = path[i + 1]

            # 获取当前位置的海流速度分量
            x, y = int(current_position[0]), int(current_position[1])
            flow_speed_x = self.Vx_total[y, x]
            flow_speed_y = self.Vy_total[y, x]

            # 根据AUV的行动方向调整真实速度计算
            if next_position[0] > current_position[0]:  # 前
                true_speed_x = speed + flow_speed_x
                true_speed_y = -flow_speed_y  # 抵消海流速度
            elif next_position[0] < current_position[0]:  # 后
                true_speed_x = speed - flow_speed_x
                true_speed_y = flow_speed_y  # 抵消海流速度
            elif next_position[1] > current_position[1]:  # 上
                true_speed_x = -flow_speed_x  # 抵消海流速度
                true_speed_y = speed + flow_speed_y
            elif next_position[1] < current_position[1]:  # 下
                true_speed_x = -flow_speed_x  # 抵消海流速度
                true_speed_y = speed + flow_speed_y
            elif next_position[2] > current_position[2]:  # 左（假设Z轴代表上下运动）
                true_speed_x = -flow_speed_x  # 抵消海流速度
                true_speed_y = speed + flow_speed_y
            elif next_position[2] < current_position[2]:  # 右（假设Z轴代表上下运动）
                true_speed_x = flow_speed_x  # 抵消海流速度
                true_speed_y = speed - flow_speed_y
            else:  # 如果没有移动，则跳过这一步
                continue

            true_speed = np.sqrt(true_speed_x ** 2 + true_speed_y ** 2)

            # 计算当前速度下的水阻力和推进器效率
            resistance = water_resistance(true_speed)
            efficiency = thruster_efficiency(true_speed)

            # 计算当前路径段的能耗，并累加到总能耗
            energy += (resistance * 1) / efficiency  # 每一步的距离为1

        return energy
