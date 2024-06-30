import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class EnvironmentPlotter:
    def __init__(self, env_data_path):
        with open(env_data_path, 'rb') as f:
            self.depth_data, self.Vx_total, self.Vy_total, self.auv_start, self.auv_target = pickle.load(f)
        self.space_size = self.depth_data.shape[0]
        self.x, self.y = np.meshgrid(np.linspace(0, self.space_size - 1, self.space_size),
                                     np.linspace(0, self.space_size - 1, self.space_size))

    def plot_environment_model(self):
        # 将深度数据从米转换为公里
        depth_data_km = self.depth_data / 1000.0
        # 绘制3D地形数据
        fig = plt.figure(figsize=(14, 8))

        ax = fig.add_subplot(121, projection='3d')
        # 标记AUV的初始位置和目标位置
        ax.scatter(self.auv_start[0], self.auv_start[1], self.auv_start[2], color='blue', s=100,
                   label='AUV Start')
        ax.scatter(self.auv_target[0], self.auv_target[1], self.auv_target[2], color='red', s=100,
                   label='AUV Target')

        # 绘制3D表面
        surf = ax.plot_surface(self.x, self.y, depth_data_km, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Depth (km)')

        # 设置轴标签
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Depth (km)')

        # 设置标题
        ax.set_title('3D Seafloor Terrain Model')

        # 计算海流速度的大小
        magnitude = np.sqrt(self.Vx_total ** 2 + self.Vy_total ** 2)

        # 绘制2D静态海流模型
        ax2 = fig.add_subplot(122)
        ax2.contourf(self.x, self.y, depth_data_km, cmap='viridis')
        quiver = ax2.quiver(self.x, self.y, self.Vx_total, self.Vy_total, magnitude, scale=50, cmap='coolwarm')
        fig.colorbar(quiver, ax=ax2, shrink=0.5, aspect=5, label='Current Velocity (m/s)')

        # 绘制初始位置和目标位置
        ax2.scatter(self.auv_start[0], self.auv_start[1], color='blue', label='AUV Start')
        ax2.scatter(self.auv_target[0], self.auv_target[1], color='red', label='AUV Target')

        # 设置轴标签
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.set_title('Static Ocean Current Model')

        plt.tight_layout()
        plt.show()

    def plot_auv_path(self, path):
        # 绘制3D地形数据
        depth_data_km = self.depth_data / 1000.0

        fig = plt.figure(figsize=(14, 8))

        ax = fig.add_subplot(121, projection='3d')
        # 标记AUV的初始位置和目标位置
        ax.scatter(self.auv_start[0], self.auv_start[1], self.auv_start[2], color='blue', label='AUV Start')
        ax.scatter(self.auv_target[0], self.auv_target[1], self.auv_target[2], color='red', label='AUV Target')

        # 绘制3D表面
        surf = ax.plot_surface(self.x, self.y, depth_data_km, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Depth (km)')
        # 打印路径坐标以调试
        for p in path:
            print(f"Path Point: x={p[0]}, y={p[1]}")
        # 绘制路径
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        path_z = [p[2] for p in path]
        ax.plot(path_x, path_y, path_z, color='yellow', label='Path')

        # 设置轴标签
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Depth (km)')

        # 设置标题
        ax.set_title('3D Seafloor Terrain Model with Path')
        ax.legend()

        # 计算海流速度的大小
        magnitude = np.sqrt(self.Vx_total ** 2 + self.Vy_total ** 2)
        # 绘制2D静态海流模型
        ax2 = fig.add_subplot(122)
        ax2.contourf(self.x, self.y, self.depth_data, cmap='viridis')
        quiver = ax2.quiver(self.x, self.y, self.Vx_total, self.Vy_total, magnitude, scale=50, cmap='coolwarm')
        fig.colorbar(quiver, ax=ax2, shrink=0.5, aspect=5, label='Current Velocity (m/s)')

        # 绘制初始位置和目标位置
        ax2.scatter(self.auv_start[0], self.auv_start[1], color='blue', label='AUV Start')
        ax2.scatter(self.auv_target[0], self.auv_target[1], color='red', label='AUV Target')

        # 绘制路径
        ax2.plot(path_x, path_y, color='yellow', label='Path')

        # 设置轴标签
        ax2.set_xlabel('X (km)')
        ax2.set_ylabel('Y (km)')
        ax2.set_title('Static Ocean Current Model')
        ax2.legend()

        plt.tight_layout()
        plt.show()


