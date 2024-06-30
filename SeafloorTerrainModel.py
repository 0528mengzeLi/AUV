import numpy as np
import pickle


class SeafloorTerrain:
    def __init__(self, space_size=11):
        self.space_size = space_size
        self.grid_size = 1  # 每个网格的边长为1 km
        self.depth_data = self.generate_depth_data()

    def generate_depth_data(self):
        # 生成X, Y轴的坐标
        x = np.linspace(0, self.space_size - 1, self.space_size)
        y = np.linspace(0, self.space_size - 1, self.space_size)

        # 创建二维网格
        x, y = np.meshgrid(x, y)

        # 生成简化的深度数据
        depth_data = (
            2000 * np.sin(2 * np.pi * x / self.space_size) * np.sin(2 * np.pi * y / self.space_size) +
            1000 * np.cos(4 * np.pi * x / self.space_size) * np.cos(4 * np.pi * y / self.space_size)
        )

        # 中心的噪声峰值区域
        center = self.space_size // 2
        peak_size = 1  # 噪声峰值的区域大小
        depth_data[center - peak_size:center + peak_size, center - peak_size:center + peak_size] += 3000

        # 将深度数据裁剪到合理的范围
        depth_data = np.clip(depth_data, 0, 6000)

        # 确保所有深度都从0开始
        min_depth = np.min(depth_data)
        depth_data = depth_data - min_depth

        return depth_data

    def save_depth_data(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.depth_data, f)

    def print_depth_data(self):
        # 打印每个网格点的深度数据
        for i in range(self.space_size):
            for j in range(self.space_size):
                print(f"Point ({i}, {j}): Depth = {self.depth_data[i, j]}")


# 示例使用
terrain = SeafloorTerrain()
terrain.save_depth_data('envs-data/terrain_data.pkl')
terrain.print_depth_data()
