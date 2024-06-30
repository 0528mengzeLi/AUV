import numpy as np
import pickle


class OceanCurrents:
    def __init__(self, space_size=11, delta=1.0, gamma=10):
        self.space_size = space_size
        self.delta = delta
        self.gamma = gamma
        self.Vx_total, self.Vy_total = self.generate_currents()

    def vortex_velocity(self, x, y, r0):
        r_squared = (x - r0[0]) ** 2 + (y - r0[1]) ** 2
        exp_component = np.exp(-r_squared / self.delta ** 2)
        r_squared += 1e-10  # 添加一个小常数以避免除零错误

        Vx = -self.gamma * (y - r0[1]) / (2 * np.pi * r_squared) * (1 - exp_component)
        Vy = self.gamma * (x - r0[0]) / (2 * np.pi * r_squared) * (1 - exp_component)

        return Vx, Vy

    def generate_currents(self):
        np.random.seed(42)  # 固定随机种子以便重复实验
        vortex_centers = np.random.randint(0, self.space_size, size=(3, 2))

        x = np.linspace(0, self.space_size - 1, self.space_size)
        y = np.linspace(0, self.space_size - 1, self.space_size)
        x, y = np.meshgrid(x, y)

        Vx_total = np.zeros((self.space_size, self.space_size))
        Vy_total = np.zeros((self.space_size, self.space_size))

        for r0 in vortex_centers:
            Vx, Vy = self.vortex_velocity(x, y, r0)
            Vx_total += Vx
            Vy_total += Vy

        return Vx_total, Vy_total

    def save_currents(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.Vx_total, self.Vy_total), f)

# # 示例使用
# currents = OceanCurrents()
# currents.save_currents('envs-data/ocean_currents.pkl')
