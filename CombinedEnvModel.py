import pickle


class CombinedEnvironment:
    def __init__(self, terrain, currents, auv_start, auv_target):
        self.terrain = terrain
        self.currents = currents
        self.auv_start = auv_start
        self.auv_target = auv_target

    def save_environment(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.terrain.depth_data, self.currents.Vx_total, self.currents.Vy_total, self.auv_start,
                         self.auv_target), f)
