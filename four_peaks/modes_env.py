import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Mode_Seeking_Environment:
    def __init__(self, peak_steepness=0.005, action_discretization=5, grid_size=(100, 100), epsilon=1.0, max_distance=10, local_region_size=(3, 3), reset_interval=1000):
        self.peak_steepness = peak_steepness
        self.max_distance = max_distance
        self.action_discretization = action_discretization
        self.grid_size = grid_size
        self.epsilon = epsilon
        self.local_region_size = local_region_size
        self.reset_interval = reset_interval
        self.peaks = [(25, 25), (75, 25), (25, 75), (75, 75)]
        self.agent_position = np.array([grid_size[0] / 2, grid_size[1] / 2])
        self.action_space = self._create_action_space(action_discretization)
        self.reward_matrix = self._create_reward_matrix()
        self.step_count = 0 

    def _create_action_space(self, discretization):
        directions = np.linspace(0, 2 * np.pi, discretization, endpoint=False)
        distances = np.linspace(0, self.max_distance, discretization)
        actions = [(np.array([np.cos(d), np.sin(d)]) * dist) for d in directions for dist in distances if dist > 0]
        return np.array(actions)

    def _create_reward_matrix(self):
        x = np.linspace(0, self.grid_size[0], self.grid_size[0])
        y = np.linspace(0, self.grid_size[1], self.grid_size[1])
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                Z[i, j] = self.reward_function([X[i, j], Y[i, j]])
        return Z

    def reward_function(self, position):
        return np.sum([np.exp(-self.peak_steepness * np.linalg.norm(position - np.array(peak))**2) for peak in self.peaks])

    def get_local_matrix(self):
        half_height = self.local_region_size[0] // 2
        half_width = self.local_region_size[1] // 2
        center_x, center_y = int(self.agent_position[0]), int(self.agent_position[1])
        
        local_matrix = np.zeros(self.local_region_size)
        
        for i in range(-half_height, half_height + 1):
            for j in range(-half_width, half_width + 1):
                x, y = center_x + i, center_y + j
                if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]:
                    local_matrix[i + half_height, j + half_width] = self.reward_matrix[x, y]
        
        return local_matrix

    def step(self, action_index):
        action = self.action_space[action_index]
        new_state = self.agent_position + action
        new_state = np.clip(new_state, [0, 0], np.array(self.grid_size) - 1)
        self.agent_position = new_state
        reward = self.reward_function(self.agent_position)
        self.step_count += 1
        local_matrix = self.get_local_matrix()
        if self.step_count >= self.reset_interval:
            self.reset()
        return local_matrix, reward, False

    def reset(self):
        self.agent_position = np.array([self.grid_size[0] / 2, self.grid_size[1] / 2])
        self.step_count = 0
        return self.get_local_matrix()

    def visualize(self):
        x = np.linspace(0, self.grid_size[0], 200)
        y = np.linspace(0, self.grid_size[1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(200):
            for j in range(200):
                Z[i, j] = self.reward_function([X[i, j], Y[i, j]])
        
        contour_levels = np.arange(0.1, 1.0, 0.1)
        plt.figure(figsize=(8, 8)) 
        contours = plt.contour(X, Y, Z, levels=contour_levels, cmap='coolwarm')
        plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
        plt.plot(self.agent_position[0], self.agent_position[1], 'ko', markersize=10)
        
        peak_color = cm.coolwarm(1.0)
        for peak in self.peaks:
            plt.plot(peak[0], peak[1], 'o', color=peak_color, markersize=8)
        
        plt.title('Reward Function Contours')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()