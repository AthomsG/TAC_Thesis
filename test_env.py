import numpy as np


class TestEnv():
    def __init__(self, states_shape, rewarded_action=1):
        self.rewarded_action= rewarded_action
        self.states_shape   = states_shape

    def reset(self):
        return np.random.rand(*self.states_shape)
    
    def step(self, action):
        obs = np.random.rand(*self.states_shape)
        reward = float(action==self.rewarded_action)
        done = np.random.rand() > 0.7
        info = None

        return obs, reward, done, info
    
    
class GridWorld():
    def __init__(self, grid_size=84):
        self.grid_size = grid_size
        self.observation_space = np.zeros((grid_size, grid_size))
        self.action_space = np.array([0, 1, 2, 3])  # Assuming 4 possible actions
        self.rewarded_positions = [(0, 0), (0, grid_size-1), (grid_size-1, 0), (grid_size-1, grid_size-1)]
        self.agent_position = [grid_size // 2, grid_size // 2]

    def reset(self):
        self.agent_position = [self.grid_size // 2, self.grid_size // 2]
        return self._get_state()

    def step(self, action):
        # action effects
        action_effects = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        # compmute new position
        new_position = [self.agent_position[0] + action_effects[action][0], 
                        self.agent_position[1] + action_effects[action][1]]
        
        # if new position is outside grid bounds reset to starting position
        if new_position[0] < 0 or new_position[0] >= self.grid_size or new_position[1] < 0 or new_position[1] >= self.grid_size:
            self.agent_position = [self.grid_size // 2, self.grid_size // 2]
        else:
            self.agent_position = new_position

        done = tuple(self.agent_position) in self.rewarded_positions
        reward = 1.0 if done else -1.0
        return self._get_state(), reward, done, None

    def _get_state(self):
        state = np.zeros((self.grid_size, self.grid_size))
        state[self.agent_position[0], self.agent_position[1]] = 1.0
        for pos in self.rewarded_positions:
            state[pos[0], pos[1]] = 0.5
        # stack the state 4 times along a new axis for the RAC agent to process
        state = np.stack([state]*4, axis=0)
        return state

# flatten state to use with MLP
class GridWorldWrapper(GridWorld):
    def __init__(self, grid_size=84):
        super().__init__(grid_size)

    def _get_state(self):
        state = super()._get_state()
        state = state[0].flatten() # I'm doing this here because I stacked 4 layers since its how the atari env is setup
        return state