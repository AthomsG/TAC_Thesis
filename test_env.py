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