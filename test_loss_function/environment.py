import numpy as np

class test_env():
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.step_i = 0
        self.state  = np.linspace(1, self.n_actions, self.n_actions)

    def reset(self):
        self.step_i = 0
        self.state  = np.linspace(1, self.n_actions, self.n_actions)
        return self.state
    
    def print_state(self):
        print(self.state)

    def step(self, action):
        self.step_i += 1
        reward       = self.state[action]
        next_state   = np.sin(np.linspace(1, self.n_actions, self.n_actions) + self.step_i)
        self.state   = next_state

        return next_state, reward, 0, 'no info'