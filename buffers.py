from collections import defaultdict, deque
import numpy as np
import random 
import pickle
from atari_env import atari_env

DEBUG = False

class Replay_Buffer(object):
    '''
    This is the Replay Buffer class. It's a part of an Actor-Critic algorithm implementation.
    The Replay Buffer is a type of data structure to store the transitions that the agent observes,
    allowing us to reuse this data later. This allows the agent to learn from the past experiences.
    '''
    def __init__(self, capacity):
        '''
        Initialize the Replay Buffer with a given capacity.
        '''
        self.mem = deque(maxlen=capacity)

    def size(self):
        '''
        Return the current size of the Replay Buffer.
        '''
        return len(self.mem)

    def store(self, transition):
        '''
        Store a transition in the Replay Buffer
        '''
        self.mem.append(transition)

    def sample(self, batch_size):
        '''
        Sample a batch of transitions from the Replay Buffer.
        '''
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))

class debug_Replay_Buffer(Replay_Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.mem, f)

    def load(self, file_path):
        with open(file_path, 'rb') as f:
            self.mem = pickle.load(f)

    def sample(self, batch_size):
        samples = list(self.mem)[:batch_size]
        return map(np.array, zip(*samples))