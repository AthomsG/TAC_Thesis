from collections import deque
import numpy as np
import random 

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

    def append(self, transition):
        '''
        Append a transition to the Replay Buffer.
        '''
        self.mem.append(transition)

    def extend(self, transitions):
        '''
        Extend the Replay Buffer with a list of transitions.
        '''
        for t in transitions:
            self.append(t)

    def sample(self, batch_size):
        '''
        Sample a batch of transitions from the Replay Buffer.
        '''
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))
    
if DEBUG: from IPython import embed; embed()