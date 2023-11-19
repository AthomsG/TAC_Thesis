from collections import defaultdict, deque
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
    

class Results_Buffer(object):
    '''
    This is the Results Buffer class. It's a part of an Actor-Critic algorithm implementation.
    The Results Buffer is a data structure to store the results of the agent's actions,
    allowing us to track the agent's performance over time. This helps in monitoring the learning progress.
    '''
    def __init__(self, rewards_history=[]):
        '''
        Initialize the Results Buffer with an optional initial rewards history.
        '''
        self.buffer = defaultdict(list)
        assert isinstance(rewards_history, list)
        self.rewards_history = rewards_history

    def update_infos(self, info, total_t):
        '''
        Update the Results Buffer with new information from the agent's actions on the environment (environment information).
        '''
        for key in info:
            msg = info[key]
            self.buffer['reward'].append(msg[b'reward'])
            self.buffer['length'].append(msg[b'length'])
            if b'real_reward' in msg:
                self.buffer['real_reward'].append(msg[b'real_reward'])
                self.buffer['real_length'].append(msg[b'real_length'])
                self.rewards_history.append(
                    [total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries): # não percebo para que preciso deste método
        '''
        Update the Results Buffer with new summaries of the agent's training (store the loss functions).
        '''
        for key in summaries:
            self.buffer[key].append(summaries[key])

    def add_to_writer(self, summary_writer, total_t, time):
        '''
        Add all scalars to the TensorBoar SummaryWriter.
        '''
        s = {'time': time}
        for key in self.buffer:
            if self.buffer[key]:
                s[key] = np.mean(self.buffer[key])
                self.buffer[key].clear()

        for key in s:
            summary_writer.add_scalar(key, s[key], total_t)
    
if DEBUG: from IPython import embed; embed()