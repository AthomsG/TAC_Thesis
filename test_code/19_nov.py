import sys
sys.path.append("..")

import matplotlib.pyplot as plt
from algorithm import RAC
from atari_env import atari_env
import torch

# check if agent learns the policy that corresponds to getting reward 1 if action corresponds to 0 or reward is null otherwise
agent = RAC('', 4, num_iterations=100)
agent.train('TEST_ENV')

with torch.no_grad(): policy = agent.actor(torch.rand(4, 84, 84))
print(f'policy for uniform sampled input is: {policy}') # -> seems to make sense

# check if frames are being properly stored on memory buffer.
# transition -> (states.squeeze(), action, rewards, next_states.squeeze(), dones)

agent = RAC('', 4, num_iterations=20)
agent.train('BreakoutNoFrameskip-v4')

last_trans  = agent.replay_buffer.mem[-1]
last_vision = last_trans[0][-1]

# plot last seen frame, to see if everything is ok with the replay buffer
plt.imshow(last_vision, cmap='gray')
plt.show() # -> Everything checks out...

from IPython import embed; embed()