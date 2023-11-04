import sys
sys.path.append("..")

from value_networks import *
from atari_env import atari_env
import torch

BATCH_SIZE = 1

# BreakoutNoFrameskip-v4 -> n_actions=4  
net_input = torch.rand(BATCH_SIZE, 4, 84, 84) 
for alpha in [1, 1.5, 2]:
	actor = Actor(n_actions=4, alpha=alpha)
	actor(net_input)

critic = Critic(n_actions=4)
critic(net_input)

print('Code is checked for torch random input.')
print('Cheking neural nets on Gym environment...')

import gym

env = atari_env("BreakoutNoFrameskip-v4")
obs = env.reset()

obs_tensor = torch.tensor(obs)

critic(obs_tensor)
actor(obs_tensor)

print('Code is checked for Gym env observations')