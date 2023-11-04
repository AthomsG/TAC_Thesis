import sys
sys.path.append("..")

from value_networks import *
import torch
import numpy as np

# Fix seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0) 

BATCH_SIZE = 2
RAC_agent = RAC(n_actions=4)

# choose random observation
obs = torch.rand(2, 4, 84, 84)

# get policy distribution condition by observation
with torch.no_grad(): actions = RAC_agent.actor(obs)

print(f'Policy distribytion for random observation: {actions}')

# sample action from policy
action = RAC_agent.get_action(obs.numpy(), deterministic=False)
print(f'Sampled action: {action}')

# choose most likely action
action = RAC_agent.get_action(obs.numpy(), deterministic=True)
print(f'Deterministic action: {action}')

# test save and load methods

path_to_weights = 'test_RAC_save_and_load/original_RAC'

RAC_agent.save(path=path_to_weights)

new_RAC_agent = RAC(n_actions=4)
new_RAC_agent.load(path_to_weights)

# replicate experiment to check if model was properly loaded:
print('Reproducing experiment for loaded RAC agent:')

# get policy distribution condition by observation
with torch.no_grad(): actions = new_RAC_agent.actor(obs)

print(f'Policy distribytion for random observation: {actions}')

# sample action from policy
action = new_RAC_agent.get_action(obs.numpy(), deterministic=False)
print(f'Sampled action: {action}')

# choose most likely action
action = new_RAC_agent.get_action(obs.numpy(), deterministic=True)
print(f'Deterministic action: {action}')