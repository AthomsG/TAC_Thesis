import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os

from environment import test_env
from networks import Actor, Critic

# set seeds for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
import random; random.seed(seed)

n_actions    = 4
n_iterations = 10000

discount = 0.99
lambd    = 0.1
lr       = 5e-2
alpha    = 1

q_losses = list()
p_losses = list()

# function approximators
actor  = Actor(n_actions)
critic = Critic(n_actions)

# optimizers
critic_opt = torch.optim.Adam(critic.parameters(), lr=lr)
actor_opt  = torch.optim.Adam(actor.parameters() , lr=lr)

# test environment
env = test_env(n_actions)
state = env.reset() # initial environment state

for i in range(n_iterations):
    # Environment step
    policy = actor(torch.tensor(state).float()) # compute policy
    action = np.argmax(policy.detach())          # sample from policy (using argmax for reproducibility)

    next_state, reward, _, _ = env.step(action.numpy()) # take environment step

    # Gradient step
    # Critic
    qvals = critic(torch.tensor(state).float())
    # compute target
    with torch.no_grad():
        next_qvals  = critic(torch.tensor(next_state).float())
        next_policy = actor(torch.tensor(next_state).float())
        next_action = np.argmax(next_policy.detach())
        target = reward + discount*(next_qvals[next_action] + (lambd/alpha)*torch.log(next_policy[next_action]))

    v_loss = F.mse_loss(qvals[action], target)/2
    q_losses.append(v_loss.item())

    # Actor
    p_loss = -(policy*(qvals.detach() - (lambd/alpha)*torch.log(policy))).sum()
    p_losses.append(p_loss.item())

    # Update parameters
    # Critic
    critic_opt.zero_grad()
    v_loss.backward()
    critic_opt.step()

    # Actor
    actor_opt.zero_grad()
    p_loss.backward()
    actor_opt.step()

    state = next_state

from IPython import embed; embed()