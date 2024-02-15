import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect
import numpy as np

class Critic(nn.Module):
    def __init__(self, n_actions):
        super(Critic, self).__init__()
        # Linear Layers
        self.fc1 = nn.Linear(in_features=n_actions,
                             out_features=n_actions)
    def forward(self, x):
        x = self.fc1(x)
        return x

class Actor(nn.Module):
    def __init__(self, n_actions, alpha=1): # fixed alpha for entmax transformation
        super(Actor, self).__init__()
        self.alpha     = alpha
        # Linear Layer
        self.fc1 = nn.Linear(in_features=n_actions,
                             out_features=n_actions)

    def forward(self, x):
        x = self.fc1(x)
        # alpha - entmax is the optimal policy for Tsallis regularized MDP
        if self.alpha==1:
            x = F.softmax(x, dim=-1)
        elif self.alpha==1.5:
            x = entmax15(x, dim=-1)
        elif self.alpha==2:
            x = sparsemax(x, dim=-1)
        else:
            x = entmax_bisect(x, alpha=self.alpha, dim=-1) 
        return x