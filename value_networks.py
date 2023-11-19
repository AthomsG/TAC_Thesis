import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect

"""
Critic Network Hyperparameters:
- Input: 4-channel image (grayscale, 84x84 pixels, stack of 4 frames)
- Convolutional Layer 1: 32 filters, 8x8 kernel, stride of 4
- Convolutional Layer 2: 64 filters, 4x4 kernel, stride of 2
- Convolutional Layer 3: 64 filters, 3x3 kernel
"""

"""
Actor Network Hyperparameters:
- Input: 4-channel image (grayscale, 84x84 pixels, stack of 4 frames)
- Convolutional Layer 1: 32 filters, 8x8 kernel, stride of 4
- Convolutional Layer 2: 64 filters, 4x4 kernel, stride of 2
- Convolutional Layer 3: 64 filters, 3x3 kernel
"""

class Critic(nn.Module):
    def __init__(self, n_actions):
        super(Critic, self).__init__()
        self.n_actions = n_actions
        self.conv1 = nn.Conv2d(in_channels = 4, # Input is in gray scale - Atari env 4 frames (84 x 84) - Like DQN
                               out_channels=32, 
                               kernel_size=8, 
                               stride=4)
        
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        
        # Linear Layers
        self.fc1 = nn.Linear(in_features=64*7*7,
                             out_features=512)
        
        self.fc2 = nn.Linear(in_features=512,
                             out_features=self.n_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # linear layers
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_actions, alpha=1): # fixed alpha for entmax transformation
        super(Actor, self).__init__()
        self.n_actions = n_actions
        self.alpha     = alpha
        self.conv1 = nn.Conv2d(in_channels = 4, # Input is in gray scale - Atari env 4 frames (84 x 84) - Like DQN
                               out_channels=32,
                               kernel_size=8, 
                               stride=4)
        
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)
        
        # Linear Layers
        self.fc1 = nn.Linear(in_features=64*7*7,
                             out_features=512)
        
        self.fc2 = nn.Linear(in_features=512,
                             out_features=self.n_actions)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        # linear layers
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # alpha - entmax is the optimal policy for Tsallis regularized MDP
        if self.alpha==1:
            x = F.softmax(x, dim=1)
        elif self.alpha==1.5:
            x = entmax15(x, dim=1)
        elif self.alpha==2:
            x = sparsemax(x, dim=1)
        else:
            x = entmax_bisect(x, alpha=self.alpha) 
        return x