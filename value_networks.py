import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect

import numpy as np

'''
Actor-Critic Networks take as input an a (BATCH_SIZE, 4, 84, 84) sized tensor. 
-- Input Layout is different in torch and tensorflow
'''

DEBUG = False

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

# Nov 3 - Create RAC model

'''
The RAC algorithm uses one NN to parameterize the policy and two NNs to parameterize the Q-values,
to overcome the positive bias incurred by overestimation of Q-value, which is known to yield a poor performance.
Target Q-value network are used to stabilize learning.
'''

class RAC:
    def __init__(self, n_actions, discount=0.99, lr=1e-4, alpha=1, lambd=0): # default method is unregularized Deep Q-Learning Agent
        # discount factor
        self.discount = discount
        # optimizer learning rate (the same of all networks)
        self.lr = lr
        # environment nummber of actions
        self.n_ac = n_actions
        # alpha-Tsallis value
        self.alpha = alpha
        # lambda value (regularization weight)
        self.lambd = lambd
        # Critic networks
        self.critic1 = Critic(self.n_ac)
        self.critic2 = Critic(self.n_ac)
        # Target Critic networks
        self.target_critic1 = Critic(self.n_ac)
        self.target_critic2 = Critic(self.n_ac)
        # Actor network
        self.actor = Actor(self.n_ac, self.alpha)

    # If computing action for one observation, the obs input should still have the batchsize value on index 0
    def get_action(self, obs, deterministic=False):
        obs = torch.from_numpy(obs).float()
        with torch.no_grad(): pi = self.actor(obs)
        pi = pi.detach().numpy()
        batch_size = obs.shape[0]
        # choose most likely action
        if deterministic: actions = np.argmax(pi, axis=1)
        # sample action from policy
        else: actions = np.array([np.random.choice(self.n_ac, p=pi[i]) for i in range(batch_size)])
        return actions
    
    # store RAC agent weights in 'path'
    def save(self, path):
        torch.save({
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
        }, path)

    # load RAC agent weights from 'path'
    def load(self, path):
        checkpoint = torch.load(path)
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
    
if DEBUG: from IPython import embed; embed()