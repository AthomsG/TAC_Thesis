import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect

import numpy as np

from replay_buffer import Replay_Buffer
from atari_env import atari_env

'''
Actor-Critic Networks take as input an a (BATCH_SIZE, 4, 84, 84) sized tensor. 
-- Input Layout is different in torch and tensorflow
'''

DEBUG = True

if DEBUG: torch.autograd.set_detect_anomaly(True)

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

# SPECIFY WHICH GPU TO USE: 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
'''

class RAC:       # default method is unregularized Deep Q-Learning Agent
    def __init__(self,
                 log_path,            # path to directory where information regarding training is stored
                 n_actions,           # environment nummber of actions
                 discount=0.99,       # discount factor
                 lr=1e-4,             # optimizer learning rate (the same for all networks)
                 alpha=1,             # alpha-Tsallis value
                 lambd=0,             # lambda value (regularization weight)
                 batch_size=32,       # batch size for training the function approximators
                 gradient_step=4,     # number of env steps between each gradient step
                 learning_starts=200, # how many environment steps are taken to fill the replay buffer
                 memory_size=500000): # replay buffer size 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount        
        self.lr = lr
        self.replay_buffer = Replay_Buffer(memory_size)
        self.learning_starts = learning_starts
        self.n_ac = n_actions
        self.alpha = alpha
        self.lambd = lambd
        self.gradient_step = gradient_step
        # Critic networks
        self.critic1 = Critic(self.n_ac)
        self.critic2 = Critic(self.n_ac)
        # Target Critic networks
        self.target_critic1 = Critic(self.n_ac)
        self.target_critic2 = Critic(self.n_ac)
        # Actor network
        self.actor = Actor(self.n_ac, self.alpha)
        # Adam Optimmizers for Q-Value and Policy Networks -> I added the optimizer for the target Q network, but I'm not using it in this version
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        
        # environment interaction step
        self.global_step = 0

    def train(self, game_name, skip=4, stack=4):
        env = atari_env(env_id=game_name, skip=skip, stack=stack)
        states = env.reset().reshape(1, 4, 84, 84) # we have to ALWAYS include the batch size for 'get_action' method
        
        for i in range(self.learning_starts): # fill replay buffer
            action = self.get_action(states)[0] 
            next_states, rewards, dones, info = env.step(action)
            self.replay_buffer.store(transition=(states.reshape(4, 84, 84), action, rewards, next_states.reshape(4, 84, 84), dones))
            states = next_states.reshape(1, 4, 84, 84)
            self.global_step+=1 # %todo SHOULD I INCREMENT THIS HERE?

        # start training

    def perform_gradient_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # convert numpy arrays to PyTrch tensors
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch.astype(np.uint8)).float().to(self.device)

        # compute Q-values and policy
        qvals = self.critic1(state_batch)
        policy = self.actor(state_batch)

        # compute phi
        phi = - torch.log(policy + 1e-6)

        # compute Q-value loss
        action_qvals = qvals.gather(1, action_batch.unsqueeze(1)).squeeze(1) # Q-values of actions taken
        action_phi = phi.gather(1, action_batch.unsqueeze(1)).squeeze(1) # phi values of actions taken
        action_policy = policy.gather(1, action_batch.unsqueeze(1)).squeeze(1) # probability of actions taken

        with torch.no_grad():
            next_qvals = self.target_critic1(next_state_batch)
            next_policy = self.actor(next_state_batch)

            next_phi = - torch.log(next_policy + 1e-6)
            next_actions = torch.multinomial(next_policy, 1).squeeze(1)
            next_action_qvals = next_qvals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_action_phi = next_phi.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            # defined between equations (9) and (10)
            targets = reward_batch + (1 - done_batch) * self.discount * (next_action_qvals + self.lambd * next_action_phi)

        # compute value loss
        v_loss = F.mse_loss(action_qvals, targets)

        # compute policy loss
        p_loss = torch.mean(torch.sum((- self.lambd * action_phi - action_qvals) * action_policy, dim=0))

        # zero all the gradients # had this problem: https://discuss.pytorch.org/t/multiple-loss-functions-in-a-model/111464/14
        self.critic1_opt.zero_grad()
        self.actor_opt.zero_grad()

        v_loss.backward(retain_graph=True)
        p_loss.backward()

        self.critic1_opt.step()
        self.actor_opt.step()

        return v_loss.item(), p_loss.item()

    # If computing action for one observation, the obs input should still have the batchsize value on index 0
    def get_action(self, obs, deterministic=False): # I SHOULD IMPLEMENT THIS WITH TORCH MULTINOMIAL METHOD!
        obs = torch.from_numpy(obs).float().to(self.device)
        with torch.no_grad(): pi = self.actor(obs)
        # choose most likely action
        if deterministic: actions = torch.argmax(pi, dim=1)
        # sample action from policy
        else: actions = torch.multinomial(pi, num_samples=1)
        
        return actions.squeeze(1).cpu().numpy() # Isto faz sentido aqui?
    
    # store RAC agent weights in 'path'
    def save(self, path=None):
        if not path: path=self.log_path
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