import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import time

from buffers import Replay_Buffer
from value_networks import Actor, Critic
from atari_env import atari_env
# for debugging
from test_env import TestEnv, GridWorld
import matplotlib.pyplot as plt

'''
The RAC algorithm uses one NN to parameterize the policy and two NNs to parameterize the Q-values,
to overcome the positive bias incurred by overestimation of Q-value, which is known to yield a poor performance.
Target Q-value network are used to stabilize learning.
'''

# SPECIFY WHICH GPU TO USE: 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # will iose the first GPU
'''

class RAC:       # default method is unregularized Actor-Critic Agent
    def __init__(self,
                 log_path,             # path to directory where information regarding training is stored
                 n_actions,            # environment nummber of actions
                 discount=0.99,        # discount factor
                 lr=1e-4,              # optimizer learning rate (the same for all networks)
                 alpha=1,              # alpha-Tsallis value
                 lambd=0.01,           # lambda value (regularization weight)
                 batch_size=32,        # batch size for training the function approximators
                 num_iterations=200000,# number of environment steps
                 log_every=1000,       # log cumulative rewards and losses every _ steps
                 gradient_step=4,      # number of env steps between each gradient step
                 learning_starts=5000, # how many environment steps are taken to fill the replay buffer
                 memory_size=100000):  # replay buffer size 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount        
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer = Replay_Buffer(memory_size)
        self.learning_starts = learning_starts
        self.n_ac = n_actions
        self.alpha = alpha
        self.lambd = lambd
        self.gradient_step = gradient_step
        self.num_iterations = num_iterations
        # Critic networks
        self.critic1 = Critic(self.n_ac)
        #self.critic2 = Critic(self.n_ac) %todo make corrections to include this net
        # Target Critic networks
        self.target_critic1 = Critic(self.n_ac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        #self.target_critic2 = Critic(self.n_ac) %todo make corrections to include this net
        # Actor network
        self.actor = Actor(self.n_ac, self.alpha)
        # Adam Optimmizers for Q-Value and Policy Networks -> I added the optimizer for the target Q network, but I'm not using it in this version
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        #self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.log_every = log_every
        # Tensorboard SummaryWriter to log relevant values
        self.writer = SummaryWriter(log_dir=log_path)

    def train(self, game_name, skip=4, stack=4): # %todo I think I should add some of the atributes defined in the class instantiation in the train method
        env = atari_env(env_id=game_name, skip=skip, stack=stack)
        states = env.reset()
        
        # fill the replay buffer with an initial sample
        for _ in range(self.learning_starts):
            action = self.get_action(states).squeeze()
            next_states, rewards, dones, info = env.step(action)
            self.replay_buffer.store(transition=(states.squeeze(), action, rewards, next_states.squeeze(), dones)) # we squeeze states here, because before they had dimention (BATCH_SIZE, 1, 4, 84, 84) when sampled from replay buffer
            states = next_states

        start_time = time.time()
        states = env.reset() # initialize environment

        # tensorboard variables

        store_rewards = 0
        count_episods = 1
        policy_loss   = 0
        value_loss    = 0

        # start training
        for global_step in tqdm(range(1, self.num_iterations + 1)): # $self.gradient_step environment steps, 1 gradient step
            action = self.get_action(states).squeeze()

            next_states, rewards, dones, info = env.step(action)
            store_rewards += rewards
            # store information on buffers
            self.replay_buffer.store(transition=(states.squeeze(), action, rewards, next_states.squeeze(), dones))
            if dones: # restart game after ending
                states = env.reset()
                count_episods += 1
            else: states = next_states

            if (global_step+1) % self.gradient_step == 0: # gradient step

                    batch = self.replay_buffer.sample(self.batch_size)
                    summaries = self.perform_gradient_step(*batch) # gradient is successfully computed
                    policy_loss += summaries['p_loss']/self.log_every
                    value_loss  += summaries['v_loss']/self.log_every

            if (global_step) % self.log_every == 0:
                t = time.time() - start_time
                print(f"Save model, global_step: {global_step}, delta_time: {t}.")
                # store quantites in tensorboard
                self.writer.add_scalar('Sum_rewards', store_rewards, global_step)
                self.writer.add_scalar('Num_episodes', count_episods, global_step)
                self.writer.add_scalar('Policy_Loss', policy_loss, global_step)
                self.writer.add_scalar('Value_Loss', value_loss, global_step)
                store_rewards = 0
                count_episods = 1
                value_loss = 0
                policy_loss = 0
                start_time = time.time()

            if global_step % 2500:
                # Copy parameters from qnet to target
                self.target_critic1.load_state_dict(self.critic1.state_dict())
        
                
    def perform_gradient_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # convert numpy arrays to PyTrch tensors
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # compute Q-values and policy
        qvals = self.critic1(state_batch) # Q_{\theta} ( s_t, . )
        policy = self.actor(state_batch)  # \pi_{\psi} ( s_t, . )

        # compute phi
        phi = - torch.log(policy + 1e-8)  # \phi(\pi_{\psi} ( s_t, . ))

        # compute Q-value loss
        action_qvals = qvals.gather(1, action_batch.unsqueeze(1)).squeeze(1) # Q_{\theta} ( s_t, a_t )

        with torch.no_grad():
            next_qvals = self.target_critic1(next_state_batch) # Q_{\theta} ( s_{t+1}, . )
            next_policy = self.actor(next_state_batch)         # \pi_{\psi} ( s_{t+1} | . )

            # compute phi
            next_phi = - torch.log(next_policy + 1e-8)  # \phi(\pi_{\psi} ( s_{t+1} | . ))

            # sample actions
            next_actions = torch.multinomial(next_policy, 1).squeeze(1) # a_{t+1} ~ \pi_{\psi} ( s_{t+1}, . )

            next_action_qvals = next_qvals.gather(1, next_actions.unsqueeze(1)).squeeze(1) # Q_{\theta} ( s_{t+1}, a_{t+1} )
            next_action_phi = next_phi.gather(1, next_actions.unsqueeze(1)).squeeze(1)     # \phi(\pi_{\psi} ( s_{t+1} | a_{t+1} ))
            # defined between equations (9) and (10)
            targets = reward_batch + (1 - done_batch) * self.discount * (next_action_qvals + self.lambd * next_action_phi)

        # compute value loss (equation 9)
        v_loss = F.mse_loss(action_qvals, targets)

        # compute policy loss (equation 10)
        p_loss = torch.mean(torch.sum(policy * (- self.lambd * phi - qvals.detach()), 1))

        self.critic1_opt.zero_grad()
        self.actor_opt.zero_grad()

        p_loss.backward()
        v_loss.backward()

        self.actor_opt.step()
        self.critic1_opt.step()
        summaries = {'v_loss': v_loss.item(), 'p_loss': p_loss.item()}

        return summaries
    
    '''
    Samples action from policy directly from state observation
    '''
    # If computing action for one observation, the obs input should still have the batchsize value on index 0
    def get_action(self, obs, deterministic=False):
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