import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

from tqdm import tqdm
import numpy as np
import time

from buffers import Replay_Buffer
from value_networks import Actor, Critic, Actor_Tensorflow
from atari_env import atari_env

'''
The RAC algorithm uses one NN to parameterize the policy and two NNs to parameterize the Q-values,
to overcome the positive bias incurred by overestimation of Q-value, which is known to yield a poor performance.
Target Q-value network are used to stabilize learning.
'''

# SPECIFY WHICH GPU TO USE: 
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # will use the first GPU
'''

class RAC:       # default method is unregularized Actor-Critic Agent
    def __init__(self,
                 log_path,              # path to directory where information regarding training is stored
                 discount=0.99,         # discount factor
                 lr=1e-4,               # optimizer learning rate (the same for all networks)
                 alpha=1,               # alpha-Tsallis value
                 lambd=0.01,            # lambda value (regularization weight)
                 batch_size=32,         # batch size for training the function approximators
                 num_iterations=200000, # number of environment steps
                 log_every=1000,        # log cumulative rewards and losses every _ gradient steps
                 gradient_step_every=4, # number of env steps between each gradient step
                 learning_starts=5000,  # how many environment steps are taken to fill the replay buffer
                 memory_size=100000     # replay buffer size 
                 ):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount        
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer = Replay_Buffer(memory_size)
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.lambd = lambd
        self.gradient_step_every = gradient_step_every
        self.num_iterations = num_iterations
        # Tensorboard SummaryWriter to log relevant values
        self.writer = SummaryWriter(log_dir=log_path)
        self.log_every = log_every

    def __init_networks__(self):
        # Critic network
        self.critic1 = Critic(self.n_ac)
        # Target Critic network
        self.target_critic1 = Critic(self.n_ac)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        # Actor network
        # self.actor = Actor_Tensorflow(self.n_ac, self.alpha)
        self.actor = Actor(self.n_ac, self.alpha)
        # Network's optimizers
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def train(self, game_name, skip=4, stack=4, verbose=True): # %todo I think I should add some of the atributes defined in the class instantiation in the train method
        env = atari_env(env_id=game_name, skip=skip, stack=stack)
        states = env.reset() # initialize environment
        self.n_ac = env.action_space.n # get actions space cardinality 
        
        self.__init_networks__() # initialize parameterized policy and Q-value functions

        # fill the replay buffer with an initial samples
        if verbose:
            range_func = tqdm(range(self.learning_starts), desc='Filling Replay Buffer with experience!')
        else:
            range_func = range(self.learning_starts)

        for _ in range_func:
            action = self.get_action(states).squeeze()
            next_states, rewards, dones, info = env.step(action)
            self.replay_buffer.store(transition=(states.squeeze(), action, rewards, next_states.squeeze(), dones)) # we squeeze states here, because before they had dimention (BATCH_SIZE, 1, 4, 84, 84) when sampled from replay buffer
            if dones: states = env.reset() # reset environment if episode has ended
            else: states = next_states

        start_time = time.time()
        states = env.reset()     # reset environment

        # tensorboard quantities
        store_rewards = 0
        count_episods = 1
        policy_loss   = 0
        value_loss    = 0

        # start training
        if verbose:
            range_func = tqdm(range(1, self.num_iterations + 1))
        else:
            range_func = range(1, self.num_iterations + 1)

        for environment_step in range_func: # $self.gradient_step_every environment steps, 1 gradient step
            # environment step
            action = self.get_action(states).squeeze()
            next_states, rewards, dones, info = env.step(action)
            store_rewards += rewards
            # store transitions in replay buffer
            self.replay_buffer.store(transition=(states.squeeze(), action, rewards, next_states.squeeze(), dones))
            if dones: # restart game after ending
                states = env.reset()
                count_episods += 1
            else: states = next_states

            if (environment_step) % self.gradient_step_every == 0: # take gradient step
                batch = self.replay_buffer.sample(self.batch_size)
                summaries = self.perform_gradient_step(*batch)
                # running mean lossesse
                policy_loss += summaries['p_loss']/self.log_every
                value_loss  += summaries['v_loss']/self.log_every

                # log every $self.log_avery gradient steps
                if (environment_step) % (self.log_every * self.gradient_step_every) == 0: # log results onto tensorboard
                    t = time.time() - start_time
                    print("environment_step: {}, delta_time: {}.".format(environment_step, t))
                    # store quantites in tensorboard
                    gradient_step = environment_step/self.gradient_step_every
                    self.writer.add_scalar('reward', store_rewards/count_episods, gradient_step)
                    self.writer.add_scalar('length', 4 * self.log_every/count_episods, gradient_step) # 4 frames 
                    self.writer.add_scalar('p_loss', policy_loss, gradient_step)
                    self.writer.add_scalar('v_loss', value_loss, gradient_step)
                    self.writer.add_scalar('n_episodes', count_episods, gradient_step)

                    # assessing numerical problems in the policy:
                    self.writer.add_histogram('actions', self.last_actions, gradient_step)

                    # reset quantities
                    store_rewards = 0
                    count_episods = 1
                    value_loss = 0
                    policy_loss = 0
                    start_time = time.time()

                # update target network every 2500 gradient steps
                if environment_step % (2500 * self.gradient_step_every):
                    # Copy parameters from critic to target critic
                    self.target_critic1.load_state_dict(self.critic1.state_dict())

    # phi is the q-log function
    def compute_phi(self, policy):
        if self.alpha == 1:
            return - torch.log(policy)
        else: 
            return - (policy**(self.alpha - 1) - 1)/(self.alpha * (self.alpha - 1))

    def perform_gradient_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # convert numpy arrays to PyTroch tensors
        state_batch = torch.from_numpy(state_batch).float().to(self.device)
        action_batch = torch.from_numpy(action_batch).long().to(self.device)
        reward_batch = torch.from_numpy(reward_batch).float().to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).float().to(self.device)
        done_batch = torch.from_numpy(done_batch).float().to(self.device)

        # compute Q-values and policy
        qvals = self.critic1(state_batch) # Q_{\theta} ( s_t, . )
        policy = self.actor(state_batch)  # \pi_{\psi} ( s_t, . )

        # compute phi
        phi = self.compute_phi(policy)  # \phi(\pi_{\psi} ( s_t, . ))

        # compute Q-value loss
        action_qvals = qvals.gather(1, action_batch.unsqueeze(1)).squeeze(1) # Q_{\theta} ( s_t, a_t )

        with torch.no_grad(): # target critic has fixed parameters
            next_qvals = self.target_critic1(next_state_batch) # Q_{\theta} ( s_{t+1}, . )
            next_policy = self.actor(next_state_batch)         # \pi_{\psi} ( s_{t+1} | . )

            # compute phi
            next_phi = self.compute_phi(next_policy)  # \phi(\pi_{\psi} ( s_{t+1} | . ))

            # sample actions
            next_actions = torch.multinomial(next_policy, 1).squeeze(1) # a_{t+1} ~ \pi_{\psi} ( s_{t+1}, . )

            next_action_qvals = next_qvals.gather(1, next_actions.unsqueeze(1)).squeeze(1) # Q_{\theta} ( s_{t+1}, a_{t+1} )
            next_action_phi = next_phi.gather(1, next_actions.unsqueeze(1)).squeeze(1)     # \phi(\pi_{\psi} ( s_{t+1} | a_{t+1} ))

            # defined between equations (9) and (10)
            targets = reward_batch + (1 - done_batch) * self.discount * (next_action_qvals + self.lambd * next_action_phi)

        # compute value loss (equation 9)
        v_loss = 0.5 * F.mse_loss(action_qvals, targets)

        # compute policy loss (equation 10)
        p_loss = torch.mean(torch.sum((- self.lambd * phi - qvals.detach()) * policy, dim=1))

        # optimize Critic
        self.critic1_opt.zero_grad()
        v_loss.backward()
        self.critic1_opt.step()

        # optimize Actor
        self.actor_opt.zero_grad()
        p_loss.backward()
        self.actor_opt.step()

        summaries = {'v_loss': v_loss.item(), 'p_loss': p_loss.item()}

        # to store histogram of last sampled actions on the gradient step
        self.last_actions = next_actions

        return summaries
    
    '''
    Samples action from policy directly from state observation
    '''
    # if computing action for one observation, the obs input should still have the batchsize value on index 0
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

    def print_hyperparameters(self):
        print("Hyperparameters:")
        for attr in vars(self):
            print(f"{attr}: {getattr(self, attr)}")

    # 
    def soft_update(self, local_model , target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)