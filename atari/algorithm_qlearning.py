import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import time
import numpy as np

from summary import Summary, mean_wasserstein_distance
from buffers import ReplayBuffer
from value_networks import Actor, Critic
from environment import atari_env

class RAC:       
    def __init__(self,
                 discount,              # discount factor
                 lr,                    # optimizer learning rate (the same for all networks)
                 alpha,                 # alpha-Tsallis value
                 lambd,                 # lambda value (regularization temperature)
                 batch_size,            # batch size for training the function approximators
                 num_iterations,        # number of environment steps
                 gradient_step_every,   # number of env steps between each gradient step
                 update_target_every,   # number of steps between each hard update 
                 learning_starts,       # how many environment steps are taken to fill the replay buffer
                 log_every,             # log tensorboard stored quantities every... gradient steps
                 memory_size,           # replay buffer size
                 env_id,                # environment name
                 log_dir,               # directory where tensorboard information is stored
                 deterministic = False  # if true, take policy argmax. Else, sample from policy
                 ):
        
        self.log_every = log_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount        
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=memory_size, batch_size=self.batch_size, device=self.device)
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.lambd = lambd
        self.gradient_step_every = gradient_step_every
        self.update_target_every = update_target_every
        self.num_iterations = num_iterations
        self.env_id = env_id
        self.log_dir = log_dir
        self.deterministic = deterministic
        # gradient Clipping
        self.clip_grad_param = 1
        # polyak Averaging constant
        self.tau = 1e-2

    def __init_networks__(self):
        # Critic network
        self.critic1 = Critic(self.n_actions).to(self.device)
        # Target Critic network
        self.target_critic1 = Critic(self.n_actions).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        # Actor network
        self.actor = Actor(self.n_actions, self.alpha).to(self.device)
        # Network's optimizers
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    # -----------Tsallis---------- # 

    def log_alpha(self, pi): # %todo is there another solution to numerical problems?
        # Numerical issues arise (due to autograd) for sparse policies when alpha < 2 
        eps = torch.any(pi==0)
        eps = eps.float() * 1e-10
        if self.alpha == 1:
            return torch.log(pi + eps)
        else:
            return (torch.pow(pi + eps, self.alpha - 1) - 1)/(self.alpha - 1)

    def Tsallis_Entropy(self, pi):
            return - pi * self.log_alpha(pi)/self.alpha
    
    # ----------------------------- # 

    def train(self, verbose=True):
        self.env = atari_env(env_id=self.env_id, skip=4, stack=4)

        state = self.env.reset() # initialize environment
        self.n_actions = self.env.action_space.n # get action set cardinality

        # tensorboard SummaryWriter wrapper to monitor useful metrics
        self.summary = Summary(log_dir="tensorboard_logs_q/{}/".format(self.env_id[:-14]) + self.log_dir,
                               n_actions=self.n_actions)

        self.__init_networks__()  # initialize parameterized policy and Q-value functions

        # fill the replay buffer with an initial samples
        if verbose:
            range_func = tqdm(range(self.learning_starts), desc='Filling Replay Buffer with experience!')
        else:
            range_func = range(self.learning_starts)

        for _ in range_func:
            next_state, reward, done, info = self.perform_environment_step(state, deterministic=self.deterministic)
            if done: state = self.env.reset() # reset environment if episode has ended
            else: state = next_state

        self.summary.reset() # initialize summary for monitoring useful metrics

        state = self.env.reset()     # reset environment
        start = time.time()          # start timer

        # start training
        if verbose:
            range_func = tqdm(range(1, self.num_iterations + 1))
        else:
            range_func = range(1, self.num_iterations + 1)

        for environment_step in range_func: # $self.gradient_step_every environment steps, 1 gradient step
            # environment step
            next_state, reward, done, info = self.perform_environment_step(state, deterministic=self.deterministic)
            if done: state = self.env.reset()
            else: state = next_state

            # gradient step
            if (environment_step) % (self.gradient_step_every) == 0: # take gradient step
                batch = self.replay_buffer.sample()
                self.perform_gradient_step(*batch)

                HARD_UPDATE = True
                if HARD_UPDATE:
                    # hard update target network every update_target_every gradient steps
                    if environment_step % (self.update_target_every * self.gradient_step_every) == 0:
                        self.hard_update()
                else:
                    self.soft_update()

                # log onto tensorboard
                if (environment_step) % (self.log_every * self.gradient_step_every) == 0:
                    gradient_step = environment_step/self.gradient_step_every
                    self.summary.log_to_tensorboard(gradient_step=gradient_step)

                    if not verbose: # print onto .log
                        print("Save model, global_step: {}, delta_time: {}.".format(
                            environment_step, time.time() - start))
                        start = time.time()

    def perform_environment_step(self, state, deterministic=False):
        with torch.no_grad(): pi = self.actor(torch.tensor(state).unsqueeze(0).to(self.device))
        # monitor sparsity
        sparse_actions = (pi==0).sum().item()
        self.summary.update('sparsity', value=sparse_actions, count=self.n_actions-1)
        if deterministic:
            action = torch.argmax(pi).item()
        else:
            action = Categorical(pi).sample().item()
        next_state, reward, done, info = self.env.step(action)
        self.replay_buffer.add(state, action, reward, next_state, done)

        #  -- store monitored quantities -- 

        # update running sums and counts
        self.summary.update('reward', reward, count=done) # only add to count if end of episode
        self.summary.update('tsallis_entropy', self.Tsallis_Entropy(pi).sum().item())
        self.summary.update('alpha_2_entropy', (- pi * (pi - 1)/2).sum().item())

        # update action probabilities
        action_probs = pi.squeeze().tolist()
        for i, prob in enumerate(action_probs):
            self.summary.update(f'action_probs_{i}', prob)

        return next_state, reward, done, info

    def perform_gradient_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # compute Q-values and policy
        qvals  = self.critic1(state_batch).squeeze() # Q_{\theta} ( s_t, . )
        policy = self.actor(state_batch).squeeze()   # \pi_{\psi} ( s_t, . )

        # monitor sparsity -- should I monitor here?
        # sparse_actions = (policy==0).sum().item()
        # self.summary.update('sparsity', value=sparse_actions, count=self.batch_size * self.n_actions)

        # compute Q-value loss
        if self.batch_size == 1: qvals = qvals.unsqueeze(0)

        with torch.no_grad(): # The value loss is not being computed properly
            next_qvals = self.target_critic1(next_state_batch).squeeze() # Q_{\theta} ( s_{t+1}, . )
            next_policy = self.actor(next_state_batch).squeeze()         # \pi_{\psi} ( s_{t+1} | . )
            if self.batch_size == 1: 
                next_qvals  = next_qvals.unsqueeze(0)
                next_policy = next_policy.unsqueeze(0)
            # compute log-alpha
            next_log_alpha = self.log_alpha(next_policy)  # \phi(\pi_{\psi} ( s_{t+1} | . ))
            # sample next actions
            next_actions = torch.multinomial(next_policy, 1).squeeze(1) # a_{t+1} ~ \pi_{\psi} ( s_{t+1}, . )

            next_action_qvals = next_qvals.gather(1, next_actions.unsqueeze(1)).squeeze(1) # Q_{\theta} ( s_{t+1}, a_{t+1} )
            next_action_log_alpha = next_log_alpha.gather(1, next_actions.unsqueeze(1)).squeeze(1)     # \phi(\pi_{\psi} ( s_{t+1} | a_{t+1} ))

            # defined between equations (9) and (10)
            targets = reward_batch.squeeze(1) + (1 - done_batch.squeeze(1)) * self.discount * (next_action_qvals + self.lambd * next_action_log_alpha)

        action_qvals = qvals.gather(1, action_batch).squeeze(1)
        # compute value loss (equation 9)
        v_loss = F.mse_loss(action_qvals, targets)/2

        # compute policy loss (equation 10)
        linear_term  = policy * qvals.detach()
        entropy_term = self.Tsallis_Entropy(policy)
        p_loss = -(linear_term + entropy_term).sum(1).mean()

        # update Critic
        self.critic1_opt.zero_grad()
        v_loss.backward()
        critic_norm = clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_opt.step()

        # update Actor
        self.actor_opt.zero_grad()
        p_loss.backward()
        actor_norm = clip_grad_norm_(self.actor.parameters(), self.clip_grad_param)
        self.actor_opt.step()

        #  -- store monitored quantities --

        # update summary of running sums and counts
        self.summary.update('v_loss', v_loss.item())
        self.summary.update('p_loss', p_loss.item())
        self.summary.update('wsn_distance', mean_wasserstein_distance(policy, self.actor(state_batch).squeeze()))
        # update actor_grad and critic_grad (we're only monitoring for the max value)
        self.summary.update('max_actor_grad', actor_norm.item())
        self.summary.update('max_critic_grad', critic_norm.item())
        
    # -- Target Update -- 
    
    def hard_update(self): 
        """Hard update model parameters.
        θ_target = θ_local
        """
        self.target_critic1.load_state_dict(self.critic1.state_dict())

    def soft_update(self):
        """Soft update model parameters through Polyak averaging.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    # -- Save Function Approximators --
            
    def save_networks(self, path):
        """Save the state dictionaries of the networks to the specified path."""
        torch.save({
            'critic1_state_dict': self.critic1.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
        }, path)