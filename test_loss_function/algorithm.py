import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from buffers import ReplayBuffer
from value_networks import Actor, Critic
from environment import test_env

class RAC:       # default method is unregularized Actor-Critic Agent
    def __init__(self,
                 discount=1,         # discount factor
                 lr=5e-1,               # optimizer learning rate (the same for all networks)
                 alpha=1,               # alpha-Tsallis value
                 lambd=0.1,            # lambda value (regularization temperature)
                 batch_size=1,         # batch size for training the function approximators
                 num_iterations=2000, # number of environment steps
                 gradient_step_every=1, # number of env steps between each gradient step
                 learning_starts=1,  # how many environment steps are taken to fill the replay buffer
                 memory_size=1,    # replay buffer size
                 n_actions=2):
        
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount        
        self.lr = lr
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=memory_size, batch_size=self.batch_size, device=self.device)
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.lambd = lambd
        self.gradient_step_every = gradient_step_every
        self.num_iterations = num_iterations

    def __init_networks__(self):
        # Critic network
        self.critic1 = Critic(self.n_actions)
        # Target Critic network
        self.target_critic1 = Critic(self.n_actions)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        # Actor network
        self.actor = Actor(self.n_actions, self.alpha)
        # Network's optimizers
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def print_network_parameters(self):
        print("Actor Network Parameters:")
        for name, param in self.actor.state_dict().items():
            print(name, param)

        print("\nCritic Network Parameters:")
        for name, param in self.critic1.state_dict().items():
            print(name, param)

        print("\nTarget Critic Network Parameters:")
        for name, param in self.target_critic1.state_dict().items():
            print(name, param)

    def save_network_parameters(self, filename):
        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic1.state_dict()
        target_critic_state_dict = self.target_critic1.state_dict()

        for name, param in actor_state_dict.items():
            np.savetxt(f'{filename}_actor_{name}.txt', param.cpu().detach().numpy())

        for name, param in critic_state_dict.items():
            np.savetxt(f'{filename}_critic_{name}.txt', param.cpu().detach().numpy())

        for name, param in target_critic_state_dict.items():
            np.savetxt(f'{filename}_target_critic_{name}.txt', param.cpu().detach().numpy())

    def train(self, verbose=True): # %todo I think I should add some of the atributes defined in the class instantiation in the train method
        self.__init_networks__() # initialize parameterized policy and Q-value functions
        # self.print_network_parameters()
        # self.save_network_parameters('weights/simple')
    
        env = test_env(self.n_actions)
        state = env.reset() # initialize environment

        summaries = {'v_loss':list(), 'p_loss':list(), 'reward':list()}

        # fill the replay buffer with an initial samples
        if verbose:
            range_func = tqdm(range(self.learning_starts), desc='Filling Replay Buffer with experience!')
        else:
            range_func = range(self.learning_starts)

        for _ in range_func:
            with torch.no_grad(): pi = self.actor(torch.tensor(state).float().unsqueeze(0))
            action = torch.argmax(pi)
            next_state, reward, done, info = env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done) # we squeeze state here, because before they had dimention (BATCH_SIZE, 1, 4, 84, 84) when sampled from replay buffer
            if done: state = env.reset() # reset environment if episode has ended
            else: state = next_state

        state = env.reset()     # reset environment

        # start training
        if verbose:
            range_func = tqdm(range(1, self.num_iterations + 1))
        else:
            range_func = range(1, self.num_iterations + 1)

        for environment_step in range_func: # $self.gradient_step_every environment steps, 1 gradient step
            # environment step
            with torch.no_grad(): pi = self.actor(torch.tensor(state).float().unsqueeze(0))
            action = torch.argmax(pi, dim=1)
            next_state, reward, done, info = env.step(action)
            # store transitions in replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            state = next_state

            if (environment_step) % self.gradient_step_every == 0: # take gradient step
                batch = self.replay_buffer.sample()
                summary = self.perform_gradient_step(*batch)
                for key in summary:
                    summaries[key].append(summary[key])

        from IPython import embed; embed()
                            
    def log_alpha(self, pi):
        if self.alpha == 1:
            return torch.log(pi)
        else:
            return (torch.pow(pi, self.alpha-1)-1)/(self.alpha * (self.alpha-1))

    def Tsallis_Entropy(self, pi):
            return - pi * self.log_alpha(pi)

    def perform_gradient_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        # compute Q-values and policy
        qvals  = self.critic1(state_batch).squeeze() # Q_{\theta} ( s_t, . )
        policy = self.actor(state_batch).squeeze()   # \pi_{\psi} ( s_t, . )

        # compute phi
        log_alpha = self.log_alpha(policy)  # \phi(\pi_{\psi} ( s_t, . ))
        # compute Q-value loss
        if self.batch_size == 1: qvals = qvals.unsqueeze(0)
        action_qvals = qvals.gather(1, action_batch) # Q_{\theta} ( s_t, a_t )

        with torch.no_grad(): # target critic has fixed parameters
            next_qvals = self.target_critic1(next_state_batch).squeeze() # Q_{\theta} ( s_{t+1}, . )
            next_policy = self.actor(next_state_batch).squeeze()         # \pi_{\psi} ( s_{t+1} | . )
            if self.batch_size == 1: 
                next_qvals  = next_qvals.unsqueeze(0)
                next_policy = next_policy.unsqueeze(0)

            # compute phi
            next_log_alpha = self.log_alpha(next_policy)  # \phi(\pi_{\psi} ( s_{t+1} | . ))
            # sample actions
            next_actions = torch.argmax(next_policy, dim=1).reshape(-1, 1) # a_{t+1} ~ \pi_{\psi} ( s_{t+1}, . )
            next_action_qvals = next_qvals.gather(1, next_actions) # Q_{\theta} ( s_{t+1}, a_{t+1} )
            next_action_log_alpha = next_log_alpha.gather(1, next_actions)    # \phi(\pi_{\psi} ( s_{t+1} | a_{t+1} ))
            # defined between equations (9) and (10)
            targets = reward_batch + (1 - done_batch) * self.discount * (next_action_qvals - self.lambd * next_action_log_alpha)

        # compute value loss (equation 9)
        v_loss = F.mse_loss(action_qvals, targets)/2
        # compute policy loss (equation 10)
        p_loss = (policy * (self.lambd * log_alpha/self.alpha - qvals.detach())).sum().mean()
        # update Critic
        self.critic1_opt.zero_grad()
        v_loss.backward()
        # print critic gradients
        print('\n\nCritic Gradients:\n\n')
        for name, param in self.critic1.named_parameters():
            if param.requires_grad:
                print(name, param.grad)
        self.critic1_opt.step()

        # update Actor - The error comes from the policy
        self.actor_opt.zero_grad()
        p_loss.backward()
        # print actor gradients
        print('\n\nActor Gradients:\n\n')
        for name, param in self.actor.named_parameters():
            if param.requires_grad:
                print(name, param.grad.numpy())
        self.actor_opt.step()

        summaries = {'v_loss': v_loss.item(), 'p_loss': p_loss.item()}
        return summaries