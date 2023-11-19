import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import time

from buffers import Replay_Buffer, Results_Buffer
from value_networks import Actor, Critic
from atari_env import atari_env
# for debugging
from test_env import TestEnv
import matplotlib.pyplot as plt

# Nov 3 - Create RAC model

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

class RAC:       # default method is unregularized Deep Q-Learning Agent
    def __init__(self,
                 log_path,             # path to directory where information regarding training is stored
                 n_actions,            # environment nummber of actions
                 discount=0.99,        # discount factor
                 lr=1e-4,              # optimizer learning rate (the same for all networks)
                 alpha=1,              # alpha-Tsallis value
                 lambd=0.1,            # lambda value (regularization weight)
                 batch_size=32,        # batch size for training the function approximators
                 num_iterations=200000,# number of environment steps
                 log_every=1000,       # log cumulative rewards and losses every _ steps
                 gradient_step=4,      # number of env steps between each gradient step
                 learning_starts=200,  # how many environment steps are taken to fill the replay buffer
                 memory_size=500000):  # replay buffer size 
        
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
        # for param in self.critic1.parameters(): param.requires_grad = False
        #self.target_critic2 = Critic(self.n_ac) %todo make corrections to include this net
        # Actor network
        self.actor = Actor(self.n_ac, self.alpha)
        # Adam Optimmizers for Q-Value and Policy Networks -> I added the optimizer for the target Q network, but I'm not using it in this version
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        #self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=self.lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.log_every = log_every
        # Tensorboard SummaryWriter to log relevant values
        log_path = 'train_log'
        self.results_buffer = Results_Buffer() # % I don't think I'll use this...
        self.writer = SummaryWriter(log_dir=log_path)

    def train(self, game_name, skip=4, stack=4): # %todo I think I should add some of the atributes defined in the class instantiation in the train method
        env = atari_env(env_id=game_name, skip=skip, stack=stack)
        states = env.reset()
        
        # fill the replay buffer with an initial sample
        for _ in range(self.learning_starts):
            action = self.get_action(states)[0] 
            next_states, rewards, dones, info = env.step(action)
            #print(f'FILL REPLAY INFO = {info}') # THE GAME DOESN'T RESTART AFTER LOOSING!!!! %todo
            self.replay_buffer.store(transition=(states.squeeze(), action, rewards, next_states.squeeze(), dones)) # we squeeze states here, because before they had dimention (BATCH_SIZE, 1, 4, 84, 84) when sampled from replay buffer
            states = next_states

        start_time = time.time()
        states = env.reset() # initialize environment
        store_rewards = 0 # ERASE ME %todo YOU NEED TO IMPLEMENT THIS BETTER
        count_episods = 1
        # start training
        for global_step in tqdm(range(self.num_iterations)): # $self.gradient_step environment steps, 1 gradient step
            global_step+=1 # %todo epa, por isto como deve ser em vez de andar aqui com estas coisas
            action = self.get_action(states)[0]
            next_states, rewards, dones, info = env.step(action)
            # %todo - storing frames to understand behaviour
            # if global_step % 50 == 0: plt.imsave(f'play/frame_{global_step}.png', next_states[-1], cmap='gray')
            store_rewards += rewards
            #if dones == True: print('ACABOU E SUPOSTAMENTE NÃO DEVIA FICAR STUCK!')
            # store information on buffers
            self.replay_buffer.store(transition=(states.squeeze(), action, rewards, next_states.squeeze(), dones))
            #self.results_buffer.update_infos(info, global_step)
            if dones: 
                states = env.reset()
                count_episods += 1
            else: states = next_states

            if (global_step+1) % self.gradient_step == 0: # gradient step

                    batch = self.replay_buffer.sample(self.batch_size)
                    #from IPython import embed; embed()
                    self.perform_gradient_step(*batch) # gradient is successfully computed

            if (global_step+1) % self.log_every == 0:
                t = time.time() - start_time
                print(f"Save model, global_step: {global_step}, delta_time: {t}.")
                print(f'SUM_REWARDS: {store_rewards}')
                print(f'NUM EPISODES: {count_episods}')
                # store quantites in tensorboard
                self.writer.add_scalar('Sum_rewards', store_rewards, global_step)
                self.writer.add_scalar('Num_episodes', count_episods, global_step)
                store_rewards = 0
                count_episods = 0
                # self.results_buffer.add_summary(self.summary_writer, global_step, t)
                start_time = time.time()

            if global_step % 2500: # UPDATE TARGET EVERY %todo this is in their code but not the paper
                # Copy parameters from qnet to target
                self.target_critic1.load_state_dict(self.critic1.state_dict())
        
                
    # the gradient step is performed on the whole replay buffer
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
        phi = - torch.log(policy + 1e-8)

        # compute Q-value loss
        action_qvals = qvals.gather(1, action_batch.unsqueeze(1)).squeeze(1) # Q-values of actions taken
        action_phi = phi.gather(1, action_batch.unsqueeze(1)).squeeze(1) # phi values of actions taken
        action_policy = policy.gather(1, action_batch.unsqueeze(1)).squeeze(1) # probability of actions taken

        with torch.no_grad():
            next_qvals = self.target_critic1(next_state_batch)
            next_policy = self.actor(next_state_batch)

            next_phi = - torch.log(next_policy + 1e-8)
            next_actions = torch.multinomial(next_policy, 1).squeeze(1)
            next_action_qvals = next_qvals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_action_phi = next_phi.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            # defined between equations (9) and (10)
            targets = reward_batch + (1 - done_batch) * self.discount * (next_action_qvals + self.lambd * next_action_phi)

        # compute value loss
        v_loss = F.mse_loss(action_qvals, targets)

        # compute policy loss (equation 10)
        p_loss = torch.mean(torch.sum((- self.lambd * action_phi - action_qvals) * action_policy, dim=0))

        # zero all the gradients # had this problem: https://discuss.pytorch.org/t/multiple-loss-functions-in-a-model/111464/14
        # because action_qvals is used on two loss functions
        self.critic1_opt.zero_grad()
        self.actor_opt.zero_grad()

        v_loss.backward(retain_graph=True) # %todo maybe it's better to clone tensor than keep the computational graph
        p_loss.backward()

        self.critic1_opt.step()
        self.actor_opt.step()
        summaries = {'v_loss': v_loss.item(), 'p_loss': p_loss.item()}
        #from IPython import embed; embed()
        return v_loss.item(), p_loss.item()
    '''
    Samples action from policy directly from state observation
    '''
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

'''
from IPython import embed; embed()

import matplotlib.pyplot as plt

agent = RAC('', 4, num_iterations=200000)
agent.train('BreakoutNoFrameskip-v4')

env = atari_env('BreakoutNoFrameskip-v4')

def move(obs, env):
    import matplotlib.pyplot as plt
    action = agent.get_action(obs)
    next_states, rewards, dones, info = env.step(action[0])
    with torch.no_grad(): 
        policy = agent.actor(torch.tensor(obs).float())
    print(f'Policy: {policy}')
    plt.imshow(next_states[-1], cmap='gray')
    plt.show()
    return next_states

from IPython import embed; embed()
'''