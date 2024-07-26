import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from buffers import ReplayBuffer
from value_networks import Actor, Critic
from tqdm import tqdm

class RAC:
    def __init__(self,
                 discount,
                 lr,
                 alpha,
                 lambd,
                 batch_size,
                 gradient_step_every,
                 update_target_every,
                 learning_starts,
                 log_every,
                 memory_size,
                 log_dir,
                 deterministic=False,
                 save_best_mdl=True):
        
        self.log_every = log_every
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discount = discount
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.replay_buffer = ReplayBuffer(buffer_size=self.memory_size, batch_size=self.batch_size, device=self.device)
        self.learning_starts = learning_starts
        self.alpha = alpha
        self.lambd = lambd
        self.gradient_step_every = gradient_step_every
        self.update_target_every = update_target_every
        self.log_dir = log_dir
        self.deterministic = deterministic
        self.clip_grad_param = 1
        self.tau = 1e-2
        self.save_best_mdl = save_best_mdl
        self.best_reward = -float('inf')

    def __init_networks__(self, obs_dim, n_actions):
        self.critic1 = Critic(obs_dim, n_actions).to(self.device)
        self.target_critic1 = Critic(obs_dim, n_actions).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.actor = Actor(obs_dim, n_actions, self.alpha).to(self.device)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    def log_alpha(self, policy):
        eps = torch.any(policy == 0).float() * 1e-10
        if self.alpha == 1:
            return torch.log(policy + eps)
        else:
            return (torch.pow(policy + eps, self.alpha - 1) - 1) / (self.alpha - 1)

    def Tsallis_Entropy(self, policy):
        return -policy * self.log_alpha(policy) / self.alpha

    def train(self, env, writer, num_iterations, num_steps_per_reset, return_rewards=False):
        state = env.reset()
        self.n_actions = env.action_space.shape[0]
        obs_dim = state.flatten().shape[0]

        self.__init_networks__(obs_dim, self.n_actions)

        episode_rewards = []
        global_step = 0
        total_reward = 0

        for iteration in tqdm(range(num_iterations)):
            action, log_prob = self.get_action(state)
            next_state, reward, done = env.step(action)
            self.replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            global_step += 1

            if len(self.replay_buffer) >= self.batch_size:
                if global_step % self.gradient_step_every == 0:
                    batch = self.replay_buffer.sample()
                    v_loss, p_loss = self.perform_gradient_step(*batch)

                if global_step % self.update_target_every == 0:
                    self.hard_update()

            if global_step % num_steps_per_reset == 0:
                state = env.reset()
                writer.add_scalar('Reward', total_reward, global_step)
                episode_rewards.append(total_reward)
                total_reward = 0

            if global_step % self.log_every == 0 and len(self.replay_buffer) >= self.batch_size:
                writer.add_scalar('Loss', v_loss + p_loss, global_step)

        writer.close()
        if return_rewards:
            return episode_rewards

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy = self.actor(state)
        dist = Categorical(policy)
        if self.deterministic:
            action = dist.probs.argmax().item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action)).item()
        return action, log_prob

    def perform_gradient_step(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        qvals = self.critic1(state_batch)
        policy = self.actor(state_batch)

        with torch.no_grad():
            next_qvals = self.target_critic1(next_state_batch)
            next_policy = self.actor(next_state_batch)
            next_log_alpha = self.log_alpha(next_policy)
            targets = reward_batch + (1 - done_batch) * self.discount * (next_policy * (next_qvals - self.lambd * next_log_alpha / self.alpha)).sum(1, keepdim=True)

        action_batch = action_batch.unsqueeze(1).long() if action_batch.dim() == 1 else action_batch.long()

        action_qvals = qvals.gather(1, action_batch)

        v_loss = F.mse_loss(action_qvals, targets) / 2

        linear_term = policy * qvals.detach()
        entropy_term = self.Tsallis_Entropy(policy)
        p_loss = -(linear_term + entropy_term).sum(1).mean()

        self.critic1_opt.zero_grad()
        v_loss.backward()
        clip_grad_norm_(self.critic1.parameters(), self.clip_grad_param)
        self.critic1_opt.step()

        self.actor_opt.zero_grad()
        p_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.clip_grad_param)
        self.actor_opt.step()

        return v_loss.item(), p_loss.item()
    
    def save_model(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic_path)
        print(f"Models saved: Actor -> {actor_path}, Critic -> {critic_path}")

    def load_model(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic1.load_state_dict(torch.load(critic_path))
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        print(f"Models loaded: Actor -> {actor_path}, Critic -> {critic_path}")

    def hard_update(self):
        self.target_critic1.load_state_dict(self.critic1.state_dict())