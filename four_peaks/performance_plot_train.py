import numpy as np
import torch
from algorithm import RAC
from buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from modes_env import Mode_Seeking_Environment
import argparse
import os

parser = argparse.ArgumentParser(description='Train RAC model')
parser.add_argument('--alpha', type=float, default=1.0, help='Value of alpha')
parser.add_argument('--lambd', type=float, default=2.0, help='Value of lambda')
parser.add_argument('--disc', type=int, default=20, help='Action discretization')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--num_iterations', type=int, default=600000, help='Number of training iterations')
parser.add_argument('--episode_length', type=int, default=1000, help='Number of iterations per episode')
args = parser.parse_args()

# Hyperparameters
alpha = args.alpha
lambd = args.lambd
action_discretization = args.disc
seed = args.seed
# Training parameters
num_iterations = args.num_iterations
num_steps_per_reset = args.episode_length

# Set random seed for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

log_dir = f'performance_runs/alpha_{alpha}_lambda_{lambd}_disc_{action_discretization}_seed_{seed}'
writer = SummaryWriter(log_dir=log_dir)

env = Mode_Seeking_Environment(peak_steepness=0.002, action_discretization=action_discretization, max_distance=5, local_region_size=(3, 3), reset_interval=1000)

obs_dim = np.prod(env.local_region_size)  # Flatten the input dimension
n_actions = env.action_space.shape[0]

rac = RAC(
    discount=0.99,
    lr=1e-4,
    alpha=alpha,
    lambd=lambd,
    batch_size=32,
    gradient_step_every=4,
    update_target_every=1000,
    learning_starts=32,
    log_every=1000,
    memory_size=10000,
    log_dir='multi_peak_experiment',
    deterministic=False,
    save_best_mdl=True
)

rac.replay_buffer = ReplayBuffer(buffer_size=rac.memory_size, batch_size=rac.batch_size, device=rac.device)

# Run training and store rewards
rewards = rac.train(env, writer, num_iterations, num_steps_per_reset, return_rewards=True)
reward_path = f'performance_rewards/rewards_alpha_{alpha}_lambda_{lambd}_disc_{action_discretization}_seed_{seed}.npy'
os.makedirs(os.path.dirname(reward_path), exist_ok=True)
np.save(reward_path, rewards, allow_pickle=True)

# Store weights for inference
actor_path = f'performance_models/actor_alpha_{alpha}_lambda_{lambd}_disc_{action_discretization}_seed_{seed}.pth'
critic_path = f'performance_models/critic_alpha_{alpha}_lambda_{lambd}_disc_{action_discretization}_seed_{seed}.pth'
rac.save_model(actor_path, critic_path)