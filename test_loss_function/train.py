import argparse
from algorithm import RAC
import torch

# for debugging the exploding gradient
# torch.autograd.set_detect_anomaly(True)

# argument parser (makes it easier to run several experiments simultaneously)
parser = argparse.ArgumentParser(description='Train RAC agent')

parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
parser.add_argument('--learning_starts', type=int, default=1, help='Number of steps before learning starts')
parser.add_argument('--memory_size', type=int, default=1, help='Size of the replay memory')
parser.add_argument('--num_iterations', type=int, default=2000, help='Number of iterations for training')
parser.add_argument('--lr', type=float, default=5e-1, help='Learning rate')
parser.add_argument('--lambd', type=float, default=0.1, help='Temperature Parameter')
parser.add_argument('--alpha', type=float, default=1, help='Alpha value of Alpha-Tsallis entropy') # Default is Softmax Policy
parser.add_argument('--discount', type=float, default=1, help='Discount factor for the MDP')
parser.add_argument('--game', type=str, default='BreakoutNoFrameskip-v4', help='Name of the game')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
parser.add_argument('--show_progress', type=bool, default=False, help='Show progress bar and expected time to simulation completion')

args = parser.parse_args()

# set seeds for reproducibility
import torch; torch.manual_seed(args.seed)
import numpy as np; np.random.seed(args.seed)
import random; random.seed(args.seed)

# experiment setup
agent = RAC(batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            memory_size=args.memory_size,
            num_iterations=args.num_iterations,
            lr=args.lr,
            lambd=args.lambd,
            alpha=args.alpha,
            discount=args.discount)

print(f'Environemnt name: {args.game}')

# run experiment
agent.train(verbose=args.show_progress)