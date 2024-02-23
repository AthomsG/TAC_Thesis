import argparse
from algorithm import RAC
import torch
        
# argument parser (makes it easier to run several experiments simultaneously)
parser = argparse.ArgumentParser(description='Train RAC agent')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_starts', type=int, default=5000, help='Number of steps before learning starts')
parser.add_argument('--memory_size', type=int, default=1000, help='Size of the replay memory')
parser.add_argument('--num_iterations', type=int, default=1000, help='Number of iterations for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lambd', type=float, default=1, help='Temperature Parameter')
parser.add_argument('--alpha', type=float, default=1, help='Alpha value of Alpha-Tsallis entropy') # Default is Softmax Policy
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor for the MDP')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
parser.add_argument('--gradient_step_every', type=int, default=4, help='Amount of ENVIRONMENT steps between each gradient step')
parser.add_argument('--log_every', type=int, default=1000, help='Amount of GRADIENT steps between each value log')
parser.add_argument('--show_progress', type=bool, default=False, help='Show progress bar and expected time to simulation completion')
parser.add_argument('--game_name', type=str, help='Game name')

args = parser.parse_args()

# set seeds for reproducibility
import torch; torch.manual_seed(args.seed)
import numpy as np; np.random.seed(args.seed)
import random; random.seed(args.seed)

# directory where tensorboard quantities will be stored
log_dir = 'env_{}_temp_{}_alpha_{}_seed_{}'.format(args.game_name[:-14], args.lambd, args.alpha, args.seed)

# experiment setup
agent = RAC(batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            memory_size=args.memory_size,
            num_iterations=args.num_iterations,
            lr=args.lr,
            lambd=args.lambd,
            alpha=args.alpha,
            gradient_step_every=args.gradient_step_every,
            log_every=args.log_every,
            discount=args.discount,
            env_id=args.game_name,
            log_dir=log_dir)

# run experiment
agent.train(verbose=args.show_progress)