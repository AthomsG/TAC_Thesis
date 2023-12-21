import argparse
from algorithm import RAC

# argument parser (makes it easier to run several experiments simultaneously)
parser = argparse.ArgumentParser(description='Train RAC agent')

parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--learning_starts', type=int, default=5000, help='Number of steps before learning starts')
parser.add_argument('--memory_size', type=int, default=100000, help='Size of the replay memory')
parser.add_argument('--num_iterations', type=int, default=400000, help='Number of iterations for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--lambd', type=float, default=0.1, help='Temperature Parameter')
parser.add_argument('--alpha', type=float, default=1, help='Alpha value of Alpha-Tsallis entropy') # Default is Softmax Policy
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor for the MDP')
parser.add_argument('--game', type=str, default='BreakoutNoFrameskip-v4', help='Name of the game')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')
parser.add_argument('--show_progress', type=bool, default=False, help='Show progress bar and expected time to simulation completion')

args = parser.parse_args()

# set seeds for reproducibility
import torch; torch.manual_seed(args.seed)
import numpy as np; np.random.seed(args.seed)
import random; random.seed(args.seed)

# experiment setup
agent = RAC(log_path=f'train_log/{args.game[:-14]}/alpha_{args.alpha}_lambda_{args.lambd}_seed_{args.seed}_SPARSEMAX_POLICY', # Action space for Boxing is 18
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            memory_size=args.memory_size,
            num_iterations=args.num_iterations,
            lr=args.lr,
            lambd=args.lambd,
            alpha=args.alpha,
            discount=args.discount)

agent.print_hyperparameters()
print(f'Environemnt name: {args.game}')

# run experiment
agent.train(args.game, verbose=args.show_progress)


''' About the Atari actions

All Atari environments action space is a subset of the following action space:

Actions:  0 - NOOP (no operation)
          1 - Fire
          2 - UP
          3 - RIGHT
          4 - LEFT
          5 - DOWN
          6 - UP RIGHT
          7 - UP LEFT
          8 - DOWN RIGHT
          9 - DOWN LEFT
          10- UP FIRE
          11- RIGHT FIRE
          12- LEFT FIRE
          13- DOWN FIRE
          14- UP RIGHT FIRE
          15- LEFT RIGHT FIRE
          16- DOWN RIGHT FIRE
          17- DOWN LEFT FIRE
'''