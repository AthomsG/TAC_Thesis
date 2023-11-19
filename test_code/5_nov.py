import sys
sys.path.append("..")

from value_networks import *
import torch
import numpy as np


agent = RAC("", 4)
agent.train('BreakoutNoFrameskip-v4')

batches = agent.replay_buffer.sample(10)
agent.perform_gradient_step(*batches) # gradient is successfully computed

from IPython import embed; embed()
