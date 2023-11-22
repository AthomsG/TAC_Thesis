# Reinforcement Learning with RAC

Repository for Reinforcement Actor-Critic (RAC) model implementation for solving reinforcement learning tasks. This model is built using PyTorch and is designed to effectively learn policies that maximize α-Tsallis regularized expected returns in various environments.

## Structure

### value_networks.py
This file contains the implementation of the Critic and Actor networks utilized in the RAC model. The Critic network is a Convolutional Neural Network (CNN) that takes a batch of 4 grayscale images of size 84x84 as input and produces a value for each possible action. The Actor network, on the other hand, takes the environment state as input and generates a policy distribution over the available actions.

### algorithm.py
In this file, you'll find the core implementation of the RAC model. It orchestrates the interaction between the Critic and Actor networks to learn a policy optimizing the expected return. The file includes the training loop, as well as methods for saving and loading the model.

### buffers.py
Here, the Replay Buffer is implemented for storing and sampling past experiences, a crucial component in reinforcement learning.

### atari_env.py
This file comprises the implementation of the Atari environment wrapper, responsible for preprocessing game frames before they are fed into the model.