# Reinforcement Learning with RAC

This repository contains the implementation of a Reinforcement Actor-Critic (RAC) model for reinforcement learning tasks. The model is implemented in PyTorch.

## Structure

The main components of the repository are:

- `value_networks.py`: This file contains the implementation of the Critic and Actor networks used in the RAC model. The Critic network is a convolutional neural network (CNN) that takes as input a batch of 4 grayscale images of size 84x84, and outputs a value for each possible action. The Actor network is a network that takes as input the state of the environment and outputs a policy distribution over the possible actions.

- `RAC.py`: This file contains the implementation of the RAC model, which uses the Critic and Actor networks to learn a policy that maximizes the expected return.

## Requirements

The code is written in Python and requires PyTorch. Additionally, the `entmax` package is used for the sparsemax and entmax15 functions.
