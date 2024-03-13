import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect

class ConvBase(nn.Module):
    def __init__(self):
        super(ConvBase, self).__init__()
        # define convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),  # Conv Layer 1: 32 filters, 8x8 kernel, stride 4
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),  # Conv Layer 2: 64 filters, 4x4 kernel, stride 2
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),  # Conv Layer 3: 64 filters, 3x3 kernel
            nn.ReLU(),
        )
        # compute the output size for linear layers
        self.output_size = self._get_conv_output((4, 84, 84))

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = self.conv_layers(input)
        return int(torch.numel(output) / batch_size)

    def forward(self, x):
        return self.conv_layers(x)

class Critic(ConvBase):
    """
    Critic Network for Actor-Critic models.
    
    Takes a 4-channel image (grayscale, 84x84 pixels, stack of 4 frames) as input.
    """
    def __init__(self, n_actions):
        super(Critic, self).__init__()
        # Linear Layers
        self.fc1 = nn.Linear(self.output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = super().forward(x)
        x = x.view(-1, self.output_size)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Actor(ConvBase):
    """
    Actor Network for Actor-Critic models, with an option for entmax transformation.
    
    Takes a 4-channel image (grayscale, 84x84 pixels, stack of 4 frames) as input.
    Allows for a fixed alpha parameter to adjust the sparsity of the action selection policy.
    """
    def __init__(self, n_actions, alpha=1):
        super(Actor, self).__init__()
        self.alpha = alpha  # Fixed alpha for entmax transformation
        self.fc1 = nn.Linear(self.output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = super().forward(x)
        x = x.view(-1, self.output_size)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Alpha - entmax is the optimal policy for Tsallis regularized MDP
        if self.alpha == 1:
            x = F.softmax(x, dim=1)
        elif self.alpha == 1.5:
            x = entmax15(x, dim=1)
        elif self.alpha == 2:
            x = sparsemax(x, dim=1)
        else:
            x = entmax_bisect(x, alpha=self.alpha, dim=1)
        return x