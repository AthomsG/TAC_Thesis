import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15, entmax_bisect

class BaseNetwork(nn.Module):
    def __init__(self, input_dim):
        super(BaseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Critic(BaseNetwork):
    def __init__(self, input_dim, n_actions):
        super(Critic, self).__init__(input_dim)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = super().forward(x)
        value = self.fc3(x)
        return value

class Actor(BaseNetwork):
    def __init__(self, input_dim, n_actions, alpha=1):
        super(Actor, self).__init__(input_dim)
        self.fc3 = nn.Linear(128, n_actions)
        self.alpha = alpha

    def forward(self, x):
        x = super().forward(x)
        x = self.fc3(x)
        if self.alpha == 1:
            x = F.softmax(x, dim=1)
        elif self.alpha == 1.5:
            x = entmax15(x, dim=1)
        elif self.alpha == 2:
            x = sparsemax(x, dim=1)
        else:
            x = entmax_bisect(x, alpha=self.alpha, dim=1)
        return x