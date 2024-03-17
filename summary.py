from torch.utils.tensorboard import SummaryWriter
import numpy as np
from scipy.stats import wasserstein_distance

def mean_wasserstein_distance(pis_1, pis_2):
    pis_1 = pis_1.detach().cpu().numpy()
    pis_2 = pis_2.detach().cpu().numpy()

    distances = [wasserstein_distance(pi_1, pi_2) for pi_1, pi_2 in zip(pis_1, pis_2)]

    return np.mean(distances)

class Summary:
    """
    Buffer used to log training statistics for reinforcement learning algorithms.
    """
    def __init__(self, log_dir, n_actions):
        """
        Initialize a Summary object.
        
        Params
        ======
            log_dir (str): Directory for TensorBoard logs.
            n_actions (int): Number of actions in the action space.
        """
        self.n_actions = n_actions
        self.writer = SummaryWriter(log_dir=log_dir)
        self.reset()

    def reset(self):
        """
        Reset the statistics stored in the summaries dictionary.
        """
        # tensorboard quantities
        self.summaries = {'v_loss': {'sum': 0, 'count': 0},          # value loss values during training  - Average value loss PER GRADIENT STEP
                          'p_loss': {'sum': 0, 'count': 0},          # policy loss values during training - Average policy loss PER GRADIENT STEP
                          'reward': {'sum': 0, 'count': 1},          # rewards during environment steps   - Average reward PER EPISODE
                          'sparsity': {'sum': 0, 'count': 0},        # percentage of actions with null probability of happening (sum: sparse actions, count: dim(A))
                          'max_actor_grad': 0,                       # maximum L1 gradient norm for the actor network
                          'max_critic_grad': 0,                      # maximum L1 gradient norm for the critic network
                          'tsallis_entropy': {'sum': 0, 'count': 0}, # Tsallis entropy with run's alpha value - Average self.alpha Tsallis entropy PER STEP
                          'alpha_2_entropy': {'sum': 0, 'count': 0}, # alpha-2 Tsallis entropy used across experiments to compare uncertainty during training
                          'wsn_distance': {'sum': 0, 'count': 0},    # Wasserstein distance between policies of same states after gradient step - Average Wasserstein Distance PER GRADIENT STEP
                          # probabilities of each action
                          **{f'action_probs_{i}': {'sum': 0, 'count': 0} for i in range(self.n_actions)},
                          **{f'max_action_probs_{i}': 0 for i in range(self.n_actions)}, # max probability value
                          # ReplayBuffer occurences of each action
                          **{f'buff_actions_{i}': {'sum': 0, 'count': 0} for i in range(self.n_actions)},
                          # Expected Log-alpha for each action
                          **{f'action_log_alpha_{i}': {'sum': 0, 'count': 0} for i in range(self.n_actions)}
                        }

    def update(self, name, value, count=1):
        """
        Update the value of a statistic in the summaries dictionary.
        
        Params
        ======
            name (str): Name of the statistic.
            value (float): Value to add to the statistic.
            count (int): Number of occurrences to add to the statistic count.
        """
        if isinstance(self.summaries[name], dict):
            self.summaries[name]['sum'] += value
            self.summaries[name]['count'] += count
        else:
            self.summaries[name] = max(self.summaries[name], value)

    def mean(self, name):
        """ Compute the sample mean"""
        return self.summaries[name]['sum'] / self.summaries[name]['count']
    
    def log_to_tensorboard(self, gradient_step):
        """
        Log the statistics in the summaries dictionary to TensorBoard.
        
        Params
        ======
            gradient_step (int): Current gradient step (x-axis value).
        """
        n_episodes   = self.summaries['reward']['count']
        # log environment quantities
        self.writer.add_scalar('reward', self.mean('reward'), gradient_step)
        self.writer.add_scalar('n_episodes', n_episodes, gradient_step)
        # log losses and gradient norms
        self.writer.add_scalar('p_loss', self.mean('p_loss'), gradient_step)
        self.writer.add_scalar('v_loss', self.mean('v_loss'), gradient_step)
        self.writer.add_scalar('max_actor_grad', self.summaries['max_actor_grad'], gradient_step)
        self.writer.add_scalar('max_critic_grad', self.summaries['max_critic_grad'], gradient_step)
        # log sparsity quantities
        self.writer.add_scalar('sparsity', self.mean('sparsity'), gradient_step)
        mean_action_probs = {str(action): self.mean(f'action_probs_{action}') for action in range(self.n_actions)}
        self.writer.add_scalars('action_probs', mean_action_probs, gradient_step)
        max_action_probs = {str(action): self.summaries[f'max_action_probs_{action}'] for action in range(self.n_actions)}
        self.writer.add_scalars('max_action_probs', max_action_probs, gradient_step)
        # log ReplayBuffer statistics
        buff_actions_dist = {str(action): self.mean(f'buff_actions_{action}') for action in range(self.n_actions)}
        self.writer.add_scalars('actions_in_buffer', buff_actions_dist, gradient_step)
        # log policy change
        self.writer.add_scalar('wsn_distance', self.mean('wsn_distance'), gradient_step)
        # log entropies
        self.writer.add_scalar('tsallis_entropy', self.mean('tsallis_entropy'), gradient_step)
        self.writer.add_scalar('alpha_2_entropy', self.mean('alpha_2_entropy'), gradient_step)
        action_log_alpha = {str(action): self.mean(f'action_log_alpha_{action}') for action in range(self.n_actions)}
        self.writer.add_scalars('actions_log_alpha', action_log_alpha, gradient_step)

        # reset quantities
        self.reset()