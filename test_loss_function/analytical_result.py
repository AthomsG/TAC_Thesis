import numpy as np
import os

from algorithm import test_env
from scipy.special import softmax as softmax

n_actions    = 3
n_iterations = 2000

discount = 0.99
lambd    = 0.1
lr       = 5e-1
alpha    = 1

# path to parameters
path_to_weights = 'weights/'

q_losses = list()
p_losses = list()

def softmax_gradient(y):
    y = y.reshape(-1, 1)
    J = - np.dot(y, y.T)
    for i in range(len(y)):
        J[i, i] = y[i] * (1 - y[i])
    return J

# test environment
env = test_env(n_actions)

# actor parameters
actor_weights = {}
for file in os.listdir(path_to_weights):
    if 'actor' in file:
        param_name = file.replace('actor_', '').replace('.txt', '')
        actor_weights[param_name] = np.loadtxt(os.path.join(path_to_weights, file))

W_actor = actor_weights['simple_fc1.weight']
b_actor = actor_weights['simple_fc1.bias']

# critic parameters
critic_weights = {}
for file in os.listdir(path_to_weights):
    if 'critic' in file:
        param_name = file.replace('critic_', '').replace('.txt', '')
        critic_weights[param_name] = np.loadtxt(os.path.join(path_to_weights, file))

W_critic = critic_weights['simple_fc1.weight']
b_critic = critic_weights['simple_fc1.bias']

state = env.reset() # initial environment state

for i in range(n_iterations):
    # 1st Environment step
    policy = softmax(W_actor @ state + b_actor)

    action = np.argmax(policy)

    next_state, reward, _, _ = env.step(action)

    transition = (state, action, reward, next_state, 0) # the transition is stored in the replay buffer

    # 1st Gradient step

    # critic
    policy_y    = softmax(W_actor @ next_state + b_actor)
    next_action = np.argmax(policy_y)

    Q_st_at     = (W_critic @ state + b_critic)[action]
    Q_st1_at1   = (W_critic @ next_state + b_critic)[next_action]

    y = reward + discount*((Q_st1_at1) + (lambd/alpha)*np.log(policy_y[next_action]))

    # compute value loss
    q_loss = (Q_st_at - y)**2
    q_losses.append(q_loss)

    # gradient \nabla_{\theta}Q_{\theta}
    # one-hot matrix
    one_hot_matrix = np.zeros((n_actions, n_actions))
    one_hot_matrix[action, :] = np.ones(n_actions)

    one_hot_vector = np.zeros(n_actions)
    one_hot_vector[action] = 1

    # element-wise multiplication
    delw_C = one_hot_matrix * (Q_st_at - y) * np.array([state for i in range(n_actions)])
    delb_C = one_hot_vector * (Q_st_at - y) * np.ones(n_actions)

    # actor
    log_2a  = np.log(policy)

    Q_st    = W_critic @ state + b_critic

    # compute policy loss
    p_loss = -np.sum(policy*(Q_st - (lambd/alpha)*log_2a))
    p_losses.append(p_loss)

    del_z_w = np.repeat(state[:, np.newaxis], n_actions, axis=1) # stack vectors vertically, side by side (could also use reshape)
    del_pi_z = softmax_gradient(policy)

    delw_A = -(Q_st - log_2a + 1) * del_pi_z * np.array([state, state, state])

    state = next_state

    # update weights
    W_actor  -= lr*delw_A
    W_critic -= lr*delw_C

    b_critic -= lr*delb_C

from IPython import embed; embed()