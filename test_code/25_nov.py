import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
# run with rqn_env for TensorFlow
import sys
sys.path.append("..")

from value_networks import Actor, Critic

SAVE_WEIGHTS=False

def compare_weights(pytorch_model, tensorflow_model):
    # Get the weights from the PyTorch model
    pytorch_weights = {name: param.data.numpy() for name, param in pytorch_model.named_parameters()}

    # Get the variables from the TensorFlow graph
    tensorflow_variables = {v.name: v for v in tf.global_variables()}

    # Compare the weights
    for name, weight in pytorch_weights.items():
        # TensorFlow uses a different naming convention, so we need to find the corresponding variable in the TensorFlow graph
        for tf_name, tf_variable in tensorflow_variables.items():
            if name in tf_name:
                # Get the value of the TensorFlow variable
                tf_weight = tf_variable.eval(session=tensorflow_model)
                # Check if the weights are the same
                if not np.allclose(weight, tf_weight, atol=1e-6):
                    print(f"Weights for layer {name} are different.")
                    return False

    print("All weights are the same.")
    return True

actor  = Actor(n_actions=4, alpha=1)
critic = Critic(n_actions=4)

if SAVE_WEIGHTS:
    # save model weights to load onto Tensorflow nets
    torch.save(critic.state_dict(), 'starting_weights/critic_starting_weights.pth')
    torch.save(actor.state_dict(),  'starting_weights/actor_starting_weights.pth')

else:
    critic.load_state_dict(torch.load('starting_weights/critic_starting_weights.pth'))
    actor.load_state_dict(torch.load('starting_weights/actor_starting_weights.pth'))

critic_weights = torch.load('starting_weights/critic_starting_weights.pth')
actor_weights  = torch.load('starting_weights/actor_starting_weights.pth')
# convert to numpy arrays
critic_weights = {k: v.numpy() for k, v in critic_weights.items()}
actor_weights = {k: v.numpy() for k, v in actor_weights.items()}

# Tensorflow networks
def net(x): # Define q-value network function estimator (CRITIC)
    conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu, 
                                     weights_initializer=tf.constant_initializer(critic_weights['conv1.weight'].transpose(2, 3, 1, 0)), 
                                     biases_initializer=tf.constant_initializer(critic_weights['conv1.bias']))
    conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, 
                                     weights_initializer=tf.constant_initializer(critic_weights['conv2.weight'].transpose(2, 3, 1, 0)), 
                                     biases_initializer=tf.constant_initializer(critic_weights['conv2.bias']))
    conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu, 
                                     weights_initializer=tf.constant_initializer(critic_weights['conv3.weight'].transpose(2, 3, 1, 0)), 
                                     biases_initializer=tf.constant_initializer(critic_weights['conv3.bias']))
    flattened = tf.contrib.layers.flatten(conv3)
    fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu, 
                                            weights_initializer=tf.constant_initializer(critic_weights['fc1.weight'].T), 
                                            biases_initializer=tf.constant_initializer(critic_weights['fc1.bias']))
    fc2 = tf.contrib.layers.fully_connected(fc1, 4, activation_fn=None, 
                                            weights_initializer=tf.constant_initializer(critic_weights['fc2.weight'].T), 
                                            biases_initializer=tf.constant_initializer(critic_weights['fc2.bias']))
    return fc2

def policy_net(x): # Define policy network (ACTOR)
    conv1 = tf.contrib.layers.conv2d(x, 32, 8, 4, activation_fn=tf.nn.relu, 
                                     weights_initializer=tf.constant_initializer(actor_weights['conv1.weight'].transpose(2, 3, 1, 0)), 
                                     biases_initializer=tf.constant_initializer(actor_weights['conv1.bias']))
    conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu, 
                                     weights_initializer=tf.constant_initializer(actor_weights['conv2.weight'].transpose(2, 3, 1, 0)), 
                                     biases_initializer=tf.constant_initializer(actor_weights['conv2.bias']))
    conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu, 
                                     weights_initializer=tf.constant_initializer(actor_weights['conv3.weight'].transpose(2, 3, 1, 0)), 
                                     biases_initializer=tf.constant_initializer(actor_weights['conv3.bias']))
    flattened = tf.contrib.layers.flatten(conv3)
    fc1 = tf.contrib.layers.fully_connected(flattened, 512, activation_fn=tf.nn.relu, 
                                            weights_initializer=tf.constant_initializer(actor_weights['fc1.weight'].T), 
                                            biases_initializer=tf.constant_initializer(actor_weights['fc1.bias']))
    fc2 = tf.contrib.layers.fully_connected(fc1, 4, activation_fn=tf.nn.softmax, 
                                            weights_initializer=tf.constant_initializer(actor_weights['fc2.weight'].T), 
                                            biases_initializer=tf.constant_initializer(actor_weights['fc2.bias']))
    return fc2

x = tf.placeholder(tf.float32, shape=(None, 84, 84, 4))

# initialize networks
# TensorFlow
critic_tf = net(x)
actor_tf = policy_net(x)

# compare models' weights
compare_weights(critic, critic_tf)
compare_weights(actor, actor_tf)

#critic_tf2 = net(x)
#actor_tf2 = policy_net(x)

# start TensorFlow session
sess = tf.Session()

# initialize the variables
sess.run(tf.global_variables_initializer())

# compare models' weights
compare_weights(critic, sess)
compare_weights(actor, sess)

'''
we'll compute the squared error between actor and critic models in both frameworks.
'''

# input
rand_tensor = np.ones((1, 84, 84, 4))
pytorch_input = torch.tensor(rand_tensor.reshape(1, 4, 84, 84)).float()

# pytorch
with torch.no_grad():
    pytorch_actor = actor(pytorch_input)
    pytorch_critic = critic(pytorch_input)

# tensorboard
tflow_critic, tflow_actor = sess.run([critic_tf, actor_tf], feed_dict={x: rand_tensor})

''' PyTorch
        - Actor:  [0.2495, 0.2392, 0.2543, 0.2570]
        - Critic: [-0.0246,  0.0137,  0.0719, -0.0592]
'''

''' TensorFlow
        - Actor: [0.2507, 0.2372, 0.2552, 0.2569]
        - Critic:[ 0.5945, -0.7763,  1.5128, -1.0385]
'''

from IPython import embed; embed()