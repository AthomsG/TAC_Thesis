import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from value_networks import Actor
from modes_env import Mode_Seeking_Environment

# Function to load the model weights
def load_actor_weights(actor, actor_path):
    actor.load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
    print(f"Actor model loaded from {actor_path}")

# Function to plot trajectories
def plot_trajectories(env, actor, device, num_trajectories=20, trajectory_length=20, save=False, show=False):
    trajectories = []
    for _ in range(num_trajectories):
        trajectory = []
        state = env.reset().flatten()  # Flatten the initial state
        initial_position = env.agent_position.copy()
        trajectory.append(initial_position)
        # print(f"Initial position: {initial_position}")
        for _ in range(trajectory_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                policy = actor(state_tensor)
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample().item()
            next_state, reward, _ = env.step(action)
            next_position = env.agent_position.copy()  # Store the (x, y) position
            trajectory.append(next_position)
            state = next_state.flatten()  # Flatten the next state
            # print(f"Step: position={next_position}, action={action}, reward={reward}")
        trajectories.append(np.array(trajectory))

    # Plotting the reward contours
    x = np.linspace(0, env.grid_size[0], 200)
    y = np.linspace(0, env.grid_size[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(200):
        for j in range(200):
            Z[i, j] = env.reward_function([X[i, j], Y[i, j]])

    contour_levels = np.arange(0.1, 1.0, 0.1)
    plt.figure(figsize=(8, 8))  # Make the plot square
    contours = plt.contour(X, Y, Z, levels=contour_levels, cmap='coolwarm')
    plt.clabel(contours, inline=True, fontsize=8, fmt='%.1f')

    # Plotting the trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, num_trajectories))
    for idx, trajectory in enumerate(trajectories):
        plt.plot(trajectory[:, 0], trajectory[:, 1], color=colors[idx], alpha=0.6)

    # Plot the goals as red dots
    peak_color = cm.coolwarm(1.0)  # Get the color for maximum value in coolwarm colormap
    for peak in env.peaks:
        plt.plot(peak[0], peak[1], 'o', color=peak_color, markersize=8)

    plt.title('Trajectories in 2D Multi-Goal Environment')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the plot is square
    if save:
        plt.savefig(save)
    if show:
        plt.show()

# Variables for number of trajectories and steps per trajectory
num_trajectories = 100
trajectory_length = 100

alphas = [1.0, 1.5, 2.0]
lambds = [0.1]
discs  = [50]

for alpha in alphas:
    for lambd in lambds:
        for disc in discs:
            # Initialize the environment
            env = Mode_Seeking_Environment(peak_steepness=0.002, action_discretization=disc, max_distance=5, local_region_size=(3, 3), reset_interval=1000)

            # Initialize the actor model
            obs_dim = np.prod(env.local_region_size)  # Flatten the input dimension
            n_actions = env.action_space.shape[0]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            actor = Actor(obs_dim, n_actions, alpha=alpha).to(device)

            load_actor_weights(actor, f'models/actor_alpha_{alpha}_lambda_{lambd}_disc_{disc}.pth')
            plot_trajectories(env, actor, device, num_trajectories, trajectory_length, save=f'trajectories/alpha_{alpha}_lambda_{lambd}_disc_{disc}.png', show=False)