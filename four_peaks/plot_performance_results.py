import os
import numpy as np
import matplotlib.pyplot as plt

# Function to extract parameters from filenames
def extract_parameters(filename):
    parts = filename.split('_')
    alpha = float(parts[2])
    lambd = float(parts[4])
    disc = int(parts[6])
    seed = int(parts[8].split('.')[0])
    return alpha, lambd, disc, seed

# Function to calculate Area Under the Reward Curve (AURC)
def calculate_aurc(rewards):
    return np.sum(rewards)

# Scan the directory and extract all unique parameters
directory = 'performance_rewards'
all_files = os.listdir(directory)

alphas = set()
discretizations = set()
seeds = set()
lambd_values = set()

for file in all_files:
    if file.endswith('.npy'):
        alpha, lambd, disc, seed = extract_parameters(file)
        alphas.add(alpha)
        discretizations.add(disc)
        seeds.add(seed)
        lambd_values.add(lambd)

# Assuming there is only one lambda value
if len(lambd_values) == 1:
    lambd = lambd_values.pop()
else:
    raise ValueError("There should be exactly one lambda value.")

# Convert sets to sorted lists
alphas = sorted(list(alphas))
discretizations = sorted(list(discretizations))
seeds = sorted(list(seeds))

aurc_results = {alpha: [] for alpha in alphas}

for alpha in alphas:
    for disc in discretizations:
        all_rewards = []
        for seed in seeds:
            reward_path = f'{directory}/rewards_alpha_{alpha}_lambda_{lambd}_disc_{disc}_seed_{seed}.npy'
            if os.path.exists(reward_path):
                rewards = np.load(reward_path, allow_pickle=True)
                all_rewards.append(calculate_aurc(rewards))
        if all_rewards:
            aurc_mean = np.mean(all_rewards)
            aurc_std = np.std(all_rewards)
            aurc_results[alpha].append((aurc_mean, aurc_std))
        else:
            aurc_results[alpha].append((None, None))

# Plotting
plt.figure(figsize=(10, 6))
for alpha in alphas:
    means = [result[0] for result in aurc_results[alpha] if result[0] is not None]
    stds = [result[1] for result in aurc_results[alpha] if result[1] is not None]
    corresponding_discs = [discretizations[i] for i, result in enumerate(aurc_results[alpha]) if result[0] is not None]
    if means and stds:
        plt.errorbar(corresponding_discs, means, yerr=stds, label=f'alpha={alpha}', fmt='-o')

plt.xlabel('Action Discretization')
plt.ylabel('Area Under Reward Curve (AURC)')
plt.title('Performance vs. Action Discretization')
plt.legend()
plt.grid(True)
plt.show()