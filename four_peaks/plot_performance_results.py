import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from modes_env import Mode_Seeking_Environment
import argparse

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

# Calculate maximum possible score
def calculate_max_possible_score(action_discretization):
    episode_length = 1000
    env = Mode_Seeking_Environment(
        peak_steepness=0.002,
        action_discretization=action_discretization,
        max_distance=5,
        local_region_size=(3, 3),
        reset_interval=1000
    )
    max_rewards = 0
    for _ in tqdm(range(episode_length), desc=f"Calculating Max Rewards for discretization {action_discretization}"):
        action_rewards = env.action_rewards()
        best_action_index = np.argmax(list(action_rewards.values()))
        _, reward, _ = env.step(best_action_index)
        max_rewards += reward
    return max_rewards

# Main function
def main(normalize):
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

    # Convert sets to sorted lists
    alphas = sorted(list(alphas))
    discretizations = sorted(list(discretizations))
    seeds = sorted(list(seeds))
    lambd_values = sorted(list(lambd_values))

    # Compute maximum possible scores for each discretization
    if normalize: max_possible_scores = {disc: calculate_max_possible_score(disc) for disc in discretizations}

    for lambd in lambd_values:
        print(f"Processing lambda: {lambd}")

        aurc_results = {alpha: [] for alpha in alphas}

        for alpha in alphas:
            for disc in discretizations:
                all_rewards = []
                for seed in seeds:
                    reward_path = f'{directory}/rewards_alpha_{alpha}_lambda_{lambd}_disc_{disc}_seed_{seed}.npy'
                    if os.path.exists(reward_path):
                        rewards = np.load(reward_path, allow_pickle=True)
                        if normalize:
                            normalized_rewards = calculate_aurc(rewards) / (max_possible_scores[disc] * len(rewards))
                        else:
                            normalized_rewards = calculate_aurc(rewards)
                        all_rewards.append(normalized_rewards)
                if all_rewards:
                    aurc_mean = np.mean(all_rewards)
                    aurc_std = np.std(all_rewards)
                    aurc_results[alpha].append((aurc_mean, aurc_std))
                else:
                    aurc_results[alpha].append((None, None))

        # Plotting
        plt.figure(figsize=(12, 8))
        for alpha in alphas:
            means = [result[0] for result in aurc_results[alpha] if result[0] is not None]
            stds = [result[1] for result in aurc_results[alpha] if result[1] is not None]
            corresponding_discs = [discretizations[i]**2 - discretizations[i] for i, result in enumerate(aurc_results[alpha]) if result[0] is not None]
            if means and stds:
                plt.errorbar(
                    corresponding_discs, means, yerr=stds, label=f'alpha={alpha}', 
                    fmt='-o', capsize=5, capthick=2, elinewidth=2, markersize=10, linewidth=2
                )

        ylabel_text = 'Normalized Area Under Reward Curve (AURC)' if normalize else 'Area Under Reward Curve (AURC)'
        plt.xlabel('Action Discretization', fontsize=14)
        plt.ylabel(ylabel_text, fontsize=14)
        plt.title(f'Performance vs. Action Discretization (lambda={lambd})', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.ylim(0, 1 if normalize else None)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process rewards data.")
    parser.add_argument('--normalize', action='store_true', help='Whether to normalize the rewards')
    parser.set_defaults(normalize=False)
    args = parser.parse_args()
    main(args.normalize)
