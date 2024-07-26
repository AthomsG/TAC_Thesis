#!/bin/bash

# Check dependencies
if ! command -v tmux &> /dev/null; then
    printf "tmux could not be found. Please install tmux.\n" >&2
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    printf "python3 could not be found. Please install python3.\n" >&2
    exit 1
fi

# Ensure performance_plot_train.py exists
if [[ ! -f performance_plot_train.py ]]; then
    printf "performance_plot_train.py not found in the current directory.\n" >&2
    exit 1
fi

# creates the runs and rewards directories if they don't exist
mkdir -p performance_runs
mkdir -p performance_rewards
mkdir -p performance_models

# define arrays of values to be used in experiments
alphas=(1.0 1.5)
lambd=0.01
discretizations=(4 10 20 30 40 50)
seeds=(0 1 2 3 4 5)

total_experiments=$(( ${#alphas[@]} * ${#discretizations[@]} ))
experiment_count=0
start_time=$(date +%s)

# Function to run a single experiment
run_experiment() {
    local alpha=$1
    local disc=$2
    local seed=$3
    local session_name="alpha_${alpha}_disc_${disc}_seed_${seed}"
    tmux new-session -d -s "$session_name" "python performance_plot_train.py --alpha $alpha --lambd $lambd --disc $disc --seed $seed --num_iterations 600000"
}

# Function to display progress
display_progress() {
    local progress=$(printf "%.2f" $(echo "scale=2; $1 / $2 * 100" | bc))
    local elapsed_time=$(( $(date +%s) - $start_time ))
    local hours=$(( elapsed_time / 3600 ))
    local minutes=$(( (elapsed_time % 3600) / 60 ))
    local seconds=$(( elapsed_time % 60 ))
    printf "\rCompleted %d out of %d experiments (%s%%). Time elapsed: %02d:%02d:%02d." "$1" "$2" "$progress" "$hours" "$minutes" "$seconds"
}

# Function to clean up tmux sessions
cleanup() {
    printf "\n\nStopping all running experiments...\n"
    tmux list-sessions | grep -o 'alpha_[^:]*' | xargs -I {} tmux kill-session -t {}
    printf "All running experiments stopped.\n"
    exit 0
}

# Trap SIGINT (Ctrl+C) to clean up tmux sessions
trap cleanup SIGINT

# Main function
main() {
    for alpha in "${alphas[@]}"; do
        for disc in "${discretizations[@]}"; do
            # Run experiments in parallel using tmux
            for seed in "${seeds[@]}"; do
                run_experiment $alpha $disc $seed
            done
            
            # Wait for all tmux sessions to complete
            while [ $(tmux list-sessions | grep -c "alpha_${alpha}_disc_${disc}_seed_") -gt 0 ]; do
                sleep 1
                display_progress $experiment_count $total_experiments
            done

            # Update progress bar
            experiment_count=$((experiment_count + 1))
            display_progress $experiment_count $total_experiments
        done
    done

    echo
    echo "All experiments completed."
}

# Call main function
main