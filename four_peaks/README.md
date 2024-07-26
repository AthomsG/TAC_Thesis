# Performance Experiment Runner

This script is designed to run a series of performance experiments with varying hyperparameters in parallel using `tmux` sessions. Each experiment runs in a separate `tmux` session, and a progress bar updates as the experiments complete. This setup works on both macOS and Linux systems.

## Requirements

- `tmux`: A terminal multiplexer that allows you to run multiple terminal sessions within a single window. Install it using the following commands:
  - macOS: `brew install tmux`
  - Linux: `sudo apt-get install tmux`
- `python3`: Ensure you have Python 3 installed. Install it using:
  - macOS: `brew install python`
  - Linux: `sudo apt-get install python3`
- Python script `performance_plot_train.py`: This script is expected to be present in the same directory as the bash script.

## Script Description

### Directory Setup

The script creates the following directories if they do not exist:
- `performance_runs`
- `performance_rewards`
- `performance_models`

### Hyperparameters

The script runs experiments over a grid of hyperparameters:
- `alphas`: An array of alpha values (`1.0`, `1.5`).
- `lambd`: A fixed lambda value (`0.01`).
- `discretizations`: An array of discretization values (`4, 6, 8, 10, 12, 14, 16, 20, 30, 40, 50`).
- `seeds`: An array of seed values (`0, 1, 2, 3, 4, 5`).

### Experiment Execution

For each combination of `alpha` and `discretization`, the script runs experiments in parallel for each seed value. Each experiment runs in a separate `tmux` session with a name that includes the hyperparameter values for easy identification.

### Progress Bar

The script displays a progress bar indicating the completion status of the experiments.

### Kill Switch

The script includes a kill switch that allows you to stop all running experiments by pressing `Ctrl+C`. This will trigger a cleanup function to terminate all running `tmux` sessions related to the experiments.

## Usage

### Running the Script

To run the script, use the following command:

\`\`\`bash
./run_performance_experiment.sh
\`\`\`

### Listing Running Experiments

To list the currently running experiments, use the following command:

\`\`\`bash
tmux list-sessions
\`\`\`

This will display all active `tmux` sessions, including those created by the script.

### Stopping the Experiments

To stop all running experiments, press `Ctrl+C` in the terminal where the script is running. This will trigger the cleanup function, which terminates all `tmux` sessions related to the experiments and exits the script gracefully.

## Script Breakdown

### Function Definitions

- **run_experiment**: Starts a new `tmux` session for each experiment with a unique name based on the hyperparameters.
- **display_progress**: Updates and displays the progress bar.
- **cleanup**: Terminates all running `tmux` sessions related to the experiments and exits the script.

### Main Loop

The main loop iterates over all combinations of `alpha` and `discretizations` values. For each combination, it runs experiments in parallel for each seed value. It waits for all experiments in a batch to complete before proceeding to the next set of hyperparameters and updates the progress bar accordingly.

## Example Output

\`\`\`bash
Completed 1 out of 22 experiments (4.55%).
Completed 2 out of 22 experiments (9.09%).
...
Completed 22 out of 22 experiments (100.00%).
All experiments completed.
\`\`\`

## Conclusion

This script provides an efficient way to run multiple experiments in parallel while keeping track of their progress. The use of `tmux` sessions allows for easy management and cleanup of running processes. By following the instructions above, you can set up and run your own performance experiments with this script.