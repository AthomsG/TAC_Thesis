#!/bin/bash

# creates the train_log directory if it doesn't exist
mkdir -p train_log
mkdir -p train_log/$1 # $1 is the environment name that is passed

# define an array of seed values
seeds=(1)

for seed in "${seeds[@]}"; do
    # define the log file with the specified format
    log_dir="train_log/$1/rac_alpha2_lambd0.1_seed_${seed}.log"
    # run the Python script and redirect the output to the log file
    python -u train.py --game $1 --alpha 2 --lambd 0.1 --seed $seed > $log_dir 2>&1 &
done

echo "All processes have started!"

# pkill -f "train.py --game $1" # to kill all started processes
