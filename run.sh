#!/bin/bash

# creates the train_log directory if it doesn't exist
mkdir -p train_log
mkdir -p train_log/$1 # $1 is the environment name that is passed

# define an array of values to be used in experiments
seeds=(0 1)
lambd=(0.1)
alpha=(1 1.5 2 5)

# array to store process IDs
pids_file="pids.txt"

# remove existing pids file if exists
rm -f "$pids_file"

# specify the GPU you want to use, default is 0
export CUDA_VISIBLE_DEVICES=${2:-0}

for seed in "${seeds[@]}"; do
    for lam in "${lambd[@]}"; do
        for alp in "${alpha[@]}"; do
            # define the log file where terminal outputs are stored
            log_dir="train_log/$1/temp_${lam}_alpha_${alp}_seed_${seed}.log"
            # run Python script and redirect the output to log file
            python -u train.py --game_name $1 --alpha $alp --lambd $lam --seed $seed --num_iterations 3000000 --learning_starts 5000 --update_target_every 2500 > $log_dir 2>&1 &
            # store the PID of the last background process and write it to the pids file
            echo $! >> "$pids_file"
        done
    done
done

echo "All processes have started!"