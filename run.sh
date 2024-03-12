#!/bin/bash

# creates the train_log directory if it doesn't exist
mkdir -p train_log
mkdir -p train_log/$1 # $1 is the environment name that is passed

# define an array of values to be used in experiments
seeds=(0)
lambd=(0.1)
alpha=(1 2 5)

for seed in "${seeds[@]}"; do
    for lam in "${lambd[@]}"; do
        for alp in "${alpha[@]}"; do
            # define the log file where terminal outputs are stored
            log_dir="train_log/$1/temp_${lam}_alpha_${alp}_seed_${seed}.log"
            # run Python script and redirect the output to log file
            python -u train.py --game_name $1 --alpha $alp --lambd $lam --seed $seed --num_iterations 100 --learning_starts 32 > $log_dir 2>&1 &
        done
    done
done

echo "All processes have started!"

# pkill -9 -f "train.py --game $1" # to kill all started processes