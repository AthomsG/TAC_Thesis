#!/bin/bash

# check if alpha value is provided
if [ -z "$1" ]
then
    echo "No alpha value provided. Usage: ./kill_alpha.sh <alpha_value>"
    exit 1
fi

alpha_value=$1
pids_file="pids.txt"
temp_pids_file="temp_pids.txt"

# flag to check if any process was killed
process_killed=0

# read the pids file line by line
while read pid; do
    # check if the process is running
    if ps -p $pid > /dev/null; then
        # get the command line used to start the process
        cmdline=$(ps -p $pid -o args=)
        # check if the command line contains the provided alpha value
        if [[ $cmdline == *"--alpha $alpha_value"* ]]; then
            # get the name of the process
            process_name=$(ps -p $pid -o comm=)
            # kill the process
            kill $pid
            echo "Killed process $process_name with PID $pid"
            process_killed=1
        else
            # if the process was not killed, write its pid to the temp pids file
            echo $pid >> "$temp_pids_file"
        fi
    fi
done < "$pids_file"

# replace the pids file with the temp pids file
mv "$temp_pids_file" "$pids_file"

if [ $process_killed -eq 0 ]; then
    echo "No processes found with alpha value $alpha_value"
fi