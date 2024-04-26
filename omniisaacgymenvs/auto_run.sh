#!/bin/bash

# Define the arguments
python_script="scripts/skrl/diana_tekken_PPOFD.py"

# Check if the Python script exists
if [ ! -f "$python_script" ]; then
    echo "Error: Python script '$python_script' not found."
    exit 1
fi


#First run
arg1="+mini_batches=4"
arg2="+ratio_clip=0.2"
arg3="+lambda_0=0"
arg4="+reward_shaper=False"
arg5="+value_preprocessor=False"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Second run
arg1="+mini_batches=4"
arg2="+ratio_clip=0.2"
arg3="+lambda_0=0"
arg4="+reward_shaper=True"
arg5="+value_preprocessor=True"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Third run
arg1="+mini_batches=32"
arg2="+ratio_clip=0.2"
arg3="+lambda_0=0"
arg4="+reward_shaper=False"
arg5="+value_preprocessor=False"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Fourth run
arg1="+mini_batches=32"
arg2="+ratio_clip=0.2"
arg3="+lambda_0=0"
arg4="+reward_shaper=True"
arg5="+value_preprocessor=True"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Fifth run
arg1="+mini_batches=32"
arg2="+ratio_clip=0.2"
arg3="+lambda_0=1"
arg4="+reward_shaper=False"
arg5="+value_preprocessor=False"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Fifth run
arg1="+mini_batches=32"
arg2="+ratio_clip=0.2"
arg3="+lambda_0=1"
arg4="+reward_shaper=True"
arg5="+value_preprocessor=True"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Sixth run
arg1="+mini_batches=512"
arg2="+ratio_clip=0.8"
arg3="+lambda_0=0"
arg4="+reward_shaper=False"
arg5="+value_preprocessor=False"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"

#Sixth run
arg1="+mini_batches=512"
arg2="+ratio_clip=0.8"
arg3="+lambda_0=0"
arg4="+reward_shaper=True"
arg5="+value_preprocessor=True"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5"


#Seventh run
arg1="+mini_batches=512"
arg2="+ratio_clip=0.5"
arg3="+lambda_0=1"
arg4="+reward_shaper=False"
arg5="+value_preprocessor=False"
arg6="+lambda_1=0.9999"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5" "$arg6"

#Sixth run
arg1="+mini_batches=512"
arg2="+ratio_clip=0.8"
arg3="+lambda_0=1"
arg4="+reward_shaper=True"
arg5="+value_preprocessor=True"
arg6="+lambda_1=0.9999"

# Add more arguments as needed

# Run the Python script with the defined arguments
python "$python_script" "task=DianaTekken" "num_envs=2048" "headless=True" "$arg1" "$arg2" "$arg3" "$arg4" "$arg5" "$arg6"