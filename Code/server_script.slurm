#!/bin/bash
#SBATCH -J LM_Alloy # job name
#SBATCH -p normal # partition name
#SBATCH -N 1 # number of nodes
#SBATCH -n 30 # number of cores
#SBATCH --mem=100G  # memory allocation
#SBATCH -t 24:00:00 # wall time limit
#SBATCH -o transformer_train.o%j # standard-output file
#SBATCH -e transformer_train.e%j # standard-error file


# Print the start time.
echo "Training    began  at $(date)"
# Execute the training script.
python Path_to_your_train.py
# Print the end time.
echo "Training completed at $(date)"
