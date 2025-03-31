#!/bin/bash
## sample script to run single node cpu task
#SBATCH -A m4669
## specify cpu or gpu node at next line
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 05:00:00
## node numbers
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
## change the job name to your script name
## SBATCH --output=../logs/%x_%j.log
## change the username to your username
#SBATCH --mail-user=dhanvib

## load required modules, if not found, install them perferably under /global/common/software/m4669
## nproc
module load conda
module load python
## assume you have a virtual environment in the current directory named env

# Activate conda venv. 

conda activate qaoa_w_sage

# echo "Running GA Job with Parameters:"
# echo "  Number of Qubits     : $N_QUBITS"
# echo "  Ansatz Reps          : $N_REPS"
# echo "  Generations          : $NUM_GENERATIONS"
# echo "  Mutation Probability : $MUTATION_PROB"
# echo "  Elitism              : $KEEP_ELITISM"
# echo "  Crossover Type       : $CROSSOVER_TYPE"

export PYTHONWARNINGS="ignore"

# Execute Python script with labeled parameters
# srun --cpu-bind=cores python cafqa_qaoa.py $N_QUBITS $N_REPS $NUM_GENERATIONS "$MUTATION_PROB" $KEEP_ELITISM $CROSSOVER_TYPE $SEED

#Knapsack
srun --cpu-bind=cores python knapsack.py $N_QUBITS $N_REPS $SEED

# QAOA+
# srun --cpu-bind=cores python new_ansatz_comparision.py $N_QUBITS $N_REPS $NUM_GENERATIONS "$MUTATION_PROB" $KEEP_ELITISM $CROSSOVER_TYPE $SEED
