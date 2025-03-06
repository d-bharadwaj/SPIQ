#!/bin/bash

# for num_gens in 500 1000 1500 2000; do
#     sbatch --qos=premium --output=../logs/ga_sweep/Gen_Sweep_%j.log --export=ALL,N_QUBITS=15,N_REPS=1,NUM_GENERATIONS=$num_gens,MUTATION_PROB="0.25 0.01",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point" sweep_run.sh
# done

# for mutation_prob in "0.25 0.01" "0.25 0.02" "0.25 0.03" "0.25 0.04"; do
#     sbatch --output=../logs/ga_sweep/Mut_Sweep_%j.log --export=ALL,N_QUBITS=15,N_REPS=1,NUM_GENERATIONS=1000,MUTATION_PROB="$mutation_prob",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point" sweep_run.sh
# done

# for keep_elitism in 5 10 15 20; do
#     sbatch --output=../logs/ga_sweep/Elit_Sweep_%j.log --export=ALL,N_QUBITS=15,N_REPS=1,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01",KEEP_ELITISM=$keep_elitism,CROSSOVER_TYPE="single_point" sweep_run.sh
# done

# for crossover_type in "single_point" "two_points" "uniform" "scattered"; do
#     sbatch --output=../logs/ga_sweep/Crossover_Sweep_%j.log --export=ALL,N_QUBITS=15,N_REPS=1,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE=$crossover_type sweep_run.sh
# done

for seed in {1..100}; do
    sbatch --output=../logs/graphs_sweep/15_qb_graph_sweep_%j.log --export=ALL,N_QUBITS=15,N_REPS=1,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed sweep_run.sh
done
