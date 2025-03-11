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

# for seed in {1..10}; do
#     sbatch --output=../logs/maxcut_graphs_eval/approx_ratio_14_qb_k_reg_maxcut_%j.log --export=ALL,N_QUBITS=14,N_REPS=5,NUM_GENERATIONS=2000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed sweep_run.sh
# done


for rep in 1 2 5; do
    for seed in {1..5}; do
        sbatch --output=../logs/rep_sweep/2000_gens_Rep_Sweep_%j.log --export=ALL,N_QUBITS=10,N_REPS=$rep,NUM_GENERATIONS=2000,MUTATION_PROB="0.25 0.01",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed sweep_run.sh
    done
done