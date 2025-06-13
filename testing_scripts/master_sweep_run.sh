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

#Multi Start
# for seed in {1..1}; do
#     sbatch --qos=premium --output=../logs/CAFQA_Analysis/Approx_Ratio/Maxcut/approx_ratio_COBYLA_best_spaced_5_point_analysis_%j.log --export=ALL,N_QUBITS=12,N_REPS=2,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed,NOISE=0 sweep_run.sh
# done


#Rep Sweep 
# for rep in 1 2 5; do
#     for seed in {1..5}; do
#         # sbatch --output=../logs/rep_sweep/2000_gens_Rep_Sweep_%j.log --export=ALL,N_QUBITS=10,N_REPS=$rep,NUM_GENERATIONS=2000,MUTATION_PROB="0.25 0.01",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed sweep_run.sh
#         sbatch --output=../logs/rep_sweep/knapsack/kp_rep_sweep_%j.log --export=ALL,N_QUBITS=6,N_REPS=$rep,SEED=$seed sweep_run.sh
#     done
# done

# # Knapsack
# for seed in {1..1}; do
#     sbatch --qos=premium --output=../logs/maxcut_graphs_eval/COBYQA_knapsack_%j.log --export=ALL,N_QUBITS=8,N_REPS=2,SEED=$seed sweep_run.sh
# done
 
# # Ansatz Comparision
# for seed in {1..3}; do
    # sbatch --qos=premium --output=../logs/ansatz_comparision/ma_qaoa_plus_%j.log --export=ALL,N_QUBITS=12,N_REPS=2,NUM_GENERATIONS=500,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed sweep_run.sh
# done

#Teague Code
# sbatch --qos=premium --output=../logs/SPSA_teague_%j.log sweep_run.sh

# Noisy Sim 
    # sbatch --qos=premium --output=../logs/maxcut_graphs_eval/noisy/noisy_14_qb_complete_%j.log --export=ALL,N_QUBITS=14,N_REPS=2,NUM_GENERATIONS=2000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=1,NOISE=1 sweep_run.sh

# Optimizer sweep
    # sbatch --qos=premium --output=../logs/optimizer_sweep/noise_sweep_%j.log --export=ALL,N_QUBITS=10,N_REPS=2,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=0,NOISE=0 sweep_run.sh






# Multi Start Final Data Collectio - MaxCut
for num_qubits in 12 16; do
    for seed in {1..1}; do
        sbatch --output=../logs/Final_Data_Collection/Maxcut/Less_Reps/result_$seed_%j.log --export=ALL,N_QUBITS=$num_qubits,N_REPS=1,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed,NOISE=0 sweep_run.sh
    done
done

# # # Multi Start Final Data Collection - Knapsack
# for num_qubits in 9 12; do
#     for seed in {1..10}; do
#         sbatch --output=../logs/Final_Data_Collection/Knapsack/result_$seed_%j.log --export=ALL,N_QUBITS=$num_qubits,N_REPS=2,NUM_GENERATIONS=1000,MUTATION_PROB="0.25 0.01 ",KEEP_ELITISM=5,CROSSOVER_TYPE="single_point",SEED=$seed,NOISE=0 sweep_run.sh
#     done
# done