#!/usr/bin/env bash
# Run from experiments/:  bash run_experiments.sh
# Edit args as needed.

# SPIQ MaxCut workflow: build graph → SPIQ → point selection → save to data/
echo "=== Running spiq_maxcut_workflow.py ==="
python spiq_maxcut_workflow.py --n-qubits 4 --reps 2 --n-gens 4 --seed 0 --device CPU

# SPIQ Knapsack workflow: build QUBO → SPIQ → point selection → save to data/
# echo "=== Running spiq_knapsack_workflow.py ==="
# python spiq_knapsack_workflow.py --n-items 8 --n-gens 100 --seed 0 --device CPU

# SPIQ biomarker workflow: PCBO feature selection → SPIQ → point selection → save to data/
# echo "=== Running spiq_biomarker_workflow.py ==="
# python spiq_biomarker_workflow.py --n-qubits 8 --select-n-features 3 --n-gens 100 --seed 0 --device CPU

# Statevector support-size reduction: SPIQ vs random (MaxCut)
# echo "=== Running statevec_reduction_maxcut.py ==="
# python statevec_reduction_maxcut.py --n-qubits 12 --n-gens 100 --seed 1 --num-seeds 2 --device CPU

# Statevector support-size reduction: SPIQ vs random (Knapsack)
# echo "=== Running statevec_reduction_knapsack.py ==="
# python statevec_reduction_knapsack.py --n-items 8 --n-gens 100 --seed 1 --num-seeds 2 --device CPU

# Full MaxCut multi-start: SPIQ + Red-QAOA + random/vanilla COBYLA comparison
# echo "=== Running full_opt_maxcut.py ==="
# python full_opt_maxcut.py --n-qubits 12 --n-gens 100 --seed 0 --device CPU
