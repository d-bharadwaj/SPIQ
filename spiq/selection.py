"""Point-selection strategies for SPIQ multi-start initialization."""

from __future__ import annotations

import math
import warnings

import numpy as np
from qiskit_aer.primitives import EstimatorV2 as Estimator
from sklearn.cluster import KMeans


def compute_gradient_norm(qaoa_object, parameters):
    """L2 gradient norm via the parameter-shift rule (stabilizer simulator)."""
    shift = np.pi / 2
    gradients = []
    estimator = Estimator(
        options={"backend_options": {"method": "stabilizer", "device": "CPU"}}
    )

    parameters = np.asarray(parameters, dtype=float)
    for i in range(len(parameters)):
        params_plus = parameters.copy()
        params_plus[i] += shift
        params_minus = parameters.copy()
        params_minus[i] -= shift

        job = estimator.run(
            [
                (qaoa_object.pcirc, qaoa_object.cost_hamiltonian, params_plus),
                (qaoa_object.pcirc, qaoa_object.cost_hamiltonian, params_minus),
            ]
        )
        results = job.result()
        gradients.append((results[0].data.evs - results[1].data.evs) / 2.0)

    return float(np.linalg.norm(gradients))


def deduplicate_parameters(parameters, fitness_values, tolerance=1e-10):
    """Remove duplicate SPIQ parameter sets."""
    parameters = np.asarray(parameters)
    fitness_values = np.asarray(fitness_values)

    unique_params = []
    unique_fitness = []
    original_indices = []

    for i, (params, fitness) in enumerate(zip(parameters, fitness_values)):
        if any(np.allclose(params, existing, atol=tolerance) for existing in unique_params):
            continue
        unique_params.append(params)
        unique_fitness.append(fitness)
        original_indices.append(i)

    return np.asarray(unique_params), np.asarray(unique_fitness), original_indices


def fixed_interval_selection(
    best_spiq_parameters,
    best_spiq_fitness_values,
    num_select=3,
    stride=3,
):
    """Fixed-interval energy selection over unique fitness values."""
    fitness_list = list(best_spiq_fitness_values)
    unique_fitness = np.unique(fitness_list)
    selected_fitness = list(unique_fitness[::stride][:num_select])

    selected_indices = [fitness_list.index(value) for value in selected_fitness]
    selected_parameters = np.asarray(best_spiq_parameters)[selected_indices]

    return selected_parameters, np.asarray(selected_fitness)


def k_gaps_selection(
    best_spiq_parameters,
    best_spiq_fitness_values,
    qaoa_object,
    num_select=3,
    rng=None,
):
    """K-means clustering with energy stratification and gradient filtering."""
    if rng is None:
        rng = np.random.default_rng(0)

    unique_params, unique_fitness, _ = deduplicate_parameters(
        best_spiq_parameters, best_spiq_fitness_values
    )
    if len(unique_params) == 0:
        raise ValueError("No SPIQ points available for clustering selection.")

    n_clusters = max(1, min(math.ceil(len(unique_params) / 10), len(unique_params)))
    x_periodic = np.column_stack(
        [np.cos(unique_params * np.pi / 2), np.sin(unique_params * np.pi / 2)]
    )

    if n_clusters == 1:
        cluster_labels = np.zeros(len(unique_params), dtype=int)
    else:
        cluster_labels = KMeans(
            n_clusters=n_clusters, random_state=42, n_init=10
        ).fit_predict(x_periodic)

    sampled_candidates = []
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_params = unique_params[cluster_mask]
        cluster_fitness = unique_fitness[cluster_mask]
        if len(cluster_params) == 0:
            continue

        sorted_indices = np.argsort(cluster_fitness)
        sorted_params = cluster_params[sorted_indices]
        sorted_fitness = cluster_fitness[sorted_indices]
        unique_energies = np.unique(sorted_fitness)

        if len(unique_energies) == 1:
            n_sample = min(3, len(sorted_params))
            sample_indices = rng.choice(len(sorted_params), n_sample, replace=False)
            for idx in sample_indices:
                sampled_candidates.append(
                    {"params": sorted_params[idx], "energy": float(sorted_fitness[idx])}
                )
            continue

        third = max(1, len(unique_energies) // 3)
        low_threshold = unique_energies[third - 1]
        mid_threshold = unique_energies[min(2 * third - 1, len(unique_energies) - 1)]

        layer_low = np.where(sorted_fitness <= low_threshold)[0]
        layer_mid = np.where(
            (sorted_fitness > low_threshold) & (sorted_fitness <= mid_threshold)
        )[0]

        if len(layer_low) > 0:
            n_low = min(2, len(layer_low))
            low_sample_indices = list(rng.choice(layer_low, n_low, replace=False))
            for idx in low_sample_indices:
                sampled_candidates.append(
                    {"params": sorted_params[idx], "energy": float(sorted_fitness[idx])}
                )
        else:
            low_sample_indices = []

        if len(layer_mid) > 0:
            idx = int(rng.choice(layer_mid, 1)[0])
            sampled_candidates.append(
                {"params": sorted_params[idx], "energy": float(sorted_fitness[idx])}
            )
        elif len(layer_low) > 2:
            extra = [i for i in layer_low if i not in low_sample_indices]
            if extra:
                idx = int(rng.choice(extra, 1)[0])
                sampled_candidates.append(
                    {"params": sorted_params[idx], "energy": float(sorted_fitness[idx])}
                )

    valid_candidates = []
    for candidate in sampled_candidates:
        grad_norm = compute_gradient_norm(
            qaoa_object, candidate["params"] * (np.pi / 2)
        )
        if grad_norm > 1e-10:
            candidate["grad_norm"] = grad_norm
            valid_candidates.append(candidate)

    if not valid_candidates:
        warnings.warn("No valid candidates with non-zero gradients found.")
        return None

    valid_candidates.sort(key=lambda item: item["energy"])
    n_lowest = min(2, len(valid_candidates), num_select)
    selected = valid_candidates[:n_lowest]
    remaining = valid_candidates[n_lowest:]

    if remaining and len(selected) < num_select:
        n_random = min(num_select - len(selected), len(remaining))
        for idx in rng.choice(len(remaining), n_random, replace=False):
            selected.append(remaining[idx])

    selected = selected[:num_select]
    return (
        np.asarray([item["params"] for item in selected]),
        np.asarray([item["energy"] for item in selected]),
        np.asarray([item["grad_norm"] for item in selected]),
    )
