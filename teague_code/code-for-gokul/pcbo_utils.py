import sys
import time
import json
import argparse
import glob
from pathlib import Path
from itertools import combinations

import numpy as np
import qubovert as qv

import _solve_bruteforce


def load_features_and_corr_files(problem_dir):
    """Load and return the features and correlation coefficients for the given problem

    problem_dir  the directory holding the problem data

    This also return the `feature_to_idx` dictionary, it's important the order of the features
    in the json file is respected because this is how we will index into the numpy arrays
    holding the correlation coefficients.
    """
    with open(f"{problem_dir}/sampled_features.json", "r") as file:
        feature_set = json.load(file)
    feature_to_idx = {f: i for i, f in enumerate(feature_set)}

    first_corr_arr = np.load(f"{problem_dir}/three-body-cubo_first_order_coefs.npy")
    second_corr_arr = np.load(f"{problem_dir}/three-body-cubo_second_order_coefs.npy")
    third_corr_arr = np.load(f"{problem_dir}/three-body-cubo_third_order_coefs.npy")
    # print(f'Coefficient matrices: {first_corr_arr.shape}, {second_corr_arr.shape}, {third_corr_arr.shape}')
    return feature_set, feature_to_idx, first_corr_arr, second_corr_arr, third_corr_arr


def create_three_body_cubo(
    feature_list,
    first_corr_arr,
    second_corr_arr,
    third_corr_arr,
    feature_to_idx,
    select_n_features,
    first_order_weight=1,
    second_order_weight=1,
    third_order_weight=1,
    constraint_lambda=1000,
):
    """Construct a three-body-cubo PCBO problem

    Pass in the features and correlation coefficients, and also specify:
         select_n_features    the number of features you want to select from `feature_list`
         *_order_weight       the relative weighting of the 1st, 2nd, 3rd order terms
         constraint_lambda    the penalty applied to solutions with Hamming weight not equal to select_n_features
    """
    first_order_terms = {}
    try:
        for feature in feature_list:
            first_order_terms[(feature,)] = (
                -first_order_weight * first_corr_arr[feature_to_idx[feature]]
            )
    except IndexError:
        raise IndexError(str(len(feature_to_idx)))

    second_order_terms = {}
    for feature1, feature2 in combinations(feature_list, 2):
        idxs = list(sorted([feature_to_idx[feature1], feature_to_idx[feature2]]))
        second_order_terms[(feature1, feature2)] = (
            second_order_weight * second_corr_arr[idxs[0]][idxs[1]]
        )

    third_order_terms = {}
    for feature1, feature2, feature3 in combinations(feature_list, 3):
        idxs = list(
            sorted(
                [
                    feature_to_idx[feature1],
                    feature_to_idx[feature2],
                    feature_to_idx[feature3],
                ]
            )
        )
        third_order_terms[(feature1, feature2, feature3)] = (
            third_order_weight * third_corr_arr[idxs[0]][idxs[1]][idxs[2]]
        )

    # Finish creating the problem
    full_problem_dict = {**first_order_terms, **second_order_terms, **third_order_terms}
    variables = {feature: qv.boolean_var(feature) for feature in feature_list}
    pcbo_obj = qv.PCBO(full_problem_dict)
    pcbo_obj.add_constraint_eq_zero(
        sum(variables.values()) - select_n_features, lam=constraint_lambda
    )
    return pcbo_obj


def find_bruteforce_solution(pcbo, num_threads=1, chunk_size=10):
    """Find the globally optimal solution to the provided PCBO problem.

    Inputs:
        pcbo         a qubovert PCBO object
        num_threads  the number of parallel processes to use for the bruteforce search
        chunk_size   this helps tame the memory cost of the bruteforce search (only really relevant for large problems)
    """
    solve_start = time.time()
    bruteforce_soln = _solve_bruteforce.solve_pubo_bruteforce(
        pcbo, ncpus=num_threads, chunk_size=chunk_size
    )
    solve_end = time.time()
    solve_time = solve_end - solve_start
    print(f"\tFound bruteforce soln in {solve_time:.3f} secs")

    return bruteforce_soln, solve_time
