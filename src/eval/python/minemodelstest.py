#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of fitness function evaluation.

Authors: LEC group, LISIC Lab, Univ. of Littoral Opal Coast, France

Please cite this article if you use this code:
  Sara Tari, Sebastien Verel, and Mahmoud Omidvar.
  "PUBOi: A Tunable Benchmark with Variable Importance."
  In European Conference on Evolutionary Computation in Combinatorial Optimization (Part of EvoStar), pp. 175-190. Springer, Cham, 2022.
"""

import glob
import re
import sys
import os
import random
import copy
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from model_training import (
    combine_model_explanations,
    extract_number,
    linear_regression_with_shap,
    plot_aggregated_heatmap,
    plot_normalized_heatmap,
    plot_variability_heatmap,
    rf_regression_model,
    nn_regression,
    aggregate_metrics,
    save_aggregated_metrics_to_csv
)
sys.path.insert(0, '../../generator')
from walsh_expansion import WalshExpansion

# Define problem size
PROBLEM_SIZE = 14

# Number of ILS runs per instance
RUNS_PER_INSTANCE = 10

# Initialize global evaluation count
global evalcount

termcond = 30
runs = 10
perturbation_strength = 3

def random_soln(size):
    return [random.choice([-1, 1]) for _ in range(size)]

# Custom exception for evaluation limit
class EvaluationLimitExceeded(Exception):
    pass

def get_fitness(soln, f):
    # global evalcount
    # if evalcount >= 500:
    #     print("Eval limit reached for fitness ",  evalcount)
    #     raise EvaluationLimitExceeded
    # else:
    #     evalcount += 1
    #     fitness = f.eval(soln)
    global evalcount
    evalcount += 1
    fitness = f.eval(soln)
    return fitness

def local_search(soln, f, solutions):
    max_solutions = 500
    indices = list(range(PROBLEM_SIZE))
    random.shuffle(indices)
    improvement = True
    current_soln = copy.deepcopy(soln)
    best_fitness = get_fitness(current_soln, f)
    # Append the initial solution if under the limit
    if len(solutions) < max_solutions:
        solutions.append({'fitness': best_fitness, 'solution': current_soln})
    else:
        return current_soln
    while improvement is True:
        random.shuffle(indices)
        improvement = False
        for y in indices:
            neighbour = copy.deepcopy(current_soln)
            neighbour[y] = 1 if neighbour[y] == -1 else -1
            neighbour_fitness = get_fitness(neighbour, f)
            solutions.append({'fitness': neighbour_fitness, 'solution': neighbour})
            if neighbour_fitness < best_fitness:
                current_soln = copy.deepcopy(neighbour)
                best_fitness = neighbour_fitness
                improvement = True
    return current_soln

def termination_condition(solutions):
    if len(solutions) == 500:
        return True
    elif solutions[-1]['iterations_no_improv'] > termcond:
        return True
    else:
        return False

def perturb(soln, strength):
    for idx in range(strength):
        #idx = random.randint(0, len(soln) - 1)
        soln[idx] = 1 if soln[idx] == -1 else -1
    return soln

def ils(f, solutions):
    iterations_no_improv = 0
    max_solutions = 500
    soln = random_soln(PROBLEM_SIZE)
    soln = local_search(soln, f, solutions)
    best_soln = copy.deepcopy(soln)
    best_fitness = get_fitness(soln, f)
    current_soln = copy.deepcopy(soln)
    solutions.append({'fitness': best_fitness, 'solution': current_soln, 'iterations_no_improv': iterations_no_improv})

    while not termination_condition(solutions):
        new_soln = copy.deepcopy(current_soln)
        new_soln = perturb(new_soln, perturbation_strength)

        if len(solutions) >= max_solutions:
            break  # Ensure we don't exceed the solution limit
        fitness = get_fitness(new_soln, f)
        #iterations_no_improv += 1  # Increment before local search
        solutions.append({'fitness': fitness, 'solution': new_soln, 'iterations_no_improv': iterations_no_improv})
        new_soln = local_search(new_soln, f, solutions)
        current_fitness = get_fitness(new_soln, f)
        
        if current_fitness < best_fitness:
            iterations_no_improv = 0
            best_soln = copy.deepcopy(new_soln)
            best_fitness = current_fitness
        else:
            iterations_no_improv += 1

        solutions.append({'fitness': current_fitness, 'solution': new_soln, 'iterations_no_improv': iterations_no_improv})


def aggregate_explanations_across_runs(explanation_list):
    """
    Aggregate explanations across multiple runs.

    Parameters:
    - explanation_list: List of DataFrames containing explanations.

    Returns:
    - Aggregated DataFrame.
    """
    if not explanation_list:
        return pd.DataFrame()

    shapes = [df.shape for df in explanation_list]
    if len(set(shapes)) != 1:
        return pd.DataFrame()

    explanation_arrays = np.array([df.values for df in explanation_list])
    mean_explanations = np.mean(explanation_arrays, axis=0)
    return pd.DataFrame(mean_explanations, index=explanation_list[0].index, columns=explanation_list[0].columns)

def plot_and_save_heatmap(df, title, filename):
    """
    Plot and save a heatmap.

    Parameters:
    - df: DataFrame containing data for the heatmap.
    - title: Title of the heatmap.
    - filename: Path to save the heatmap image.

    Returns:
    - None
    """
    try:
        plt.figure(figsize=(30, 15))
        ax = sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 20})
        ax.set_yticklabels(range(1, PROBLEM_SIZE + 1), rotation=0)
        ax.set_ylabel('Variables in Class', fontsize=30)
        ax.set_xlabel('Models', fontsize=30)
        ax.set_title(title, fontsize=40)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Failed to plot heatmap {filename}: {e}")

def train_models(solution_file, degree, instance_number, run_number, term_iteration, 
                 all_shap_importances, all_pfi_importances, all_lime_importances, 
                 metrics_aggregated, r2_scores):
    """
    Train ML models on a given solution file and extract feature importances.

    Parameters:
    - solution_file: Path to the solution CSV file.
    - degree: Degree information (e.g., 'deg2', 'deg10').
    - instance_number: Instance number.
    - run_number: Run number.
    - term_iteration: Current term iteration.
    - all_shap_importances: List to store SHAP importances.
    - all_pfi_importances: List to store PFI importances.
    - all_lime_importances: List to store LIME importances.
    - metrics_aggregated: Dictionary to store aggregated metrics.
    - r2_scores: Dictionary to store R2 scores.

    Returns:
    - None
    """
    try:
        # Read the solution file directly
        df = pd.read_csv(f'ils/mined_solutions/{solution_file}')
        
        # Check if required columns are present
        expected_columns = ['WalshFitness'] + [f'Variable{i+1}' for i in range(PROBLEM_SIZE)]
        if not all(col in df.columns for col in expected_columns):
            print(f"File '{solution_file}' is missing expected columns. Skipping.")
            return

        # The data is already in the format needed, so we can proceed directly with training
        # Rename target and feature columns if needed
        X = df.drop('WalshFitness', axis=1)  # Feature matrix
        y = df['WalshFitness']              # Target variable

        # Train ML Models
        # Linear Regression with SHAP
        shap_lr = linear_regression_with_shap(X, y, degree, instance_number, run_number, term_iteration)
        if shap_lr is not None:
            all_shap_importances.append(shap_lr)

        # Random Forest Regressor
        shap_rf = rf_regression_model(X, y, degree, instance_number, run_number, term_iteration)
        if shap_rf is not None:
            all_shap_importances.append(shap_rf)

        # MLP Regressor
        shap_nn = nn_regression(X, y, degree, instance_number, run_number, term_iteration)
        if shap_nn is not None:
            all_shap_importances.append(shap_nn)

    except Exception as e:
        print(f"Error during ML training for file '{solution_file}': {e}")

def process_all_runs(degrees_info, path, metrics_aggregated, r2_scores, all_shap_importances, all_pfi_importances, all_lime_importances):
    """
    Process all CSV files with multiple runs per instance across different degrees.

    Parameters:
    - degrees_info: dict, mapping degrees to a list or range of instance numbers.
    - path: str, directory path where CSV files are located.
    - metrics_aggregated: dict, to store aggregated metrics.
    - r2_scores: dict, to store R2 scores per model.
    - all_shap_importances, all_pfi_importances, all_lime_importances: lists to store aggregated importances.

    Returns:
    - None
    """
    for degree, instance_range in degrees_info.items():
        for instance_number in instance_range:
            # Define the glob pattern to include run numbers
            pattern = f'puboi_{degree}_{instance_number}_run*.csv'
            # Retrieve all matching run files for the current instance
            instance_files = sorted(
                glob.glob(os.path.join(path, pattern)),
                key=lambda x: int(
                    re.search(r'puboi_deg\d+_\d+_run(\d+)\.csv', os.path.basename(x)).group(1)
                ) if re.search(r'puboi_deg\d+_\d+_run(\d+)\.csv', os.path.basename(x)) else 0
            )

            # Check if all 10 runs are present
            expected_runs = 10
            if len(instance_files) != expected_runs:
                print(f"Warning: Expected {expected_runs} runs for Degree: {degree}, Instance: {instance_number}, but found {len(instance_files)} runs.")
            
            print(f"\nProcessing Degree: {degree}, Instance: {instance_number}")

            # Initialize aggregation dictionaries for this instance
            models_aggregated_shap_values = {}
            models_aggregated_pfi_values = {}
            models_aggregated_lime_values = {}

            # Load ground truth importances (assuming it's the same across runs)
            ground_truth_filepath = f'../../../instances/small/puboi_1000seed_{degree}/puboi_{instance_number}_vars.csv'
            if not os.path.exists(ground_truth_filepath):
                print(f"Error: Ground truth file {ground_truth_filepath} does not exist. Skipping Instance: {instance_number}.")
                continue  # Skip to the next instance

            ground_truth_df = pd.read_csv(ground_truth_filepath)
            vars_in_class_to_feature = {i: f'Variable{i+1}' for i in range(14)}
            ground_truth_df['feature_name'] = ground_truth_df['vars_in_class'].map(vars_in_class_to_feature)
            ground_truth_importances = pd.Series(
                data=ground_truth_df['var_importance'].values,
                index=ground_truth_df['feature_name']
            )

            # Recode ground truth importances if needed
            ground_truth_importances_recode = ground_truth_importances.map({0: 1, 1: 0})

            # Iterate over each run file
            for run_file in instance_files:
                run_match = re.match(r'puboi_deg(\d+)_(\d+)_run(\d+)\.csv', os.path.basename(run_file))
                if not run_match:
                    print(f"Warning: File {run_file} does not match the expected pattern. Skipping.")
                    continue

                run_degree, run_instance, run_number = run_match.groups()
                print(f"  Processing Run {run_number} for Instance {instance_number}")

                # Read the run-specific CSV file
                df = pd.read_csv(run_file)

                # Train and evaluate all models on this run
                # You can uncomment the Gradient Boosting section if needed
                # best_gb = gb_model_with_gridsearch(df, instance_number)
                # gb_regression_model(df, best_gb, instance_number, models_aggregated_shap_values, models_aggregated_pfi_values, models_aggregated_lime_values, ground_truth_importances_recode)

                # Linear Regression
                linear_regression_with_shap(
                    df=df,
                    instance_number=instance_number,
                    models_aggregated_shap_values=models_aggregated_shap_values,
                    models_aggregated_pfi_values=models_aggregated_pfi_values,
                    models_aggregated_lime_values=models_aggregated_lime_values,
                    ground_truth_importances=ground_truth_importances_recode,
                    metrics_aggregated=metrics_aggregated,
                    r2_scores=r2_scores,
                    run_number=run_number
                )

                # Random Forest
                rf_regression_model(
                    df=df,
                    instance_number=instance_number,
                    models_aggregated_shap_values=models_aggregated_shap_values,
                    models_aggregated_pfi_values=models_aggregated_pfi_values,
                    models_aggregated_lime_values=models_aggregated_lime_values,
                    ground_truth_importances=ground_truth_importances_recode,
                    metrics_aggregated=metrics_aggregated,
                    r2_scores=r2_scores,
                    run_number=run_number
                )

                # Neural Network
                nn_regression(
                    df=df,
                    instance_number=instance_number,
                    models_aggregated_shap_values=models_aggregated_shap_values,
                    models_aggregated_pfi_values=models_aggregated_pfi_values,
                    models_aggregated_lime_values=models_aggregated_lime_values,
                    ground_truth_importances=ground_truth_importances_recode,
                    metrics_aggregated=metrics_aggregated,
                    r2_scores=r2_scores,
                    run_number=run_number

                )

            # After processing all runs for the current instance, aggregate explanations
            print(f"Aggregating explanations for Instance: {instance_number}")

            combined_shap_values, combined_pfi_values, combined_lime_values = combine_model_explanations(
                models_aggregated_shap_values,
                models_aggregated_pfi_values,
                models_aggregated_lime_values
            )

            # Clean combined SHAP values
            combined_shap_values = combined_shap_values.apply(pd.to_numeric, errors='coerce')
            combined_shap_values.replace([np.inf, -np.inf], np.nan, inplace=True)
            combined_shap_values.fillna(value=0, inplace=True)
            combined_shap_values.dropna(inplace=True)

            variables = {i: f'Variable{i+1}' for i in range(14)}
            # Convert to DataFrames
            shap_df = pd.DataFrame(combined_shap_values, index=variables)
            pfi_df = pd.DataFrame(combined_pfi_values, index=variables)
            lime_df = pd.DataFrame(combined_lime_values, index=variables)

            # Append to the aggregated lists
            all_shap_importances.append(shap_df)
            all_pfi_importances.append(pfi_df)
            all_lime_importances.append(lime_df)

            # Define heatmap directory for this instance
            heatmap_dir = os.path.join('ils', str(instance_number), 'heatmap')
            os.makedirs(heatmap_dir, exist_ok=True)

            # Plot SHAP heatmap
            plt.figure(figsize=(30, 15))
            ax = sns.heatmap(combined_shap_values, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 20})
            # Set y-axis labels
            ax.set_yticklabels(range(1, 15), rotation=0)
            ax.set_ylabel('Variables in Class', fontsize=30)
            ax.set_xlabel('Models', fontsize=30)
            ax.set_title(f'Model Performance with SHAP for ILS - Instance {instance_number}', fontsize=40)
            plt.savefig(os.path.join(heatmap_dir, 'shap-median.png'), dpi=200)
            plt.close()

            # Plot PFI heatmap
            plt.figure(figsize=(30, 15))
            ax = sns.heatmap(combined_pfi_values, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 20})
            # Set y-axis labels
            ax.set_yticklabels(range(1, 15), rotation=0)
            ax.set_ylabel('Variables in Class', fontsize=30)
            ax.set_xlabel('Models', fontsize=30)
            ax.set_title(f'Model Performance with PFI for ILS - Instance {instance_number}', fontsize=40)
            plt.savefig(os.path.join(heatmap_dir, 'pfi-median.png'), dpi=200)
            plt.close()

            # Plot LIME heatmap
            plt.figure(figsize=(30, 15))
            ax = sns.heatmap(combined_lime_values, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 20})
            # Set y-axis labels
            ax.set_yticklabels(range(1, 15), rotation=0)
            ax.set_ylabel('Variables in Class', fontsize=30)
            ax.set_xlabel('Models', fontsize=30)
            ax.set_title(f'Model Performance with LIME for ILS - Instance {instance_number}', fontsize=40)
            plt.savefig(os.path.join(heatmap_dir, 'lime-median.png'), dpi=200)
            plt.close()
        # After processing all runs, aggregate the explanations
    heatmap_dir = 'ils/aggregated_heatmaps'
    os.makedirs(heatmap_dir, exist_ok=True)
##solutions = []
def main():
    if len(sys.argv) != 4:
        print("Usage: python mine-models-ils.py <termcond> <runs_per_instance> <perturbation_strength>")
        sys.exit(1)

    termcond, runs_per_instance, perturbation_strength = map(int, sys.argv[1:])

    output_dir = 'ils-split/mined_solutions'
    os.makedirs(output_dir, exist_ok=True)
    global evalcount

    degrees_info = {
        'deg2': range(1000, 1030),   # 30 instances
        'deg10': range(1050, 1080)  # 30 instances
    }

    all_shap_importances = []
    all_pfi_importances = []
    all_lime_importances = []
    metrics_aggregated = {
        'SHAP': {'Spearman': [], 'KendallTau': [], 'PrecisionAtK': []},
        'LIME': {'Spearman': [], 'KendallTau': [], 'PrecisionAtK': []},
        'PFI': {'Spearman': [], 'KendallTau': [], 'PrecisionAtK': []}
    }
    r2_scores = {'Linear Regression': [], 'MLP': [], 'Random Forest': []}

    for degree, instance_range in degrees_info.items():
        for instance_number in instance_range:
            pattern = f'../../../instances/small/puboi_1000seed_{degree}/puboi_{instance_number}.json'
            instance_files = sorted(
                glob.glob(pattern),
                key=lambda x: int(
                    re.search(r'puboi_(\d+)', os.path.basename(x)).group(1)
                ) if re.search(r'puboi_(\d+)', os.path.basename(x)) else 0
            )

            print(f"Processing Degree: {degree}, Instance: {instance_number}")
            
            f = WalshExpansion(PROBLEM_SIZE)

            if not instance_files:
                print(f"No instance files found for instance number {instance_number} in degree {degree}. Skipping.")
                continue

            match_found = False
            for input_name in instance_files:
                match = re.search(r'puboi_(\d+)\.json', input_name)
                if match:     
                    try:
                        f.load(input_name)
                        print(f"Loaded WalshExpansion from '{input_name}'")
                        match_found = True

                        for run in range(1, runs_per_instance + 1):
                            solutions = []
                            evalcount = 0
                            #random.seed(run)  # Seed for reproducibility

                            print(f"Loaded WalshExpansion from '{input_name}', run {run}")

                            try:
                                ils(f, solutions)
                                # After ILS and before saving to CSV
                                if len(solutions) != 500:
                                    print(f"Warning: Expected 500 solutions, but got {len(solutions)} for run {run} of instance {instance_number}. Truncating to 500.")
                                    solutions = solutions[:500]
                                print(f"Completed ILS Run {run}/{runs_per_instance} for Instance {instance_number}")
                            except Exception as e:
                                print(f"An error occurred during ILS for run {run} of instance {instance_number}: {e}")
                                continue

                            # Save solutions to a CSV file after each run
                            output_file = os.path.join(output_dir, f'puboi_{degree}_{instance_number}_run{run}.csv')
                            try:
                                # Convert solutions to DataFrame
                                df = pd.DataFrame(solutions)
                                if df.empty:
                                    print(f"No solutions found for run {run} of instance {instance_number}. Skipping save.")
                                    continue

                                # Split solution vectors into separate columns
                                solution_df = pd.DataFrame(df['solution'].tolist(), columns=[f'Variable{i+1}' for i in range(PROBLEM_SIZE)])

                                # Concatenate fitness and solution columns
                                df_final = pd.concat([df['fitness'], solution_df], axis=1)

                                # Assign column names
                                column_names = ['WalshFitness'] + [f'Variable{i+1}' for i in range(PROBLEM_SIZE)]
                                df_final.columns = column_names

                                # Save to CSV with headers
                                df_final.to_csv(output_file, index=False)
                                print(f"Saved mined solutions to '{output_file}'")
                            except Exception as e:
                                print(f"Failed to save solutions for run {run} of instance {instance_number}: {e}")
                                continue
                    except Exception as e:
                        print(f"Failed to load WalshExpansion from '{input_name}': {e}")
                        continue
                else:
                    print(f"Filename '{input_name}' does not match the expected pattern. Skipping.")
                    continue

            if not match_found:
                print(f"No matching instance file found for instance number {instance_number} in degree {degree}. Continuing to next instance.")
                continue

    print("Completed all ILS runs.")


if __name__ == "__main__":
    #evalcount = 0
    main()
