# model_training.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full space experiments

"""

# Standard library imports
from datetime import datetime
import glob
import json
import logging
import os
import sys
import time
import re
import argparse
import itertools


# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata, iqr
from scipy.stats import spearmanr, kendalltau
import shap
from sklearn.discriminant_analysis import StandardScaler
import torch
from xai import compute_faithfulness_regression, process_file, compute_faithfulness_pfi
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer


# OpenXAI imports
# from openxai import Evaluator
# from openxai.explainers.perturbation_methods import get_perturb_method
# from openxai.explainer import PFI as pfi

# Import necessary libraries for metrics
from scipy.stats import spearmanr, kendalltau
from itertools import combinations

# Local imports
sys.path.insert(0, '../../generator')
from walsh_expansion import WalshExpansion

from explainer_utils import SHAPExplainerWrapper, LIMEExplainerWrapper, get_shap_explainer, init_lime



# Directory to save hyperparameters
#HYPERPARAMS_DIR = 'hyperparams'
#os.makedirs(HYPERPARAMS_DIR, exist_ok=True)

# # Configure logging
# puboilog = logging.basicConfig(
#     filename='puboi_process.log',
#     filemode='a',
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     level=logging.INFO
# )
# Initialize aggregation containers organized by degree



def compute_lime_importance(model, X_test, y_test, feature_names):
    """
    Compute LIME feature importances.
    """
    lime_explainer = LimeTabularExplainer(
        training_data=X_test.values,
        feature_names=feature_names,
        mode='regression',
        discretize_continuous=False
    )
    lime_importances = pd.Series(0, index=feature_names)
    for i in range(len(X_test)):
        exp = lime_explainer.explain_instance(
            X_test.iloc[i].values,
            model.predict,
            num_features=len(feature_names)
        )
        for feature, importance in exp.as_list():
            lime_importances[feature] += importance
    lime_importances /= len(X_test)
    return lime_importances
# Global variable to store aggregated SHAP values
#models_aggregated_shap_values = {}

def setup_logger():
    """
    Sets up a logger that writes to a file based on the script name and current timestamp.
    
    Returns:
        logger: Configured logger.
    """
    # Determine the script name
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    # Create a logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name using the script name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create a logger with the script name
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
        # Create a file handler to write logs to the file
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Define a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger()

def bit_to_solution(bit):
    """
    Convert a bit character ('0' or '1') to -1 or 1.

    Parameters:
    - bit: str, bit character ('0' or '1').

    Returns:
    - int, -1 if bit is '0', else 1.
    """
    return -1 if bit == '0' else 1


def bit_to_integer(bit):
    """
    Convert a bit character ('0' or '1') to integer 0 or 1.

    Parameters:
    - bit: str, bit character ('0' or '1').

    Returns:
    - int, 0 or 1.
    """
    return int(bit)


def pad_array(arr, target_shape):
    """
    Pad the input array to the target shape with the last row repeated.

    Parameters:
    - arr: numpy array to pad.
    - target_shape: tuple, the target shape to pad to.

    Returns:
    - Padded numpy array.
    """
    padding_shape = (target_shape[0] - arr.shape[0], arr.shape[1])
    padding = np.tile(arr[-1], (padding_shape[0], 1))
    return np.vstack([arr, padding])

def aggregate_lime_values(lime_importances_list, features):
    """
    Aggregate LIME importances across folds, taking the absolute value.

    Parameters:
    - lime_values_list: List of arrays, each containing per-fold LIME importances.
    - feature_names: List or Index of feature names corresponding to the importances.

    Returns:
    - Aggregated LIME importances as a pandas Series.
    """
    print(lime_importances_list)
        # Filter out non-array elements and log warnings
    filtered_explanations = []
    for idx, values in enumerate(lime_importances_list):
        if isinstance(values, np.ndarray):
            filtered_explanations.append(values)
        else:
            print(f"Warning: Fold {idx+1} has invalid explanation type: {type(values)}. Skipping.")
    
    if not filtered_explanations:
        raise ValueError("No valid explanation arrays to aggregate.")
    
    # Check for consistent shapes
    shapes = [values.shape for values in filtered_explanations]
    if len(set(shapes)) != 1:
        print("Warning: Not all LIME explanation value arrays have the same shape!")
        print("Shapes found:", shapes)
        max_samples = max(shape[0] for shape in shapes)
        num_features = shapes[0][1]  # Assuming all have the same number of features
        
        # Pad arrays with fewer samples to match the maximum number of samples
        for i in range(len(filtered_explanations)):
            if filtered_explanations[i].shape[0] < max_samples:
                pad_width = max_samples - filtered_explanations[i].shape[0]
                # Pad with zeros at the end along the samples axis
                filtered_explanations[i] = np.pad(
                    filtered_explanations[i],
                    pad_width=((0, pad_width), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
    
    # Convert list of arrays to 3D numpy array
    #explanations_array = np.array(filtered_explanations)
    # Convert list to array: shape (n_folds, n_features)
    lime_importances_array = np.array(filtered_explanations)  # Each element should be (n_features,)
    
    # Take the absolute value
    lime_importances_array = np.abs(lime_importances_array)
    
    # Compute median across folds
    median_lime = np.median(lime_importances_array, axis=0)  # Resulting shape: (n_features,)

    median__lime =  np.median(median_lime,axis=0)
    
    # Create Series with feature names
    lime_importances_series = pd.Series(data=median__lime, index=features)
    
    return lime_importances_series


def aggregate_explanations(explanation_values_list, use_absolute=True):
    
    # Filter out non-array elements and log warnings
    filtered_explanations = []
    for idx, values in enumerate(explanation_values_list):
        if isinstance(values, np.ndarray):
            filtered_explanations.append(values)
        else:
            print(f"Warning: Fold {idx+1} has invalid explanation type: {type(values)}. Skipping.")
    
    if not filtered_explanations:
        raise ValueError("No valid explanation arrays to aggregate.")
    
    # Check for consistent shapes
    shapes = [values.shape for values in filtered_explanations]
    if len(set(shapes)) != 1:
        print("Warning: Not all explanation value arrays have the same shape!")
        print("Shapes found:", shapes)
        max_samples = max(shape[0] for shape in shapes)
        num_features = shapes[0][1]  # Assuming all have the same number of features
        
        # Pad arrays with fewer samples to match the maximum number of samples
        for i in range(len(filtered_explanations)):
            if filtered_explanations[i].shape[0] < max_samples:
                pad_width = max_samples - filtered_explanations[i].shape[0]
                # Pad with zeros at the end along the samples axis
                filtered_explanations[i] = np.pad(
                    filtered_explanations[i],
                    pad_width=((0, pad_width), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
    
    # Convert list of arrays to 3D numpy array
    explanations_array = np.array(filtered_explanations)
    
    if use_absolute:
        explanations_array = np.abs(explanations_array)
    
    # First aggregation: median over samples within each fold
    median_over_samples = np.median(explanations_array, axis=1)
    # Second aggregation: median over folds
    aggregated_values = np.median(median_over_samples, axis=0)

       # After aggregation
    if np.isscalar(aggregated_values):
        print("Warning: Aggregated explanations returned a scalar. Converting to list.")
        aggregated_values = [aggregated_values]
    
    return aggregated_values



def aggregate_pfi_values(pfi_values_list, features):
    """
    Aggregate PFI values from multiple folds using lists directly.
    Takes absolute values, then median across samples, then median across folds.
    
    Parameters:
    - pfi_values_list: List of lists, where each inner list represents PFI values for a fold.
    
    Returns:
    - aggregated_pfi_values: List of aggregated PFI values, one for each feature.

    
    variable1 - [n_repeats]
    .....
    variable14 - [n_repeats]
    """
    print(pfi_values_list)
    # Step 1: Take the absolute values for each sample across folds
  #  pfi_values_abs = [[np.abs(sample) for sample in fold] for fold in pfi_values_list]
    pfi_abs = np.abs(pfi_values_list)
    # Step 2: Calculate the median across samples for each fold
    median_across_samples = [np.median(fold, axis=1) for fold in pfi_abs]

    # Step 3: Calculate the median across folds for each feature
    aggregated_pfi_values = np.median(median_across_samples, axis=0)

    # Create Series with feature names
    pfi_importances_series = pd.Series(data=aggregated_pfi_values, index=features)
    
    return pfi_importances_series


def combine_model_explanations(models_shap_dict, models_pfi_dict, models_lime_dict):
    """
    Combine aggregated SHAP, PFI, and LIME values from different models into DataFrames.

    Parameters:
    - models_shap_dict: dict of SHAP values.
    - models_pfi_dict: dict of PFI values.
    - models_lime_dict: dict of LIME values.

    Returns:
    - combined_shap_df: DataFrame of SHAP values.
    - combined_pfi_df: DataFrame of PFI values.
    - combined_lime_df: DataFrame of LIME values.
    """
    combined_shap_df = pd.DataFrame(models_shap_dict)
    combined_shap_df.index.name = 'Variable'

    combined_pfi_df = pd.DataFrame(models_pfi_dict)
    combined_pfi_df.index.name = 'Variable'

    combined_lime_df = pd.DataFrame(models_lime_dict)
    combined_lime_df.index.name = 'Variable'

    return combined_shap_df, combined_pfi_df, combined_lime_df


def combine_model_shap_values(models_dict):
    """
    Combine aggregated SHAP values from different models into a DataFrame.

    Parameters:
    - models_dict: dictionary where keys are model names and values are aggregated SHAP values.

    Returns:
    - combined_df: pandas DataFrame containing combined SHAP values.
    """
    combined_df = pd.DataFrame(models_dict)
    combined_df.index.name = 'Variable'
    return combined_df
def binary_combinations(f, length):
    """
    Generate all combinations of binary strings and compute their fitness.

    Parameters:
    - f: WalshExpansion instance, the fitness function.
    - length: int, number of variables.

    Returns:
    - df: pandas DataFrame containing WalshFitness and Variable1 to Variable14.
    """
    # Generate all combinations of binary strings of given length
    binary_combinations = itertools.product([0, 1], repeat=length)
    # Convert each combination to a list of integers and then to strings
    binary_strings = [''.join(map(str, combination)) for combination in binary_combinations]

    # Create a DataFrame with a single column containing binary strings
    df = pd.DataFrame({'Binary': binary_strings})

    # Convert binary strings to solutions and integer bits
    df['Solution'] = df['Binary'].apply(lambda x: [bit_to_solution(bit) for bit in x])
    df['IntegerBits'] = df['Binary'].apply(lambda x: [bit_to_integer(bit) for bit in x])

    # Compute fitness values
    df['WalshFitness'] = df['Solution'].apply(lambda sol: f.eval(sol))
    df['BinFitness'] = df['IntegerBits'].apply(lambda sol: f.eval(sol))

    # Split the Solution column into multiple columns
    solution_columns = pd.DataFrame(df['Solution'].tolist(), columns=[f'Variable{i+1}' for i in range(length)])
    df = pd.concat([df, solution_columns], axis=1)

    # Select only the desired columns: WalshFitness and Variable1 to Variable14
    desired_columns = ['WalshFitness'] + [f'Variable{i+1}' for i in range(length)]
    df = df[desired_columns]

    return df


def plot_shap_summary(shap_values, X_test, model_name, fold, run_number, instance_number, degree):
    """
    Plot and save the SHAP summary plot for a given model and fold.

    Parameters:
    - shap_values: SHAP values obtained from the explainer.
    - X_test: Test data used for plotting.
    - model_name: str, name of the model (e.g., 'MLP', 'RF').
    - fold: int, fold number in cross-validation.
    - instance_number: str or int, instance identifier for saving plots.

    Returns:
    - None
    """
    # Get the script name from sys.argv[0]
    script_name = os.path.basename(sys.argv[0])
    script_name_no_ext = os.path.splitext(script_name)[0]

    # Based on the script name, adjust the directory structure
    if script_name_no_ext == 'model_training':
        # For model-training.py, use instance numbers deg2: (1000-1029), deg10: (1050-1079)
        base_dir = f'puboi/{degree}'
    elif script_name_no_ext == 'minemodelstest':
        # For minemodelstest, use run numbers deg2: (1000-1029), 10 runs per instance. Deg10: (1050-1079), 10 runs
        base_dir = f'ils/{degree}'
    elif script_name_no_ext == 'executed-separately-ils':
        # For executed-separately-ils, use run numbers deg2: (1000-1029), 10 runs per instance. Deg10: (1050-1079), 10 runs
        base_dir = f'ils-split/{degree}'
    else:
        # Default base directory
        base_dir = 'results'
        print(f"Warning: Unrecognised script name '{script_name_no_ext}'. Saving plots to 'results/' directory.")


    # Ensure instance_number is a string
    instance_number_str = str(instance_number)
    # Construct the directory path
    model_dir = os.path.join(base_dir, instance_number_str, model_name.replace(' ', '_').lower())
    os.makedirs(model_dir, exist_ok=True)
    if run_number is not None:
        plot_path = os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}_shap_fold_{fold}_run{run_number}.png')
    else:
        plot_path = os.path.join(model_dir, f'{model_name.lower().replace(" ", "_")}_shap_fold_{fold}.png')

    # Plot the SHAP summary
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"{(degree)} {model_name} SHAP Summary Plot Fold {fold}", fontsize=14, fontweight="bold")
    plt.xlabel("Feature Value", fontsize=12)
    plt.ylabel("SHAP Value", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=10)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


## Ground truth metrics 


# # Functions for computing ground truth metrics
# def compute_PRA_with_ground_truth(ground_truth, explainer_importances):
#     """
#     Compute Pairwise Rank Agreement (PRA) between ground truth and explainer importances.
#     """
#     # Ensure both Series have the same features
#     common_features = ground_truth.index.intersection(explainer_importances.index)
#     ground_truth = ground_truth.loc[common_features]
#     explainer_importances = explainer_importances.loc[common_features]
    
#     features = ground_truth.index
#     num_features = len(features)
    
#     # Generate all unique feature pairs
#     feature_pairs = list(combinations(features, 2))
    
#     if not feature_pairs:
#         # If only one feature, define PRA as 1
#         return 1.0
    
#     consistent_pairs = 0
#     total_pairs = len(feature_pairs)
    
#     for (feat_a, feat_b) in feature_pairs:
#         gt_a, gt_b = ground_truth[feat_a], ground_truth[feat_b]
#         expl_a, expl_b = explainer_importances[feat_a], explainer_importances[feat_b]
        
#         # Determine ranking in ground truth
#         if gt_a > gt_b:
#             gt_relation = 1
#         elif gt_a < gt_b:
#             gt_relation = -1
#         else:
#             gt_relation = 0  # Tie
        
#         # Determine ranking in explainer importances
#         if expl_a > expl_b:
#             expl_relation = 1
#         elif expl_a < expl_b:
#             expl_relation = -1
#         else:
#             expl_relation = 0  # Tie
        
#         # Check if relations are the same
#         if gt_relation == expl_relation:
#             consistent_pairs += 1
    
#     pra = consistent_pairs / total_pairs
#     return pra



def compute_PRA_with_ground_truth(ground_truth, explainer_importances):
    """
    Optimized version of Pairwise Rank Agreement (PRA) using vectorized operations.
    """
    # Ensure both Series have the same features
    common_features = ground_truth.index.intersection(explainer_importances.index)
    ground_truth = ground_truth.loc[common_features]
    explainer_importances = explainer_importances.loc[common_features]
    
    # Compute rankings
    ground_truth_ranks = rankdata(-ground_truth)  # Negative for descending ranking
    explainer_ranks = rankdata(-explainer_importances)
    
    # Calculate the agreement
    agreements = (np.sign(ground_truth_ranks[:, None] - ground_truth_ranks) == 
                np.sign(explainer_ranks[:, None] - explainer_ranks)).astype(int)
    
    # Calculate PRA
    pra = np.sum(agreements) / (len(ground_truth) * (len(ground_truth) - 1))
    return pra

def compute_spearman_correlation(ground_truth, explainer_importances):
    """
    Compute Spearman's Rank Correlation between ground truth and explainer importances.
    """
    # Align features
    common_features = ground_truth.index.intersection(explainer_importances.index)
    print("Common features:\n ",common_features)
    ground_truth = ground_truth.loc[common_features]
    print("Ground truth \n",ground_truth)
    explainer_importances = explainer_importances.loc[common_features]
    print("Explainer importances",explainer_importances)
    
    spearman_corr, p_value = spearmanr(ground_truth, explainer_importances)
    print(f'Spearman and p-value: {spearman_corr}, {p_value}')
    return spearman_corr, p_value

def compute_kendall_tau(ground_truth, explainer_importances):
    """
    Compute Kendall's Tau between ground truth and explainer importances.
    """
    # Align features
    common_features = ground_truth.index.intersection(explainer_importances.index)
    ground_truth = ground_truth.loc[common_features]
    explainer_importances = explainer_importances.loc[common_features]
    
    kendall_tau, p_value = kendalltau(ground_truth, explainer_importances)
    return kendall_tau, p_value

# This is designed to evaluate the precision of a model's feature importance rankings against a ground truth ranking.
# It does so by computing Precision at K, which measure how many of the top K important features that are identified by the explainer match the top k features in the ground truth.
def compute_precision_at_k(ground_truth, explainer_importances, k=5):
    """
    Compute Precision at K between ground truth and explainer importances.
    """
    # Align features 
    # This part ensures only features common to both the ground truth and the explainer's importance rankings are considered
    # It filters the two inputs so they both operate on the same set of features
    common_features = ground_truth.index.intersection(explainer_importances.index)
    ground_truth = ground_truth.loc[common_features]
    explainer_importances = explainer_importances.loc[common_features]
    
    # nlargest(k) is used to select the top K features based on their importance scores from both the ground truth and explainer importances. The indices (feature names) of these top K features are stored in sets for comparison
    top_k_ground_truth = set(ground_truth.nlargest(k).index)
    top_k_explainer = set(explainer_importances.nlargest(k).index)
    
    # The true positives and precision are calculated here. The intersection between the sets of top-K features from the ground truth and the explainer is computed, giving the number of true positives (i.e., the number of features that appear in both top-K lists).
    # Precision at K is then calculated as the ratio of true positives to K, and this value is returned.
    true_positives = len(top_k_ground_truth.intersection(top_k_explainer))
    precision_at_k = true_positives / k
    return precision_at_k

# compute ratio - either shap lime or pfi values on each of the three folds compute these, then pass this set to the function and it should return a ratio - dont need to pass ground truth
#meadian of 1234 on top and median of the remaining on the bottom and do a ratio 
# KNN not in use, but is kept here in case of future work

def compute_importance_ratio(importances, abs = False):
    """
    Compute the ratio of median importance of ground truth variables to median importance of remaining variables.
    Assumes the first four variables are the ground truth.

    Parameters:
    - importances: pandas Series, feature importances with variable names as index.

    Returns:
    - ratio: float, median_top4 / median_remaining. Returns np.nan if median_remaining is zero.
    """
    epsilon=1e-4
    if abs:
        importances = np.abs(importances)
    # Define ground truth variables
    ground_truth_vars = ['Variable1', 'Variable2', 'Variable3', 'Variable4']
    
    # Ensure all ground truth variables are present
    missing_vars = [var for var in ground_truth_vars if var not in importances.index]
    if missing_vars:
        raise ValueError(f"The following ground truth variables are missing from importances: {missing_vars}")
    
    # Extract importances for ground truth variables
    top_importances = importances[ground_truth_vars]
    print("\ntop imp: \n", top_importances)
    median_top = top_importances.median()
    print("\nmedian top imp: \n", median_top)

    
    # Extract importances for remaining variables
    remaining_vars = [var for var in importances.index if var not in ground_truth_vars]
    
    if not remaining_vars:
        # If there are no remaining variables, define behavior (e.g., return np.nan or a default value)
        return np.nan
    
    remaining_importances = importances[remaining_vars]
    print("\nRemaining: \n", remaining_importances)
    median_remaining = remaining_importances.median()
    print("\nMedian remaining: \n", median_remaining)
    
    # Apply epsilon to prevent division by zero or very small denominator
    if median_remaining < epsilon:
        logger.warning(f"Median of remaining importances ({median_remaining}) is below epsilon ({epsilon}). Using epsilon as denominator.")
        median_remaining_adjusted = epsilon
    else:
        median_remaining_adjusted = median_remaining
    # Compute ratio
    ratio = median_top / median_remaining_adjusted
    print(f"\nRatio: {ratio}")
    
    # # Optionally cap the ratio to a maximum value
    # if max_ratio is not None and ratio > max_ratio:
    #     logger.warning(f"Computed ratio ({ratio}) exceeds max_ratio ({max_ratio}). Capping the ratio.")
    #     ratio = max_ratio
    
    return ratio
def knn_model_with_gridsearch(df):
    """
    Perform hyperparameter tuning for KNN and return the best model.

    Parameters:
    - df: pandas DataFrame containing features and target variable.

    Returns:
    - best_estimator_: the best KNN model found by GridSearchCV.
    """
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']

    model = KNeighborsRegressor(algorithm='kd_tree')
    param_grid = {'n_neighbors': range(1, 21)}
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    start_time = time.time()
    print('KNN Grid search commencing...')
    grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='r2', return_train_score=True, n_jobs=-1)
    grid_search.fit(X, y)
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the KNN params: {elapsed_time:.3f} seconds")
    print("Best parameters:", grid_search.best_params_)
    print("Best R2 score:", grid_search.best_score_)

    return grid_search.best_estimator_


def knn_model(df, model, instance_number, models_aggregated_shap_values):
    """
    Train KNN models using K-fold cross-validation and compute SHAP values.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - model: KNeighborsRegressor model with best hyperparameters.
    - instance_number: str, instance identifier for saving plots.

    Returns:
    - None
    """
    all_models = []
    cv_scores = []
    X_encoded = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']
    shap_values_list = []

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    print("Computing KNN models...")

    for train_idx, test_idx in kfold.split(X_encoded, y):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        fold += 1
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        all_models.append(model)

        # Use KernelExplainer
        explainer = get_shap_explainer(model, X_train)
        shap_values = explainer.get_explanations(X_test)
        shap_values_list.append(shap_values)

        # Plot SHAP summary using the plotting function
        plot_shap_summary(shap_values, X_test, 'KNN', fold, instance_number)

    print("Average Cross-Validation R2 KNN:", sum(cv_scores) / len(cv_scores))
    models_aggregated_shap_values['KNN'] = aggregate_explanations(shap_values_list)


## Start of model training
def nn_model_with_gridsearch(df, instance_number):
    """
    Perform hyperparameter tuning for MLPRegressor and return the best model along with scalers.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str or int, instance identifier for saving hyperparameters.

    Returns:
    - model: the best MLPRegressor model found by GridSearchCV.
    - scaler: StandardScaler fitted on training data.
    - y_scaler: StandardScaler fitted on target variable.
    """
    from sklearn.preprocessing import StandardScaler

    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']

    # Initialize scalers
    scaler = StandardScaler()
    y_scaler = StandardScaler()

    # Fit scalers
    X_scaled = scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).ravel()  # Flatten back to 1D

    model = MLPRegressor(random_state=42, max_iter=5000)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['tanh', 'relu'], # remove relu, make this fixed on tanh
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define hyperparameter file path
    hyperparam_path = os.path.join(HYPERPARAMS_DIR, str(instance_number), f'mlp_{instance_number}_hyperparams.json')
    os.makedirs(os.path.dirname(hyperparam_path), exist_ok=True)


    os.makedirs(os.path.dirname(hyperparam_path), exist_ok=True)

    if os.path.exists(hyperparam_path):
        # Load hyperparameters
        with open(hyperparam_path, 'r') as f:
            best_params = json.load(f)
        print(f"Loading best MLP hyperparameters from {hyperparam_path}")
        model = MLPRegressor(**best_params, random_state=42)
    else:
        # Perform GridSearchCV
        start_time = time.time()
        print('\nNN Grid search commencing...')
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=kfold,
            scoring='r2',
            return_train_score=True,
            n_jobs=-1,
            error_score='raise'  # Raise exception if error occurs
        )
        grid_search.fit(X_scaled, y_scaled)
        elapsed_time = time.time() - start_time
        print(f"\tElapsed time to compute the NN params: {elapsed_time:.3f} seconds")
        print("\tBest parameters:", grid_search.best_params_)
        print("\tBest R2 score:", grid_search.best_score_)
        
        # Save best hyperparameters
        best_params = grid_search.best_params_
        with open(hyperparam_path, 'w') as f:
            json.dump(best_params, f)
        model = grid_search.best_estimator_

    return model, scaler, y_scaler  # Return scalers
def nn_regression(df, instance_number, degree, models_aggregated_shap_values, 
                 models_aggregated_pfi_values, models_aggregated_lime_values, 
                 ground_truth_importances=None, metrics_aggregated=None, 
                 r2_scores=None, run_number=None):
    """
    Train MLPRegressor models using K-fold cross-validation and compute SHAP, PFI, and LIME values.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str or int, instance identifier for saving plots.
    - models_aggregated_shap_values, models_aggregated_pfi_values, models_aggregated_lime_values: dicts to store aggregated explanations.
    - ground_truth_importances: pandas Series containing ground truth feature importances.

    Returns:
    - None
    """
    cv_scores = []
    shap_values_list = []
    pfi_values_list = []
    lime_values_list = []
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']
    n_splits = 3 
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    logger.info("\nComputing MLP Regressor models...")

    for train_idx, test_idx in kfold.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold += 1
        model = MLPRegressor(random_state=42, activation='tanh', max_iter=5000, hidden_layer_sizes=(50,))
        
        # Train the model once per fold
        model.fit(X_train, y_train) 
        
        # Predict and evaluate
        prediction = model.predict(X_test)
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        logger.info(f"\nMLP R2 score for fold {fold}: {r2:.7f}")
        print(f"\nMLP R2 score for fold {fold}: {r2:.7f}")

        # Initialize SHAP explainer
        explainer = get_shap_explainer(model, X_train)
        print(f"Fold {fold} - SHAP Explainer initialised with feature names: {X_test.columns}")
        logger.info(f"Fold {fold} - SHAP Explainer initialised with feature names: {X_test.columns}")
        shap_values = explainer.get_explanations(X_test)
        shap_values_list.append(shap_values)

        # Initialize LIME explainer
        lime_explainer = init_lime(X_train, model)
        print(f"Fold {fold} - LIME Explainer initialised.")
        logger.info(f"Fold {fold} - LIME Explainer initialised.")

        # Generate LIME explanations using the wrapper's method
        lime_importances = lime_explainer.get_explanations(X_test)
        # Aggregate LIME importances (e.g., mean over samples)
        #lime_importances_mean = lime_importances.mean(axis=0)
        lime_values_list.append(lime_importances)  # Append as NumPy array

        # Debugging: Check LIME importances
        print(f"Fold {fold} - LIME importances computed: {lime_importances}")
        logger.info(f"Fold {fold} - LIME importances computed")

        # Permutation Feature Importance (PFI)
        print(f"Fold {fold} - Computing permutation importance...")
        logger.info(f"Fold {fold} - Computing permutation importance...")
        result = permutation_importance(
            model,
            X_test,  # Ensure X_test is a DataFrame
            y_test,
            scoring='r2',
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        pfi_importances = result.importances
        pfi_values_list.append(pfi_importances)
        
        # Debugging: Check PFI importances
        print(f"Fold {fold} - PFI importances computed: {pfi_importances}")
        logger.info(f"Fold {fold} - PFI importances computed.")

        # Plot SHAP summary
        plot_shap_summary(shap_values, X_test, 'Multi Layer Perceptron', fold, run_number, instance_number, degree)
        logger.info(f"SHAP summary for {fold}, instance number {instance_number}")

        
        shap_vals = np.abs(shap_values)
        median_shap = np.median(shap_vals, axis=0)

        #lime_importances_array = np.array([np.abs(fold_lime_importances) for fold_lime_importances in lime_importances])  # Shape: (n_folds, n_features)
        abs_lime_importances = np.abs(lime_importances)
        median_lime = np.median(abs_lime_importances, axis=0)

        #pfi_importances_array = np.array([np.abs(fold_pfi_importances) for fold_pfi_importances in pfi_importances])  # Shape: (n_folds, n_features)
        #median_abs_pfi_importances = np.abs(pfi_importances)
       # median_pfi = np.median(pfi_importances, axis=0)
        #abs_pfi = np.median(pfi_importances)
        median_pfi_vars = np.median(pfi_importances, axis=1)
        if ground_truth_importances is not None:
            # Aggregate SHAP values to get explainer importances
            shap_importances = pd.Series(
                data=median_shap,
                index=X_test.columns                
            )
            pfi_imp = pd.Series(
                data=median_pfi_vars,
                index=X_test.columns
            )
            lime_importances_series = pd.Series(
                data=median_lime,
                index=X_test.columns
            )

            #             # Convert SHAP and LIME values to DataFrame for easier manipulation
            # shap_df = pd.DataFrame(shap_values.values, columns=X_test.columns)
            # lime_df = pd.DataFrame(lime_importances, columns=X_test.columns)  # Assuming lime_importances is array-like
            # pfi_df = pd.DataFrame(pfi_importances, columns=X_test.columns)

            # Debugging: Print importances
            print("Ground Truth Importances:")
            print(ground_truth_importances.sort_values(ascending=False))

            print("\nSHAP Importances:")
            print(shap_importances.sort_values(ascending=False))

            print("\nPFI Importances:")
            print(pfi_imp.sort_values(ascending=False))

            print("\nLIME Importances:")
            print(lime_importances_series.sort_values(ascending=False))

            print("\nSHAP Importance Ratio:")
            shap_ratio = compute_importance_ratio(shap_importances)
            print("\nLime Importance Ratio:")
            lime_ratio = compute_importance_ratio(lime_importances_series)
            print("\nPFI Importance Ratio:")
            pfi_ratio = compute_importance_ratio(pfi_imp, abs=True)

                  # we have 3 function calls for XAI eval metrics, instead of these we now have one calling the ratio function for shap lime and PFI 
            append_metric(metrics_aggregated, 'MLP', 'SHAP', 'ImportanceRatio', shap_ratio, fold=fold)
            append_metric(metrics_aggregated, 'MLP', 'LIME', 'ImportanceRatio', lime_ratio, fold=fold)
            append_metric(metrics_aggregated, 'MLP', 'PFI', 'ImportanceRatio', pfi_ratio, fold=fold)
          

    print("Median Cross-Validation R2 NN:", np.median(cv_scores))
    logger.info(f"Median Cross-Validation R2 Multi Layer Perceptron: {np.median(cv_scores)}")
    r2_scores['MLP'].append(np.median(cv_scores))
    #models_aggregated_shap_values['MLP'] = aggregate_explanations(shap_values_list)
    # After aggregation
    shap_importances = aggregate_explanations(shap_values_list)
    # Create a mapping from 0-13 to 'Variable1' to 'Variable14'
    variable_names = {i: f'Variable{i+1}' for i in range(14)}  # 0-13 mapped to 'Variable1'-'Variable14'
    shap_series = pd.Series(shap_importances, index=variable_names)
    models_aggregated_shap_values['MLP'] = shap_series
    models_aggregated_pfi_values['MLP'] = aggregate_pfi_values(pfi_values_list, X_test.columns)
    models_aggregated_lime_values['MLP'] = aggregate_lime_values(lime_values_list, X_test.columns)


def lr_model_with_gridsearch(df, instance_number):
    """
    Perform hyperparameter tuning for LinearRegression and return the best model.
    Saves the best hyperparameters to a JSON file.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str or int, identifier for the current instance.

    Returns:
    - best_estimator_: the best LinearRegression model found by GridSearchCV or loaded from file.
    """
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']

    model = LinearRegression()
    param_grid = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1, 1],
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define the hyperparameter file path
    hyperparams_file = os.path.join(HYPERPARAMS_DIR, str(instance_number), f'lr_{instance_number}_hyperparams.json')
    os.makedirs(os.path.dirname(hyperparams_file), exist_ok=True)


    if os.path.exists(hyperparams_file):
        # Load hyperparameters from file
        with open(hyperparams_file, 'r') as f:
            best_params = json.load(f)
        print(f'Loading pre-saved LinearRegression hyperparameters for instance {instance_number}: {best_params}')
        # Set the model's parameters
        model.set_params(**best_params)
    else:
        # Perform grid search
        start_time = time.time()
        print('\nLR Grid search commencing...')
        grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='r2', return_train_score=True, n_jobs=-1)
        grid_search.fit(X, y)
        elapsed_time = time.time() - start_time
        print(f"\tElapsed time to compute the LR params: {elapsed_time:.3f} seconds")
        print("\tBest parameters:", grid_search.best_params_)
        print("\tBest R2 score:", grid_search.best_score_)

        # Save the best hyperparameters to file
        best_params = grid_search.best_params_
        with open(hyperparams_file, 'w') as f:
            json.dump(best_params, f)
        print(f'\tSaved best LinearRegression hyperparameters to {hyperparams_file}')

        # Set the model to best estimator
        model = grid_search.best_estimator_

    return model
# model_training.py

def linear_regression_with_shap(df, instance_number, degree, models_aggregated_shap_values, 
                                models_aggregated_pfi_values, models_aggregated_lime_values, 
                                ground_truth_importances=None, metrics_aggregated=None, 
                                r2_scores=None, run_number=None):
    """
    Train LinearRegression models using K-fold cross-validation and compute SHAP, PFI, and LIME values.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str, instance identifier for saving plots.
    - models_aggregated_shap_values, models_aggregated_pfi_values, models_aggregated_lime_values: dicts to store aggregated explanations.
    - ground_truth_importances: pandas Series containing ground truth feature importances.

    Returns:
    - None
    """
    cv_scores = []
    shap_values_list = []
    pfi_values_list = []
    lime_values_list = []
    X_encoded = df.drop(columns='WalshFitness')  # Ensure X has column names
    y = df['WalshFitness']
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 0
    print("\nComputing Linear Regression Models with SHAP...")
    logger.info("\nComputing Linear Regression Models with SHAP...")
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_encoded, y), 1):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression()
        
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        
        # Debugging: Check the type and columns of X_test
        print(f"Fold {fold} - X_test type: {type(X_test)}, columns: {X_test.columns.tolist()}")
        
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        print(f"\nLR R2 score for fold {fold}: {r2:.7f}")
        logger.info(f"\nLR R2 score for fold {fold}: {r2:.7f}")
        
        # Initialize SHAP explainer
        explainer = get_shap_explainer(model, X_train)
        print(f"Fold {fold} - SHAP Explainer initialised with feature names: {X_test.columns.tolist()}")
        logger.info(f"Fold {fold} - SHAP Explainer initialised with feature names: {X_test.columns.tolist()}")
        shap_values = explainer.get_explanations(X_test)
        shap_values_list.append(shap_values)
        
        # Initialize LIME explainer
        lime_explainer = init_lime(X_train, model)
        print(f"Fold {fold} - LIME Explainer initialised.")
        logger.info(f"Fold {fold} - LIME Explainer initialised.")
        
        # Generate LIME explanations using the wrapper's method
        lime_importances = lime_explainer.get_explanations(X_test)
        # Aggregate LIME importances (e.g., mean over samples)
        lime_importances_mean = lime_importances.mean(axis=0)
        lime_values_list.append(lime_importances)  # Append as NumPy array
        
        # Debugging: Check LIME importances
        print(f"Fold {fold} - LIME importances computed: {lime_importances}")
        logger.info(f"Fold {fold} - LIME importances computed")
        
        # Permutation Feature Importance (PFI)
        print(f"Fold {fold} - Computing permutation importance...")
        logger.info(f"Fold {fold} - Computing permutation importance...")
        result = permutation_importance(
            model,
            X_test,  # Ensure X_test is a DataFrame
            y_test,
            scoring='r2',
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        pfi_importances = result.importances
        pfi_values_list.append(pfi_importances)
        
        # Debugging: Check PFI importances
        print(f"Fold {fold} - PFI importances computed: {pfi_importances}")
        logger.info(f"Fold {fold} - PFI importances computed")

        # Plot SHAP summary
        plot_shap_summary(shap_values, X_test, 'Linear Regression', fold, run_number, instance_number, degree)
        logger.info(f"SHAP summary for {fold}, instance number {instance_number}")
        # If ground truth importances are provided, compute metrics
        #shap_values_array = np.array([np.abs(fold_shap_values) for fold_shap_values in shap_values])  # Shape: (n_folds, n_samples, n_features)
        shap_vals = np.abs(shap_values)
        median_shap = np.median(shap_vals, axis=0)

        #lime_importances_array = np.array([np.abs(fold_lime_importances) for fold_lime_importances in lime_importances])  # Shape: (n_folds, n_features)
        abs_lime_importances = np.abs(lime_importances)
        median_lime = np.median(abs_lime_importances, axis=0)

        #pfi_importances_array = np.array([np.abs(fold_pfi_importances) for fold_pfi_importances in pfi_importances])  # Shape: (n_folds, n_features)
        #median_abs_pfi_importances = np.abs(pfi_importances)
       # median_pfi = np.median(pfi_importances, axis=0)
        median_pfi_vars = np.median(pfi_importances, axis=1)
        if ground_truth_importances is not None:
            # Aggregate SHAP values to get explainer importances
            shap_importances = pd.Series(
                data=median_shap,
                index=X_test.columns                
            )
            pfi_imp = pd.Series(
                data=median_pfi_vars,
                index=X_test.columns
            )
            lime_importances_series = pd.Series(
                data=median_lime,
                index=X_test.columns
            )

            # Debugging: Print importances
            print("Ground Truth Importances:")
            print(ground_truth_importances.sort_values(ascending=False))

            print("\nSHAP Importances:")
            print(shap_importances.sort_values(ascending=False))

            print("\nPFI Importances:")
            print(pfi_imp.sort_values(ascending=False))

            print("\nLIME Importances:")
            print(lime_importances_series.sort_values(ascending=False))
          
            # Compute and store SHAP metrics
            logger.info("SHAP ground truth metrics computing...\n")
            spearman_corr, p = compute_spearman_correlation(ground_truth_importances, shap_importances)
            spearman = {'value':f'{spearman_corr:.5f}', 'error':f'{p:.5f}'}
            metrics_aggregated['SHAP']['Spearman'].append(spearman)
            logger.info(spearman)
            
            kendall_tau, p = compute_kendall_tau(ground_truth_importances, shap_importances)
            kendall = {'value':f'{kendall_tau:.5f}', 'error':f'{p:.5f}'}
            metrics_aggregated['SHAP']['KendallTau'].append(kendall)
            logger.info(kendall)
            
            precision_k = compute_precision_at_k(ground_truth_importances, shap_importances, k=5)
            metrics_aggregated['SHAP']['PrecisionAtK'].append(precision_k)
            logger.info(precision_k)
            

            # Compute and store PFI metrics
            logger.info("PFI ground truth metrics computing...\n")
            spearman_corr_pfi, p_pfi = compute_spearman_correlation(ground_truth_importances, pfi_imp)
            spearman = {'value':f'{spearman_corr_pfi:.5f}', 'error':f'{p_pfi:.5f}'}
            metrics_aggregated['PFI']['Spearman'].append(spearman)
            logger.info(spearman)
            
            kendall_tau_pfi, p_pfi = compute_kendall_tau(ground_truth_importances, pfi_imp)
            kendall = {'value':f'{kendall_tau_pfi:.5f}', 'error':f'{p_pfi:.5f}'}
            metrics_aggregated['PFI']['KendallTau'].append(kendall)
            logger.info(kendall)
            
            precision_k_pfi = compute_precision_at_k(ground_truth_importances, pfi_imp, k=5)
            metrics_aggregated['PFI']['PrecisionAtK'].append(precision_k_pfi)
            logger.info(precision_k_pfi)

            
            logger.info("LIME ground truth metrics computing...\n")
            # Compute and store LIME metrics
            spearman_corr_lime, p = compute_spearman_correlation(ground_truth_importances, lime_importances_series)
            spearman_corr_lime = np.abs(spearman_corr_lime)
            spearman = {'value':f'{spearman_corr_lime:.5f}', 'error':f'{p:.5f}'}
            logger.info(spearman)
            metrics_aggregated['LIME']['Spearman'].append(spearman)
            
            kendall_tau_lime, p = compute_kendall_tau(ground_truth_importances, lime_importances_series)
            kendall_tau_lime = np.abs(kendall_tau_lime)
            kendall = {'value':f'{kendall_tau_lime:.5f}', 'error':f'{p:.5f}'}
            metrics_aggregated['LIME']['KendallTau'].append(kendall)
            logger.info(kendall)
            
            precision_k_lime = compute_precision_at_k(ground_truth_importances, lime_importances_series, k=5)
            metrics_aggregated['LIME']['PrecisionAtK'].append(precision_k_lime)
            logger.info(precision_k_lime)

            
    print("Average Cross-Validation R2 LR:", sum(cv_scores) / len(cv_scores))
    logger.info("Average Cross-Validation R2 LR:", sum(cv_scores) / len(cv_scores))
    r2_scores['LR'].append(sum(cv_scores)/len(cv_scores))
    #models_aggregated_shap_values['Linear Regression'] = aggregate_explanations(shap_values_list)
    # After aggregation
    shap_importances = aggregate_explanations(shap_values_list)
    # Create a mapping from 0-13 to 'Variable1' to 'Variable14'
    variable_names = {i: f'Variable{i+1}' for i in range(14)}  # 0-13 mapped to 'Variable1'-'Variable14'
    #variable_names = [f'{i+1}' for i in range(len(shap_importances))]
    shap_series = pd.Series(shap_importances, index=variable_names)
    models_aggregated_shap_values['LR'] = shap_series
    models_aggregated_pfi_values['LR'] = aggregate_pfi_values(pfi_values_list, X_test.columns)
    models_aggregated_lime_values['LR'] = aggregate_lime_values(lime_values_list, X_test.columns)


def rf_model_with_gridsearch(df, instance_number):
    """
    Perform hyperparameter tuning for RandomForestRegressor and return the best model.
    Saves the best hyperparameters to a JSON file.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str or int, identifier for the current instance.

    Returns:
    - best_estimator_: the best RandomForestRegressor model found by GridSearchCV or loaded from file.
    """
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']

    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'max_features': [None, 'sqrt', 'log2'],
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define the hyperparameter file path
    hyperparams_file = os.path.join(HYPERPARAMS_DIR, str(instance_number), f'rf_{instance_number}_hyperparams.json')
    os.makedirs(os.path.dirname(hyperparams_file), exist_ok=True)


    if os.path.exists(hyperparams_file):
        # Load hyperparameters from file
        with open(hyperparams_file, 'r') as f:
            best_params = json.load(f)
        print(f'Loading pre-saved RandomForestRegressor hyperparameters for instance {instance_number}: {best_params}')
        # Set the model's parameters
        model.set_params(**best_params)
    else:
        # Perform grid search
        start_time = time.time()
        print('\nRF Grid search commencing...')
        grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='r2', return_train_score=True, n_jobs=-1)
        grid_search.fit(X, y)
        elapsed_time = time.time() - start_time
        print(f"\tElapsed time to compute the RF params: {elapsed_time:.3f} seconds")
        print("\tBest parameters:", grid_search.best_params_)
        print("\tBest R2 score:", grid_search.best_score_)

        # Save the best hyperparameters to file
        best_params = grid_search.best_params_
        with open(hyperparams_file, 'w') as f:
            json.dump(best_params, f)
        print(f'\tSaved best RandomForestRegressor hyperparameters to {hyperparams_file}')

        # Set the model to best estimator
        model = grid_search.best_estimator_

    return model
def append_metric(metrics_aggregated, model, explainer, metric_name, value, fold=None, sample_id=None):
    """
    Append a metric to the aggregated data structure with optional fold and sample identification.

    Parameters:
    - metrics_aggregated: dict, aggregated metrics.
    - model: str, ML model name ('RF', 'MLP').
    - explainer: str, XAI technique ('SHAP', 'LIME', 'PFI').
    - metric_name: str, metric name ('ImportanceRatio', etc.).
    - value: float, metric value.
    - fold: int or None, fold number.
    - sample_id: int or None, sample identifier within the fold.

    Returns:
    - None
    """
    metric_entry = {
        'value': value,
        'Fold': fold,
        'SampleID': sample_id
    }
    # Optionally, include 'error' if applicable
    # if error is not None:
    #     metric_entry['Error'] = error
    metrics_aggregated[model][explainer][metric_name].append(metric_entry)

def rf_regression_model(df, instance_number, degree, models_aggregated_shap_values, 
                        models_aggregated_pfi_values, models_aggregated_lime_values, 
                        ground_truth_importances=None, metrics_aggregated=None, 
                        r2_scores=None, run_number=None):
    """
    Train RandomForestRegressor models using K-fold cross-validation and compute SHAP, PFI, and LIME values.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str, instance identifier for saving plots.
    - models_aggregated_shap_values, models_aggregated_pfi_values, models_aggregated_lime_values: dicts to store aggregated explanations.
    - ground_truth_importances: pandas Series containing ground truth feature importances.

    Returns:
    - None
    """
    cv_scores = []
    shap_values_list = []
    pfi_values_list = []
    lime_values_list = []
    X_encoded = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print("\nComputing Random Forest models...")
    logger.info("\nComputing Random Forest models...")

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X_encoded, y), 1):
        X_train, X_test = X_encoded.iloc[train_idx], X_encoded.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestRegressor(random_state=42, n_estimators=50)
        
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        logger.info(f"\nRF R2 score for fold {fold}: {r2:.7f}")
        print(f"\nRF R2 score for fold {fold}: {r2:.7f}")
        
        # Initialize SHAP explainer
        explainer = get_shap_explainer(model, X_train)
        shap_values = explainer.get_explanations(X_test)
        shap_values_list.append(shap_values)
        with open('shapimportance-fullspace.txt', 'w') as filehandle:
            for line in shap_values_list:
                filehandle.write(" ".join(map(str, line)) + "\n")
        
        # Initialize LIME explainer
        lime_explainer = init_lime(X_train, model)
        logger.info(f"Fold {fold} - LIME Explainer initialized.")
        print(f"Fold {fold} - LIME Explainer initialized.")
        
        # **Use the wrapper's get_explanations method instead of directly calling explain_instance**
        lime_importances = lime_explainer.get_explanations(X_test)
        # Aggregate LIME importances (e.g., mean over samples)
        
        lime_values_list.append(lime_importances)  # Append as NumPy array
        with open('limeimportance-fullspace.txt', 'w') as filehandle:
            for line in lime_values_list:
                filehandle.write(" ".join(map(str, line)) + "\n")

        
        # Debugging: Check LIME importances
        logger.info(f"Fold {fold} - LIME importances computed: {lime_importances}")
        print(f"Fold {fold} - LIME importances computed: {lime_importances}")
        
        # Permutation Feature Importance (PFI)
        logger.info(f"Fold {fold} - Computing permutation importance...")
        print(f"Fold {fold} - Computing permutation importance...")
        result = permutation_importance(
            model,
            X_test,  # Ensure X_test is a DataFrame
            y_test,
            scoring='r2',
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        pfi_importances = result.importances
        pfi_values_list.append(pfi_importances)
        with open('pfiimportance-fullspace.txt', 'w') as filehandle:
            for line in pfi_values_list:
                filehandle.write(" ".join(map(str, line)) + "\n")

        
        # Debugging: Check PFI importances
        print(f"Fold {fold} - PFI importances computed: {pfi_importances}")
        logger.info(f"Fold {fold} - PFI importances computed.")

        # Plot SHAP summary
        plot_shap_summary(shap_values, X_test, 'Random Forest', fold, run_number, instance_number, degree)
        logger.info(f"SHAP summary for {fold}, instance number {instance_number}")

        shap_vals = np.abs(shap_values)
        median_shap = np.median(shap_vals, axis=0)

        #lime_importances_array = np.array([np.abs(fold_lime_importances) for fold_lime_importances in lime_importances])  # Shape: (n_folds, n_features)
        abs_lime_importances = np.abs(lime_importances)
        median_lime = np.median(abs_lime_importances, axis=0)

        #pfi_importances_array = np.array([np.abs(fold_pfi_importances) for fold_pfi_importances in pfi_importances])  # Shape: (n_folds, n_features)
        #median_abs_pfi_importances = np.abs(pfi_importances)
       # median_pfi = np.median(pfi_importances, axis=0)
       # abs_pfi = np.median(pfi_importances)
        median_pfi_vars = np.median(pfi_importances, axis=1)
        if ground_truth_importances is not None:
            # Aggregate SHAP values to get explainer importances
            shap_importances = pd.Series(
                data=median_shap,
                index=X_test.columns                
            )
            pfi_imp = pd.Series(
                data=median_pfi_vars,
                index=X_test.columns
            )
            lime_importances_series = pd.Series(
                data=median_lime,
                index=X_test.columns
            )

            # Debugging: Print importances
            print("Ground Truth Importances:")
            print(ground_truth_importances.sort_values(ascending=False))

            print("\nSHAP Importances:")
            print(shap_importances.sort_values(ascending=False))

            print("\nPFI Importances:")
            print(pfi_imp.sort_values(ascending=False))

            print("\nLIME Importances:")
            print(lime_importances_series.sort_values(ascending=False))

            print("\nSHAP Importance Ratio:")
            shap_ratio = compute_importance_ratio(shap_importances)
            print("\nLime Importance Ratio:")
            lime_ratio = compute_importance_ratio(lime_importances_series)
            print("\nPFI Importance Ratio:")
            pfi_ratio = compute_importance_ratio(pfi_imp, abs=True)

          # we have 3 function calls for XAI eval metrics, instead of these we now have one calling the ratio function for shap lime and PFI 
            append_metric(metrics_aggregated, 'RF', 'SHAP', 'ImportanceRatio', shap_ratio, fold=fold)
            append_metric(metrics_aggregated, 'RF', 'LIME', 'ImportanceRatio', lime_ratio, fold=fold)
            append_metric(metrics_aggregated, 'RF', 'PFI', 'ImportanceRatio', pfi_ratio, fold=fold)



        #    # Compute and store SHAP metrics
        #     logger.info("SHAP ground truth metrics computing...\n")
        #     spearman_corr, p = compute_spearman_correlation(ground_truth_importances, shap_importances)
        #     spearman = {'value':f'{spearman_corr:.8f}', 'error':f'{p:.8f}'}
        #     metrics_aggregated['RF']['SHAP']['Spearman'].append(spearman)
        #     logger.info(spearman)
            
        #     kendall_tau, p = compute_kendall_tau(ground_truth_importances, shap_importances)
        #     kendall = {'value':f'{kendall_tau:.8f}', 'error':f'{p:.8f}'}
        #     metrics_aggregated['RF']['SHAP']['KendallTau'].append(kendall)
        #     logger.info(kendall)
            
        #     precision_k = compute_precision_at_k(ground_truth_importances, shap_importances, k=5)
        #     metrics_aggregated['RF']['SHAP']['PrecisionAtK'].append(precision_k)
        #     logger.info(precision_k)
            
        #     # Compute and store PFI metrics
        #     logger.info("PFI ground truth metrics computing...")
        #     spearman_corr_pfi, p_pfi = compute_spearman_correlation(ground_truth_importances, pfi_imp)
        #     spearman = {'value':f'{spearman_corr_pfi:.8f}', 'error':f'{p_pfi:.8f}'}
        #     metrics_aggregated['RF']['PFI']['Spearman'].append(spearman)
        #     logger.info(spearman)
            
        #     kendall_tau_pfi, p_pfi = compute_kendall_tau(ground_truth_importances, pfi_imp)
        #     kendall_pfi = {'value':f'{kendall_tau_pfi:.8f}', 'error':f'{p_pfi:.8f}'}
        #     metrics_aggregated['RF']['PFI']['KendallTau'].append(kendall_pfi)
        #     logger.info(kendall_pfi)
            
        #     precision_k_pfi = compute_precision_at_k(ground_truth_importances, pfi_imp, k=5)
        #     metrics_aggregated['RF']['PFI']['PrecisionAtK'].append(precision_k_pfi)
        #     logger.info(precision_k_pfi)
            
        #     # Compute and store LIME metrics
        #     logger.info("LIME ground truth metrics computing...")
        #     spearman_corr_lime, p_lime = compute_spearman_correlation(ground_truth_importances, lime_importances_series)
        #     spearman_corr_lime = np.abs(spearman_corr_lime)
        #     spearman_lime = {'value':f'{spearman_corr_lime:.8f}', 'error':f'{p_lime:.8f}'}
        #     metrics_aggregated['RF']['LIME']['Spearman'].append(spearman_lime)
        #     logger.info(spearman)
            
        #     kendall_tau_lime, p_k = compute_kendall_tau(ground_truth_importances, lime_importances_series)
        #     kendall_tau_lime = np.abs(kendall_tau_lime)
        #     kendall_lime = {'value':f'{kendall_tau_lime:.8f}', 'error':f'{p_k:.8f}'}
        #     metrics_aggregated['RF']['LIME']['KendallTau'].append(kendall_lime)
        #     logger.info(kendall_lime)
            
        #     precision_k_lime = compute_precision_at_k(ground_truth_importances, lime_importances_series, k=5)
        #     metrics_aggregated['RF']['LIME']['PrecisionAtK'].append(precision_k_lime)
        #     logger.info(precision_k_lime)
            
    print("Median Cross-Validation R2 RF:",np.median((cv_scores)))
    logger.info(f'Average Cross-Validation R2 Random Forest: {sum(cv_scores)/len(cv_scores)}')
    r2_scores['RF'].append(np.median(cv_scores))
    #shap_im
    # After aggregation
    shap_importances = aggregate_explanations(shap_values_list)
    # Create a mapping from 0-13 to 'Variable1' to 'Variable14'
    variable_names = {i: f'Variable{i+1}' for i in range(14)}  # 0-13 mapped to 'Variable1'-'Variable14'
    shap_series = pd.Series(shap_importances, index=variable_names)
    models_aggregated_shap_values['RF'] = shap_series
    #models_aggregated_shap_values['Random Forest'] = aggregate_explanations(shap_values_list)
    models_aggregated_pfi_values['RF'] = aggregate_pfi_values(pfi_values_list, X_test.columns)
    models_aggregated_lime_values['RF'] = aggregate_lime_values(lime_values_list, X_test.columns)

  
def gb_model_with_gridsearch(df, instance_number):
    """
    Perform hyperparameter tuning for GradientBoostingRegressor and return the best model.
    Saves the best hyperparameters to a JSON file.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - instance_number: str or int, identifier for the current instance.

    Returns:
    - best_estimator_: the best GradientBoostingRegressor model found by GridSearchCV or loaded from file.
    """
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']

    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'max_features': [None, 'sqrt', 'log2'],
    }
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define the hyperparameter file path
    hyperparams_file = os.path.join(HYPERPARAMS_DIR, str(instance_number), f'gb_{instance_number}_hyperparams.json')
    os.makedirs(os.path.dirname(hyperparams_file), exist_ok=True)


    if os.path.exists(hyperparams_file):
        # Load hyperparameters from file
        with open(hyperparams_file, 'r') as f:
            best_params = json.load(f)
        print(f'Loading pre-saved GradientBoostingRegressor hyperparameters for instance {instance_number}: {best_params}')
        # Set the model's parameters
        model.set_params(**best_params)
    else:
        # Perform grid search
        start_time = time.time()
        print('\nGB Grid search commencing...')
        grid_search = GridSearchCV(model, param_grid, cv=kfold, scoring='r2', return_train_score=True, n_jobs=-1)
        grid_search.fit(X, y)
        elapsed_time = time.time() - start_time
        print(f"\tElapsed time to compute the GB params: {elapsed_time:.3f} seconds")
        print("\tBest parameters:", grid_search.best_params_)
        print("\tBest R2 score:", grid_search.best_score_)

        # Save the best hyperparameters to file
        best_params = grid_search.best_params_
        with open(hyperparams_file, 'w') as f:
            json.dump(best_params, f)
        print(f'\tSaved best GradientBoostingRegressor hyperparameters to {hyperparams_file}')

        # Set the model to best estimator
        model = grid_search.best_estimator_

    return model


def gb_regression_model(df, instance_number, models_aggregated_shap_values, models_aggregated_pfi_values, models_aggregated_lime_values, ground_truth_importances=None):
    """
    Train GradientBoostingRegressor models using K-fold cross-validation and compute SHAP values.

    Parameters:
    - df: pandas DataFrame containing features and target variable.
    - model: GradientBoostingRegressor model with best hyperparameters.
    - instance_number: str, instance identifier for saving plots.

    Returns:
    - None
    """
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']
    n_splits = 10
    cv_scores = []
    shap_values_list = []
    pfi_values_list = []
    lime_values_list = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    print("\nComputing Gradient Boost Regression models...")
    #lime_explainer = init_lime()

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        # arrays  to DataFrames to give explainers labels to help with ground truth comparisons, without labels gives us worse performance alongisde the wrong mapping for importance scores
        X_train_df = pd.DataFrame(X_train, columns=X_train.columns, index=X_train.index)
        X_test_df = pd.DataFrame(X_test, columns=X_test.columns, index=X_test.index)
        model = GradientBoostingRegressor(random_state=42)
        #lime_explainer = init_lime(X_train)
        model.fit(X_train_df, y_train)
        prediction = model.predict(X_test_df)
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        print(f"\nGB R2 score for fold {fold}: {r2:.7f}")

        # Use TreeExplainer
        explainer = get_shap_explainer(model, X_train_df)
        shap_values = explainer.get_explanations(X_test_df)
        shap_values_list.append(shap_values)

        # LIME explanations
        lime_explainer = LimeTabularExplainer(
            training_data=X_train_df.values,
            feature_names=X_train_df.columns.tolist(),
            mode='regression',
            discretize_continuous=False
        )
        lime_explanations = []
        for i in range(X_test_df.shape[0]):
            exp = lime_explainer.explain_instance(
                data_row=X_test_df.iloc[i].values,
                predict_fn=model.predict,
                num_features=len(X_test_df.columns)
            )
            lime_explanations.append(dict(exp.as_list()))
        
            # Aggregate LIME explanations without taking absolute values
            lime_importances = np.zeros(len(X_test_df.columns))
            for exp in lime_explanations:
                for feature, importance in exp.items():
                    if feature in X_test_df.columns:
                        idx = X_test_df.columns.get_loc(feature)
                        #lime_values_list.append(importance)
                        lime_importances[idx] += abs(importance)  # Preserve sign

            lime_importances /= X_test_df.shape[0]

               # Compute PFI values
        #pfi_explainer = pfi(model, X_test, y_test, metric='r2', n_repeats=5, seed=42)
        #pfi_importances = pfi_explainer.get_explanations(X_test)
        result = permutation_importance(
            model,
            X_test_df,
            y_test,
            scoring='r2',  # or 'neg_mean_squared_error' for MSE
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        
        pfi_importances = result.importances_mean
        pfi_values_list.append(pfi_importances)
        
        # Plot SHAP summary
        plot_shap_summary(shap_values, X_test_df, 'Gradient Boosting', fold, instance_number)
        process_file(instance_number, n_splits, model, X_test_df, X_train_df, y_train, y_test, r2, fold, explainer, shap_values)


        # If ground truth importances are provided, compute metrics
        if ground_truth_importances is not None:
            # Aggregate SHAP values to get explainer importances
            shap_importances = pd.Series(
                data=np.abs(shap_values).mean(axis=0),
                index=X_test_df.columns                
            )
            pfi_imp = pd.Series(
                data=pfi_importances,
                index=X_test_df.columns
            )

            lime_importances_series = pd.Series(
                data=lime_importances,
                index=X_test_df.columns
            )
            lime_values_list.append(lime_importances_series.values)


            # print("PFI Importances:\n", pfi_imp)
            # print("PFI Importances contain NaN:", pfi_imp.isna().any())
            # print("PFI Importances contain Inf:", np.isinf(pfi_imp).any())
            # Compute metrics SHAP
            pra_score = compute_PRA_with_ground_truth(ground_truth_importances, shap_importances)
            print("\n\tSHAP")
            print(f"\t\tSHAP PRA for fold {fold}: {pra_score:.4f}")
            spearman_corr, spearman_p = compute_spearman_correlation(ground_truth_importances, shap_importances)
            print(f"\t\tSpearman's Rank Correlation for fold {fold}: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
            kendall_tau, kendall_p = compute_kendall_tau(ground_truth_importances, shap_importances)
            print(f"\t\tKendall's Tau for fold {fold}: {kendall_tau:.4f} (p-value: {kendall_p:.4e})")
            precision_k = compute_precision_at_k(ground_truth_importances, shap_importances, k=5)
            print(f"\t\tPrecision at K=5 for fold {fold}: {precision_k:.4f} \n")

            # Compute metrics PFI
            pra_score = compute_PRA_with_ground_truth(ground_truth_importances, pfi_imp)
            print("\n\tPFI")
            print(f"\t\tPFI PRA for fold {fold}: {pra_score:.4f}")
            spearman_corr, spearman_p = compute_spearman_correlation(ground_truth_importances, pfi_imp)
            print(f"\t\tSpearman's Rank Correlation for fold {fold}: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
            kendall_tau, kendall_p = compute_kendall_tau(ground_truth_importances, pfi_imp)
            print(f"\t\tKendall's Tau for fold {fold}: {kendall_tau:.4f} (p-value: {kendall_p:.4e})")
            precision_k = compute_precision_at_k(ground_truth_importances, pfi_imp, k=5)
            print(f"\t\tPrecision at K=5 for fold {fold}: {precision_k:.4f} \n")

            # Compute metrics LIME
            pra_score = compute_PRA_with_ground_truth(ground_truth_importances, lime_importances_series)
            print("\n\tLIME")
            print(f"\t\tLIME PRA for fold {fold}: {pra_score:.4f}")
            spearman_corr, spearman_p = compute_spearman_correlation(ground_truth_importances, lime_importances_series)
            print(f"\t\tSpearman's Rank Correlation for fold {fold}: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
            kendall_tau, kendall_p = compute_kendall_tau(ground_truth_importances, lime_importances_series)
            print(f"\t\tKendall's Tau for fold {fold}: {kendall_tau:.4f} (p-value: {kendall_p:.4e})")
            precision_k = compute_precision_at_k(ground_truth_importances, lime_importances_series, k=5)
            print(f"\t\tPrecision at K=5 for fold {fold}: {precision_k:.4f} \n")
    
    print("Average Cross-Validation R2 GB:", sum(cv_scores) / len(cv_scores))
    models_aggregated_shap_values['Gradient Boosting'] = aggregate_explanations(shap_values_list)
    models_aggregated_pfi_values['Gradient Boosting'] = aggregate_pfi_values(pfi_values_list)
        # Aggregate LIME values
    models_aggregated_lime_values['Gradient Boosting'] = aggregate_lime_values(lime_values_list)

def setup_logger():
    """
    Sets up a logger that writes to a file based on the script name and current timestamp.
    
    Returns:
        logger: Configured logger.
    """
    # Determine the script name
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    
    # Create a logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique log file name using the script name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create a logger with the script name
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    # Prevent adding multiple handlers to the same logger
    if not logger.handlers:
        # Create a file handler to write logs to the file
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(logging.INFO)
        
        # Create a console handler to output logs to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Define a logging format
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def extract_number(file_name):
    """
    Extract the numerical part of the filename.

    Parameters:
    - file_name: str, filename to extract number from.

    Returns:
    - int, extracted number.
    """
    match = re.search(r'binary_(\d+)\.csv', file_name)
    return int(match.group(1)) if match else float('inf')
# def aggregate_metrics(metrics_aggregated):
#     """
#     Compute mean, median, and standard deviation for each metric and explainer.
    
#     Parameters:
#     - metrics_aggregated: dict, container with lists of metric values.
    
#     Returns:
#     - aggregated_results: dict, with aggregated statistics.
#     """
#     aggregated_results = {}
#     logger = setup_logger()
#     for explainer, metrics in metrics_aggregated.items():
#         aggregated_results[explainer] = {}
#         for metric, values in metrics.items():
#             if not values:
#                 logger.warning(f"No values for {explainer} - {metric}. Skipping aggregation.")
#                 continue  # Skip this metric since there are no values

#             # Check if values are dicts containing 'value' and 'error'
#             if isinstance(values[0], dict) and 'value' in values[0] and 'error' in values[0]:
#                 # Separate the main values and errors/p-values
#                 main_values = []
#                 error_values = []
#                 for v in values:
#                     try:
#                         main_values.append(float(v['value']))
#                         error_values.append(float(v['error']))
#                     except (KeyError, ValueError, TypeError) as e:
#                         logger.error(f"Invalid metric entry for {explainer} - {metric}: {v} - {e}")
#                         continue  # Skip invalid entries

#                 if not main_values or not error_values:
#                     logger.warning(f"No valid entries for {explainer} - {metric}. Skipping aggregation.")
#                     continue

#                 # Compute aggregates for main values
#                 main_mean = np.mean(main_values)
#                 main_median = np.median(main_values)
#                 main_std = np.std(main_values)

#                 # Compute aggregates for error values
#                 error_mean = np.mean(error_values)
#                 error_median = np.median(error_values)
#                 error_std = np.std(error_values)

#                 # Store the aggregates as dict
#                 aggregated_results[explainer][metric] = {
#                     'Mean': {'value': main_mean, 'error': error_mean},
#                     'Median': {'value': main_median, 'error': error_median},
#                     'StdDev': {'value': main_std, 'error': error_std}
#                 }
#             else:
#                 # Assume values are numeric or improperly formatted
#                 try:
#                     values_float = [float(v) for v in values]
#                     if not values_float:
#                         logger.warning(f"No valid numeric entries for {explainer} - {metric}. Skipping aggregation.")
#                         continue

#                     aggregated_results[explainer][metric] = {
#                         'Mean': np.mean(values_float),
#                         'Median': np.median(values_float),
#                         'StdDev': np.std(values_float)
#                     }
#                 except (ValueError, TypeError) as e:
#                     logger.error(f"Invalid metric entries for {explainer} - {metric}: {values} - {e}")
#                     continue  # Skip invalid entries
#     return aggregated_results


def aggregate_metrics(aggregated_results):
    """
    Aggregate metrics by computing median, mean, standard deviation, and interquartile range (IQR).

    Parameters:
    - aggregated_results: dict, with structure:
        {
            'RF': {
                'SHAP': {'ImportanceRatio': [{'value': float}, ...], ...},
                'LIME': {'ImportanceRatio': [{'value': float}, ...], ...},
                'PFI': {'ImportanceRatio': [{'value': float}, ...], ...}
            },
            'MLP': {
                'SHAP': {'ImportanceRatio': [{'value': float}, ...], ...},
                'LIME': {'ImportanceRatio': [{'value': float}, ...], ...},
                'PFI': {'ImportanceRatio': [{'value': float}, ...], ...}
            }
        }

    Returns:
    - dict, aggregated statistics with Median, Mean, StdDev, and IQR for each metric.
    """
    aggregated_stats = {}
    for model, explainers in aggregated_results.items():
        aggregated_stats[model] = {}
        for explainer, metrics in explainers.items():
            aggregated_stats[model][explainer] = {}
            for metric, value_dicts in metrics.items():
                # Statement - if value is above 25,000, print them 
                # Extract 'value' from each dict; ignore any other keys
                values = [v['value'] for v in value_dicts if isinstance(v, dict) and 'value' in v]
                if values:
                    median = np.median(values)
                    mean = np.mean(values)
                    stddev = np.std(values)
                    interquartile_range = iqr(values)
                else:
                    median = mean = stddev = interquartile_range = np.nan  # Handle empty lists gracefully
                aggregated_stats[model][explainer][metric] = {
                    'Median': median,
                    'Mean': mean,
                    'StdDev': stddev,
                    'IQR': interquartile_range
                }
    return aggregated_stats

def transform_all_metrics_records(all_metrics_records):
    """
    Transform a list of metric records into the nested dictionary structure required for aggregation.

    Parameters:
    - all_metrics_records: list of dicts, each dict containing:
        {
            'Degree': str,
            'Instance': int,
            'Model': str,
            'Explainer': str,
            'Metric': str,
            'Value': float,
            'Fold': int or None,
            'SampleID': int or None
        }

    Returns:
    - dict, structured for aggregate_metrics:
        {
            'RF': {
                'SHAP': {'ImportanceRatio': [{'value': float}, ...], ...},
                'LIME': {'ImportanceRatio': [{'value': float}, ...], ...},
                'PFI': {'ImportanceRatio': [{'value': float}, ...], ...}
            },
            'MLP': {
                'SHAP': {'ImportanceRatio': [{'value': float}, ...], ...},
                'LIME': {'ImportanceRatio': [{'value': float}, ...], ...},
                'PFI': {'ImportanceRatio': [{'value': float}, ...], ...}
            }
        }
    """
    aggregated_results = {}
    for record in all_metrics_records:
        model = record['Model']
        explainer = record['Explainer']
        metric = record['Metric']
        value = record['Value']
        
        # Initialize nested dictionaries if they don't exist
        if model not in aggregated_results:
            aggregated_results[model] = {}
        if explainer not in aggregated_results[model]:
            aggregated_results[model][explainer] = {}
        if metric not in aggregated_results[model][explainer]:
            aggregated_results[model][explainer][metric] = []
        
        # Append the value
        aggregated_results[model][explainer][metric].append({'value': value})
    
    return aggregated_results

def save_aggregated_metrics_to_csv(aggregated_metrics, csv_path):
    """
    Save the aggregated metrics to a CSV file.

    Parameters:
    - aggregated_metrics: dict, aggregated statistics with Median, Mean, StdDev, and IQR.
    - csv_path: str, path to save the CSV file.

    Returns:
    - None
    """
    records = []
    for model, explainers in aggregated_metrics.items():
        for explainer, metrics in explainers.items():
            for metric, stats in metrics.items():
                record = {
                    'Model': model,
                    'Explainer': explainer,
                    'Metric': metric,
                    'Median': stats.get('Median', np.nan),
                    'Mean': stats.get('Mean', np.nan),
                    'StdDev': stats.get('StdDev', np.nan),
                    'IQR': stats.get('IQR', np.nan)
                }
                records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    logger.info("Aggregated metrics saved to %s.", csv_path)


def plot_aggregated_heatmap(explainer_values, method_name, heatmap_dir):

    # Create a mapping from feature names to numbers
    feature_names = explainer_values.index.tolist()
    feature_numbers = list(range(1, len(feature_names) + 1))  # Start numbering from 1

    feature_mapping = dict(zip(feature_names, feature_numbers))
    explainer_values.index = explainer_values.index.map(feature_mapping)

    plt.figure(figsize=(30, 15))
    ax = sns.heatmap(
        explainer_values,
        annot=True,
        cmap='viridis',
        fmt='.3f',
        annot_kws={"size": 50}
    )
    # Set y-axis labels to feature names
    #ax.set_yticklabels(explainer_values.index, rotation=0, fontsize=50)
    # Set x-axis labels to model names
    ax.set_xticklabels(explainer_values.columns, ha='right', fontsize=50)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=40)
    
# ax.set_yticklabels(range(1, explainer_values.shape[0]-1), rotation=0)
    ax.set_ylabel('Variables in Class', fontsize=50)
    ax.set_xlabel('Models', fontsize=50)
    ax.set_title(f'{method_name} Feature Importance Across Models\n', fontsize=55)
    color_bar = ax.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=45)  # Adjust tick label size
    color_bar.ax.set_ylabel("Importance Score", fontsize=50)  # Set label and font size
    color_bar.ax.yaxis.label.set_size(50)  # Ensure label size
    plt.tight_layout()
    os.makedirs(heatmap_dir, exist_ok=True)
    plt.savefig(os.path.join(heatmap_dir, f'{method_name.lower()}_aggregated_heatmap.png'), dpi=200)
    plt.close()


def plot_normalized_heatmap(explainer_values, method_name, heatmap_dir):
    # Normalize the DataFrame by dividing each value by the maximum value in its row
    # Create a mapping from feature names to numbers
    feature_names = explainer_values.index.tolist()
    feature_numbers = list(range(1, len(feature_names) + 1))  # Start numbering from 1

    feature_mapping = dict(zip(feature_names, feature_numbers))
    explainer_values.index = explainer_values.index.map(feature_mapping)
    ##normalized_importances = explainer_values.div(explainer_values.max(axis=1), axis=0)
    normalized_importances = explainer_values.apply(lambda col: col / col.max())
    plt.figure(figsize=(30, 15))
    ax = sns.heatmap(normalized_importances, annot=True, cmap='coolwarm', fmt='.3f', annot_kws={"size": 50})
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=40)
    #ax.set_yticklabels(range(1, normalized_importances.shape[0]-), rotation=0)
    ax.set_xticklabels(explainer_values.columns, ha='right', fontsize=50)
    #ax.set_yticklabels(explainer_values.index, rotation=0, fontsize=50)
    ax.set_ylabel('Variables in Class', fontsize=50)
    ax.set_xlabel('Models', fontsize=50)
    color_bar = ax.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=45)  # Adjust tick label size
    color_bar.ax.set_ylabel("Importance Score", fontsize=50)  # Set label and font size
    color_bar.ax.yaxis.label.set_size(50)  # Ensure label size
    ax.set_title(f'Normalised {method_name} Feature Importances Across Models', fontsize=55)
    plt.savefig(os.path.join(heatmap_dir, f'{method_name.lower()}_normalized_heatmap.png'), dpi=200)
    plt.close()
  
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
                # Create a mapping from feature names to numbers
        feature_names = df.index.tolist()
        feature_numbers = list(range(1, len(feature_names) + 1))  # Start numbering from 1

        feature_mapping = dict(zip(feature_names, feature_numbers))
        df.index = df.index.map(feature_mapping)
        plt.figure(figsize=(30, 15))
        ax = sns.heatmap(df, annot=True, cmap='viridis', fmt='.3f', annot_kws={"size": 50})
       # ax.set_yticklabels(range(1, PROBLEM_SIZE), rotation=0)
        ax.set_ylabel('Variables in Class', fontsize=50)
        ax.set_xlabel('Models', fontsize=50)
        ax.set_yticklabels(df.index, rotation=0, fontsize=50)
       # ax.set_yticklabels(ax.get_yticklabels(), fontsize=40)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=40)
        color_bar = ax.collections[0].colorbar
        color_bar.ax.tick_params(labelsize=45)  # Adjust tick label size
        color_bar.ax.set_ylabel("Importance Score", fontsize=50)  # Set label and font size
        color_bar.ax.yaxis.label.set_size(50)  # Ensure label size
        ax.set_title(title, fontsize=65)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        logger.info(f"Heatmap '{title}' saved to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to plot heatmap {filename}: {e}")


def plot_variability_heatmap(explanation_list, method_name, heatmap_dir):
    # Stack DataFrames into a 3D NumPy array
    explanation_arrays = np.array([df.values for df in explanation_list])  # Shape: (num_runs, num_variables, num_models)

    # Compute the standard deviation across runs (axis=0)
    std_explanations = np.std(explanation_arrays, axis=0)  # Shape: (num_variables, num_models)

    # Convert back to DataFrame
    variables = explanation_list[0].index
    models = explanation_list[0].columns
    variability_df = pd.DataFrame(std_explanations, index=variables, columns=models)

    # Plot the heatmap
    plt.figure(figsize=(30, 15))
    ax = sns.heatmap(variability_df, annot=True, cmap='coolwarm', fmt='.3f', annot_kws={"size": 50})
    ax.set_yticklabels(range(1, variability_df.shape[0] + 1), rotation=0)
    ax.set_ylabel('Variables in Class', fontsize=50)
    ax.set_xlabel('Models', fontsize=50)
    ax.set_title(f'{method_name} Feature Importance Variability Across Models', fontsize=55)
    color_bar = ax.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=45)  # Adjust tick label size
    color_bar.ax.set_ylabel("Importance Score", fontsize=50)  # Set label and font size
    color_bar.ax.yaxis.label.set_size(50)  # Ensure label size
    plt.savefig(os.path.join(heatmap_dir, f'{method_name.lower()}_variability_heatmap.png'), dpi=200)
    plt.close()

import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def save_raw_metrics_per_instance(metrics_aggregated, degree, instance_number, output_dir='puboi'):
    """
    Save raw Spearman, KendallTau, and PrecisionAtK metrics for each ML model and XAI technique to a CSV file for a given instance.

    Parameters:
    - metrics_aggregated: dict, aggregated metrics.
    - degree: str, degree identifier ('deg2', 'deg10', etc.).
    - instance_number: int, identifier for the instance.
    - output_dir: str, base directory to save the raw metrics file.

    Returns:
    - None
    """
    records = []
    for model, explainers in metrics_aggregated.items():
        for explainer, metrics in explainers.items():
            for metric, value_dicts in metrics.items():
                for value_dict in value_dicts:
                    # Initialize the record with common fields
                    record = {
                        'Degree': degree,
                        'Instance': instance_number,
                        'Model': model,
                        'Explainer': explainer,
                        'Metric': metric,
                        'Value': value_dict['value']
                    }
                    
                    
                    records.append(record)

    if records:
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Define the directory and file path
        instance_dir = os.path.join(output_dir, degree, str(instance_number))
        os.makedirs(instance_dir, exist_ok=True)
        raw_metrics_csv_path = os.path.join(instance_dir, 'raw_metrics.csv')
        
        # Save to CSV
        df.to_csv(raw_metrics_csv_path, index=False)
        
        logger.info(f"Raw metrics for Instance {instance_number}, Degree {degree} saved to {raw_metrics_csv_path}.")
    else:
        logger.warning(f"No valid raw metrics found for Instance {instance_number}, Degree {degree}.")

def save_aggregated_metrics(all_metrics_records, output_dir='puboi', aggregated_filename='aggregated_metrics.csv', run_id=1):
    """
    Save aggregated Importance Ratio metrics across all instances to a single CSV file.

    Parameters:
    - all_metrics_records: list of dicts, each dict representing a metric entry from any instance.
    - output_dir: str, base directory to save the aggregated metrics file.
    - aggregated_filename: str, name of the aggregated CSV file.

    Returns:
    - None
    """
    if all_metrics_records:
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics_records)
        
        # Define the file path with run identifier
        aggregated_csv_path = os.path.join(output_dir, aggregated_filename.format(run_id))
        os.makedirs(output_dir, exist_ok=True)
        
        # Append to CSV if it exists, else create it
        if os.path.exists(aggregated_csv_path):
            df.to_csv(aggregated_csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(aggregated_csv_path, mode='w', header=True, index=False)
        
        logger.info(f"Aggregated metrics saved to {aggregated_csv_path}.")
        
        logger.info(f"Aggregated metrics across all instances saved to {aggregated_csv_path}.")
    else:
        logger.warning("No metrics found to aggregate.")

def main():
    # Command-line argument parsing (if any)
    # [Existing argparse code]
    parser = argparse.ArgumentParser(description="Process subsets of instances for PUBOi and ML model training.")

    parser.add_argument("--instances", type=str, default=None, 
                        help="Comma-separated list or ranges of instance numbers to process (e.g., '1000-1030,1050-1080'). If not provided, all instances are processed.")
    parser.add_argument("--run-id", type=int, default=1, help="Unique identifier for the current run.")

    args = parser.parse_args()

    aggregated_data = {
        'deg2': {
            'shap_importances': [],
            'pfi_importances': [],
            'lime_importances': [],
            'metrics_aggregated': {
                'RF': {
                    'SHAP': {'ImportanceRatio': []},
                    'LIME': {'ImportanceRatio': []},
                    'PFI': {'ImportanceRatio': []}
                },
                'MLP': {
                    'SHAP': {'ImportanceRatio': []},
                    'LIME': {'ImportanceRatio': []},
                    'PFI': {'ImportanceRatio': []}
                }
            },
            'r2_scores': {'MLP': [], 'RF': []}
        },
        # 'deg10': {
        #     'shap_importances': [],
        #     'pfi_importances': [],
        #     'lime_importances': [],
        #     'metrics_aggregated': {
        #         'RF': {
        #             'SHAP': {'ImportanceRatio': []},
        #             'LIME': {'ImportanceRatio': []},
        #             'PFI': {'ImportanceRatio': []}
        #         },
        #         'MLP': {
        #             'SHAP': {'ImportanceRatio': []},
        #             'LIME': {'ImportanceRatio': []},
        #             'PFI': {'ImportanceRatio': []}
        #         }
        #     },
        #     'r2_scores': {'MLP': [], 'RF': []}
        # }
    }
    # Set display option to suppress scientific notation globally
    pd.set_option('display.float_format', '{:.3f}'.format)
    logger = setup_logger()
    
    # List all files in the 'model_data' directory
    path = 'puboi/model_data'
    os.makedirs(path, exist_ok=True)
    all_files = os.listdir(path)
    csv_files = [file for file in all_files if file.endswith('.csv')]
    csv_files = sorted(csv_files, key=extract_number)
    
    # Define degrees and their respective instance numbers
    degrees_info = {
        'deg2': range(1000, 1030)  # 30 instances
        # 'deg10': range(1050, 1080)  # 30 instances
    }


    # Parse the --instances argument to get a set of instance numbers to process
    selected_instances = None
    if args.instances:
        selected_instances = set()
        parts = args.instances.split(',')
        for part in parts:
            if '-' in part:
                start, end = part.split('-')
                selected_instances.update(range(int(start), int(end)+1))
            else:
                selected_instances.add(int(part))
        logger.info(f"Selected instances to process: {sorted(selected_instances)}")
    # Initialize a list to collect all metrics across instances
    all_metrics_records = []
    #print(f"Starting Term Iteration: {term_iteration}/{termcond}")
    for degree, instance_range in degrees_info.items():
        for instance_number in instance_range:
                        # Add the instance check here
            if selected_instances is not None and instance_number not in selected_instances:
                continue  # Skip this instance
            logger.info(f"Processing Degree: {degree}, Instance: {instance_number}")
            # Define the file path for the current instance
            filepath = os.path.join(path, f'puboi_{degree}_{instance_number}.csv')
            # Initialise metrics_aggregated for the current instance
            metrics_aggregated = {
                'RF': {
                    'SHAP': {'ImportanceRatio': []},
                    'LIME': {'ImportanceRatio': []},
                    'PFI': {'ImportanceRatio': []}
                },
                'MLP': {
                    'SHAP': {'ImportanceRatio': []},
                    'LIME': {'ImportanceRatio': []},
                    'PFI': {'ImportanceRatio': []}
                }
            }
            if not os.path.exists(filepath):
                logger.error(f"File {filepath} does not exist. Skipping Instance: {instance_number}.")
                continue  # Skip to the next instance
    
            # Load the dataset
            df = pd.read_csv(filepath)
            logger.info(f"Loaded data for Instance {instance_number} from {filepath}.")
    
            # Load ground truth importances
            ground_truth_filepath = f'../../../instances/small/puboi_1000seed_{degree}/puboi_{instance_number}_vars.csv'
            if not os.path.exists(ground_truth_filepath):
                logger.error(f"Ground truth file {ground_truth_filepath} does not exist. Skipping Instance: {instance_number}.")
                continue  # Skip to the next instance
    
            ground_truth_df = pd.read_csv(ground_truth_filepath)
            vars_in_class_to_feature = {i: f'Variable{i+1}' for i in range(15)}
            ground_truth_df['feature_name'] = ground_truth_df['vars_in_class'].map(vars_in_class_to_feature)
            ground_truth_importances = pd.Series(
                data=ground_truth_df['var_importance'].values,
                index=ground_truth_df['feature_name']
            )
    
            # Recode ground truth importances if needed
            ground_truth_importances_recode = ground_truth_importances.map({0: 1, 1: 0})
            logger.info(f"Loaded ground truth importances for Instance {instance_number}.")
    
            # Initialize temporary aggregation containers for this instance
            temp_shap = {}
            temp_pfi = {}
            temp_lime = {}

            logger.info(f"Instance {instance_number}: Starting Random Forest training.")
            rf_regression_model(
                df=df,
                instance_number=instance_number,
                degree=degree,
                models_aggregated_shap_values=temp_shap,
                models_aggregated_pfi_values=temp_pfi,
                models_aggregated_lime_values=temp_lime,
                ground_truth_importances=ground_truth_importances_recode,
                metrics_aggregated=metrics_aggregated,
                r2_scores=aggregated_data[degree]['r2_scores'],
                run_number=None
            )
            logger.info(f"Instance {instance_number}: Completed Random Forest training.")
    
            logger.info(f"Instance {instance_number}: Starting MLP training.")
            nn_regression(
                df=df,
                instance_number=instance_number,
                degree=degree,
                models_aggregated_shap_values=temp_shap,
                models_aggregated_pfi_values=temp_pfi,
                models_aggregated_lime_values=temp_lime,
                ground_truth_importances=ground_truth_importances_recode,
                metrics_aggregated=metrics_aggregated,
                r2_scores=aggregated_data[degree]['r2_scores'],
                run_number=None
            )
            logger.info(f"Instance {instance_number}: Completed MLP training.")
    
            # After processing all models for the current instance, aggregate explanations
            logger.info(f"Aggregating explanations for Instance: {instance_number}")
    
            combined_shap_values, combined_pfi_values, combined_lime_values = combine_model_explanations(
                temp_shap,
                temp_pfi,
                temp_lime
            )
    
            # Clean combined SHAP values
            combined_shap_values = combined_shap_values.apply(pd.to_numeric, errors='coerce')
            combined_shap_values.replace([np.inf, -np.inf], np.nan, inplace=True)
            combined_shap_values.fillna(value=0, inplace=True)
            combined_shap_values.dropna(inplace=True)
    
            # Create a mapping from 0-13 to 'Variable1' to 'Variable14'
            index_mapping = {i: f'{i+1}' for i in range(14)}  # 0-13 mapped to 'Variable1'-'Variable14'


            # Adjust the keys in the inner dictionaries
            variables = {i: f'{i+1}' for i in range(14)}
            # Convert to DataFrames
            shap_df = pd.DataFrame(combined_shap_values)
            pfi_df = pd.DataFrame(combined_pfi_values)
            lime_df = pd.DataFrame(combined_lime_values)
    
            # Append to the degree-specific aggregation lists
            aggregated_data[degree]['shap_importances'].append(shap_df)
            aggregated_data[degree]['pfi_importances'].append(pfi_df)
            aggregated_data[degree]['lime_importances'].append(lime_df)
    
            # Define heatmap directory for this instance
            heatmap_dir = os.path.join('puboi', degree, str(instance_number), 'heatmap')
            os.makedirs(heatmap_dir, exist_ok=True)
    
            # Plot SHAP heatmap
            shap_heatmap_path = os.path.join(heatmap_dir, 'shap-median.png')
            plot_and_save_heatmap(
                combined_shap_values,
                f'Model Performance with SHAP for {degree} - Instance {instance_number}',
                shap_heatmap_path
            )
            logger.info(f"Instance {instance_number}: SHAP heatmap saved to {shap_heatmap_path}.")
    
            # Plot PFI heatmap
            pfi_heatmap_path = os.path.join(heatmap_dir, 'pfi-median.png')
            plot_and_save_heatmap(
                combined_pfi_values,
                f'Model Performance with PFI for {degree} - Instance {instance_number}',
                pfi_heatmap_path
            )
            logger.info(f"Instance {instance_number}: PFI heatmap saved to {pfi_heatmap_path}.")
    
            # Plot LIME heatmap
            lime_heatmap_path = os.path.join(heatmap_dir, 'lime-median.png')
            plot_and_save_heatmap(
                combined_lime_values,
                f'Model Performance with LIME for {degree} - Instance {instance_number}',
                lime_heatmap_path
            )
            logger.info(f"Instance {instance_number}: LIME heatmap saved to {lime_heatmap_path}.")
            # aggregated_metrics = aggregate_metrics(aggregated_data[degree]['metrics_aggregated'])
            # metrics_csv_path = os.path.join('puboi', degree, str(instance_number), 'aggregated_metrics.csv')
            # save_aggregated_metrics_to_csv(aggregated_metrics, metrics_csv_path)
            # logger.info(f"Aggregated metrics for {str(instance_number)}, {degree} saved to {metrics_csv_path}.")
            # Save raw metrics for this instance
            save_raw_metrics_per_instance(
                metrics_aggregated,
                degree,
                instance_number,
                output_dir='puboi'
            )
            # Append current instance's metrics to all_metrics_records for aggregation
            for model, explainers in metrics_aggregated.items():
                for explainer, metrics in explainers.items():
                    for metric, value_dicts in metrics.items():
                        for value_dict in value_dicts:
                            record = {
                                'Degree': degree,
                                'Instance': instance_number,
                                'Model': model,
                                'Explainer': explainer,
                                'Metric': metric,
                                'Value': value_dict['value']
                            }
                            all_metrics_records.append(record)
            # Transform all_metrics_records into the required nested dictionary
            aggregated_results = transform_all_metrics_records(all_metrics_records)
            #   Aggregate the metrics
            aggregated_stats = aggregate_metrics(aggregated_results)
            # Save the aggregated statistics to a CSV for further analysis
            # Flatten the aggregated_stats and save as a CSV
            records = []
            for model, explainers in aggregated_stats.items():
                for explainer, metrics in explainers.items():
                    for metric, stats in metrics.items():
                        record = {
                            'Model': model,
                            'Explainer': explainer,
                            'Metric': metric,
                            'Median': stats['Median'],
                            'Mean': stats['Mean'],
                            'StdDev': stats['StdDev'],
                            'IQR': stats['IQR']
                        }
                        records.append(record)
            
            if records:
                aggregated_stats_df = pd.DataFrame(records)
                aggregated_stats_csv_path = os.path.join(f'puboi/{degree}/{str(instance_number)}', 'aggregated_metrics_summary.csv')
                os.makedirs(f'puboi/{degree}/{str(instance_number)}', exist_ok=True)
                aggregated_stats_df.to_csv(aggregated_stats_csv_path, index=False)
                logger.info(f"Aggregated statistics saved to {aggregated_stats_csv_path}.")
            else:
                logger.warning("No aggregated statistics to save.")
            
        # Additionally, save all metrics across instances if needed
        save_aggregated_metrics(
            all_metrics_records=all_metrics_records,
            output_dir='puboi',
            aggregated_filename='enum-aggregated_metrics.csv'
        )
        logger.info(f"Instance {instance_number}: Raw metrics saved.")
    
         
    # After processing all runs, aggregate the explanations
    heatmap_dir = 'puboi/aggregated_heatmaps'
    os.makedirs(heatmap_dir, exist_ok=True)
    logger.info("Aggregating and plotting final heatmaps across all instances.")

    def aggregate_explanations_across_runs(explanation_list):
        """
        Aggregate explanations across multiple runs.

        Parameters:
        - explanation_list: List of DataFrames containing explanations.

        Returns:
        - Aggregated DataFrame.
        """
        if not explanation_list:
            logger.warning("No explanations to aggregate.")
            return pd.DataFrame()

        shapes = [df.shape for df in explanation_list]
        if len(set(shapes)) != 1:
            logger.error("All explanation DataFrames must have the same shape.")
            return pd.DataFrame()

        explanation_arrays = np.array([df.values for df in explanation_list])
        mean_explanations = np.median(explanation_arrays, axis=0)
        return pd.DataFrame(mean_explanations, index=explanation_list[0].index, columns=explanation_list[0].columns)

   # After processing all instances, aggregate and plot per degree
    for degree in degrees_info.keys():
        # if not aggregated_data[degree]['metrics_aggregated']['SHAP']['Spearman']:
        #     logger.info(f"No data for Degree: {degree}. Skipping aggregation.")
        #     continue
        heatmap_dir = os.path.join('puboi', degree, 'aggregated_heatmaps')
        os.makedirs(heatmap_dir, exist_ok=True)
        logger.info(f"Aggregating and plotting heatmaps for Degree: {degree}")
    
        # Aggregate SHAP, PFI, and LIME explanations across all instances for this degree
        shap_aggregated = aggregate_explanations_across_runs(aggregated_data[degree]['shap_importances'])
        pfi_aggregated = aggregate_explanations_across_runs(aggregated_data[degree]['pfi_importances'])
        lime_aggregated = aggregate_explanations_across_runs(aggregated_data[degree]['lime_importances'])
        shap_aggregated.to_csv(f'{heatmap_dir}/shap_aggregated.csv')
        lime_aggregated.to_csv(f'{heatmap_dir}/lime_aggregated.csv')
        pfi_aggregated.to_csv(f'{heatmap_dir}/pfi_aggregated.csv')
        # Plot aggregated SHAP heatmap
        if not shap_aggregated.empty:
            shap_heatmap_path = os.path.join(heatmap_dir, 'shap_aggregated_heatmap.png')
            plot_aggregated_heatmap(
                shap_aggregated,
                f'{(degree)} SHAP',
                shap_heatmap_path
            )
            logger.info(f"Aggregated SHAP heatmap for {degree} saved to {shap_heatmap_path}.")

            # Optionally, also plot normalized heatmap
            shap_normalized_heatmap_path = os.path.join(heatmap_dir, 'shap_normalized_heatmap.png')
            plot_normalized_heatmap(
                shap_aggregated,
                f'{(degree)} SHAP',
                heatmap_dir
            )
            logger.info(f"Normalized SHAP heatmap for {degree} saved to {shap_normalized_heatmap_path}.")

        # Plot aggregated PFI heatmap
        if not pfi_aggregated.empty:
            pfi_heatmap_path = os.path.join(heatmap_dir, 'pfi_aggregated_heatmap.png')
            plot_aggregated_heatmap(
                pfi_aggregated,
                f'{(degree)} PFI ',
                pfi_heatmap_path
            )
            logger.info(f"Aggregated PFI heatmap for {degree} saved to {pfi_heatmap_path}.")

            # Optionally, also plot normalized heatmap
            pfi_normalized_heatmap_path = os.path.join(heatmap_dir, 'pfi_normalized_heatmap.png')
            plot_normalized_heatmap(
                pfi_aggregated,
                f'{(degree)} PFI',
                heatmap_dir
            )
            logger.info(f"Normalized PFI heatmap for {degree} saved to {pfi_normalized_heatmap_path}.")

        # Plot aggregated LIME heatmap
        if not lime_aggregated.empty:
            lime_heatmap_path = os.path.join(heatmap_dir, 'lime_aggregated_heatmap.png')
            plot_aggregated_heatmap(
                lime_aggregated,
                f'{(degree)} LIME ',
                lime_heatmap_path
            )
            logger.info(f"Aggregated LIME heatmap for {degree} saved to {lime_heatmap_path}.")

            # Optionally, also plot normalized heatmap
            lime_normalized_heatmap_path = os.path.join(heatmap_dir, 'lime_normalized_heatmap.png')
            plot_normalized_heatmap(
                lime_aggregated,
                f'{(degree)} LIME',
                heatmap_dir
            )
            logger.info(f"Normalized LIME heatmap for {degree} saved to {lime_normalized_heatmap_path}.")

    
        # # Plot variability heatmaps
        # if aggregated_data[degree]['shap_importances']:
        #     plot_variability_heatmap(
        #         aggregated_data[degree]['shap_importances'],
        #         f'{degree}-SHAP',
        #         heatmap_dir
        #     )
        #     logger.info(f"Variability SHAP heatmap for {degree} saved to {os.path.join(heatmap_dir, 'shap_variability_heatmap.png')}.")
        # if aggregated_data[degree]['pfi_importances']:
        #     plot_variability_heatmap(
        #         aggregated_data[degree]['pfi_importances'],
        #         f'{degree}-PFI',
        #         heatmap_dir
        #     )
        #     logger.info(f"Variability PFI heatmap for {degree} saved to {os.path.join(heatmap_dir, 'pfi_variability_heatmap.png')}.")
        # if aggregated_data[degree]['lime_importances']:
        #     plot_variability_heatmap(
        #         aggregated_data[degree]['lime_importances'],
        #         f'{degree}-LIME',
        #         heatmap_dir
        #     )
        #     logger.info(f"Variability LIME heatmap for {degree} saved to {os.path.join(heatmap_dir, 'lime_variability_heatmap.png')}.")
    
        # # Aggregate and save metrics
        # aggregated_metrics = aggregate_metrics(aggregated_data[degree]['metrics_aggregated'])
        # metrics_csv_path = os.path.join(heatmap_dir, 'aggregated_metrics.csv')
        # save_aggregated_metrics_to_csv(aggregated_metrics, metrics_csv_path)
        # logger.info(f"Aggregated metrics for {degree} saved to {metrics_csv_path}.")

            # After processing all instances, save the aggregated metrics
        save_aggregated_metrics(
        all_metrics_records=all_metrics_records,
        output_dir='puboi',
        aggregated_filename='enum-aggregated_metrics.csv'
        )   
    
        # Calculate and plot average R² scores
        avg_r2_across_runs = {model: np.median(scores) for model, scores in aggregated_data[degree]['r2_scores'].items()}
        models = list(avg_r2_across_runs.keys())
        avg_r2_values = list(avg_r2_across_runs.values())
    
        # Plotting R² scores
        plt.figure(figsize=(12, 8))
        bars = plt.bar(models, avg_r2_values, color=['skyblue', 'lightgreen', 'salmon'])
    
        # Annotate bars with formatted average R² scores
        for bar, avg_r2 in zip(bars, avg_r2_values):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f'{avg_r2:.5f}',
                ha='center',
                va='bottom',
                fontsize=20
            )
    
        # Customizing the plot
        plt.ylabel("Median R² Score", fontsize=30)
        plt.xlabel("Models", fontsize=30)
        plt.title(f"{degree} Model Performance Across Runs", fontsize=40)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(os.path.join(heatmap_dir, 'model_performance.png'), dpi=200)
        plt.close()
        logger.info(f"Model performance plot for {degree} saved to {os.path.join(heatmap_dir, 'model_performance.png')}.")
    
        # Save R² scores to a CSV for further analysis
        r2_df = pd.DataFrame(aggregated_data[degree]['r2_scores'])
        r2_csv_path = os.path.join(heatmap_dir, 'r2_scores_median.csv')
        r2_df.to_csv(r2_csv_path, index=False)
        logger.info(f"R² scores for {degree} saved to {r2_csv_path}.")
    
    # Final Aggregated Plots Across All Degrees (Optional)
    # If you want to compare heatmaps across degrees, you can create additional plots here.
    # For example, side-by-side heatmaps or combined metrics.

    logger.info("Completed all PUBOi ML model trainings.")
if __name__ == "__main__":
    main()
