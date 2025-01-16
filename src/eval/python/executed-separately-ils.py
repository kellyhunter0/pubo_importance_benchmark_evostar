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
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from explainer_utils import get_shap_explainer, init_lime
from model_training import (
    aggregate_explanations,
    aggregate_lime_values,
    aggregate_pfi_values,
    combine_model_explanations,
    compute_kendall_tau,
    compute_precision_at_k,
    compute_spearman_correlation,
    extract_number,
    linear_regression_with_shap,
    rf_regression_model,
    nn_regression,
    plot_aggregated_heatmap,
    plot_normalized_heatmap,
    plot_shap_summary,
    plot_variability_heatmap,
    aggregate_metrics,
    save_aggregated_metrics_to_csv,
    setup_logger,
    transform_all_metrics_records
)
import logging

sys.path.insert(0, '../../generator')
from walsh_expansion import WalshExpansion

# Initialize Logger
logger = setup_logger()

# Define problem size
PROBLEM_SIZE = 14

# Number of ILS runs per instance
RUNS_PER_INSTANCE = 10

# Initialize global evaluation count
global perturbation_strength
global evalcount
evalcount = 0
termcond = 30
runs = 10
perturbation_strength = 3
logger = setup_logger()

def save_aggregated_metrics(all_metrics_records, output_dir='ils-split', aggregated_filename='aggregated_metrics.csv'):
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
        
        # Define the file path
        aggregated_csv_path = os.path.join(output_dir, aggregated_filename)
        os.makedirs(output_dir, exist_ok=True)
        


        df.to_csv(aggregated_csv_path, header=True, index=False)
        
        logger.info(f"Aggregated metrics saved to {aggregated_csv_path}.")
        
        logger.info(f"Aggregated metrics across all instances saved to {aggregated_csv_path}.")
    else:
        logger.warning("No metrics found to aggregate.")

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
        color_bar.ax.tick_params(labelsize=30)  # Adjust tick label size
        color_bar.ax.set_ylabel("Importance Score", fontsize=30)  # Set label and font size
        color_bar.ax.yaxis.label.set_size(30)  # Ensure label size
        ax.set_title(title, fontsize=65)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        logger.info(f"Heatmap '{title}' saved to '{filename}'")
    except Exception as e:
        logger.error(f"Failed to plot heatmap {filename}: {e}")


def save_raw_metrics_per_instance(metrics_aggregated, degree, instance_number, run_number, output_dir='ils-split'):
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
                        'Instance-run': f'{instance_number}-{run_number}',
                        'Model': model,
                        'Explainer': explainer,
                        'Metric': metric,
                        'Value': value_dict['value']
                    }
                    
                    # # Only add 'Error' if it exists (i.e., not for PrecisionAtK)
                    # if metric not in ['PrecisionAtK']:
                    #     record['Error'] = value_dict.get('error', None)
                    
                    records.append(record)

    if records:
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Define the directory and file path
        instance_dir = os.path.join(output_dir, degree, str(instance_number))
        os.makedirs(instance_dir, exist_ok=True)
        raw_metrics_csv_path = os.path.join(instance_dir, f'Run{run_number}-raw_metrics.csv')
        
        # Save to CSV
        df.to_csv(raw_metrics_csv_path, index=False)
        
        logger.info(f"Raw metrics for Instance {instance_number}, Run {run_number}, Degree {degree}. Saved to {raw_metrics_csv_path}.")
    else:
        logger.warning(f"No valid raw metrics found for Instance {instance_number} Run {run_number}, Degree {degree}.")

def process_all_runs(degrees_info, path, aggregated_data, all_metric_records, selected_instances=None):
    """
    Process all CSV files with multiple runs per instance across different degrees.
    Optionally process only a selected subset of instances.

    Parameters:
    - degrees_info: dict, mapping degrees to a list or range of instance numbers.
    - path: str, directory path where CSV files are located.
    - aggregated_data: dict, container with nested dictionaries for each degree.
    - selected_instances: set or list of instance numbers to process. If None, process all.

    Returns:
    - None
    """
    all_metric_records = []
    for degree, instance_range in degrees_info.items():
        for instance_number in instance_range:
            if selected_instances is not None and instance_number not in selected_instances:
                continue  # Skip instances not in the selected subset

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
            expected_runs = RUNS_PER_INSTANCE
            if len(instance_files) != expected_runs:
                logger.warning(f"Expected {expected_runs} runs for Degree: {degree}, Instance: {instance_number}, but found {len(instance_files)} runs.")

            logger.info(f"Processing Degree: {degree}, Instance: {instance_number}")

            # Initialize aggregation dictionaries for this instance
            temp_shap = {}
            temp_pfi = {}
            temp_lime = {}

            # Load ground truth importances (assuming it's the same across runs)
            ground_truth_filepath = f'../../../instances/small/puboi_1000seed_{degree}/puboi_{instance_number}_vars.csv'
            if not os.path.exists(ground_truth_filepath):
                logger.error(f"Ground truth file {ground_truth_filepath} does not exist. Skipping Instance: {instance_number}.")
                continue  # Skip to the next instance

            ground_truth_df = pd.read_csv(ground_truth_filepath)
            vars_in_class_to_feature = {i: f'Variable{i+1}' for i in range(PROBLEM_SIZE+1)}
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
                    logger.warning(f"File {run_file} does not match the expected pattern. Skipping.")
                    continue

                run_degree, run_instance, run_number = run_match.groups()
                run_number = int(run_number)  # Ensure it's an integer
                logger.info(f"  Processing Run {run_number} for Instance {instance_number}")
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
                # Read the run-specific CSV file
                df = pd.read_csv(run_file)

                # # Train and evaluate all models on this run
                # logger.info(f"Instance {instance_number}: Starting Linear Regression training.")
                # linear_regression_with_shap(
                #     df=df,
                #     instance_number=instance_number,
                #     degree=degree,
                #     models_aggregated_shap_values=temp_shap,
                #     models_aggregated_pfi_values=temp_pfi,
                #     models_aggregated_lime_values=temp_lime,
                #     ground_truth_importances=ground_truth_importances_recode,
                #     metrics_aggregated=aggregated_data[degree]['metrics_aggregated'],
                #     r2_scores=aggregated_data[degree]['r2_scores'],
                #     run_number=run_number  # Pass run_number
                # )
                # logger.info(f"Instance {instance_number}: Completed Linear Regression training.")

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
                    run_number=run_number  # Pass run_number
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
                    run_number=run_number  # Pass run_number
                )
                logger.info(f"Instance {instance_number}: Completed MLP training.")
                save_raw_metrics_per_instance(
                    metrics_aggregated,
                    degree,
                    instance_number,
                    run_number
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
                                all_metric_records.append(record)
                # Transform all_metrics_records into the required nested dictionary
                aggregated_results = transform_all_metrics_records(all_metric_records)
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
                    aggregated_stats_csv_path = os.path.join('ils-split', f'{degree}', f'{str(instance_number)}',f'Run{str(run_number)}-aggregated_metrics_summary.csv')
                    os.makedirs('ils-split', exist_ok=True)
                    aggregated_stats_df.to_csv(aggregated_stats_csv_path, index=False)
                    logger.info(f"Aggregated statistics saved to {aggregated_stats_csv_path}.")
                else:
                    logger.warning("No aggregated statistics to save.") 

                logger.info(f"Instance {instance_number}: Raw metrics saved.")
            # After processing all runs for the current instance, aggregate explanations
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
            index_mapping = {i: f'Variable{i+1}' for i in range(14)}  # 0-13 mapped to 'Variable1'-'Variable14'


            # Adjust the keys in the inner dictionaries
            variables = {i: f'Variable{i+1}' for i in range(14)}
            # Convert to DataFrames
            shap_df = pd.DataFrame(combined_shap_values, index=index_mapping)
            pfi_df = pd.DataFrame(combined_pfi_values)
            lime_df = pd.DataFrame(combined_lime_values)

            # Append to the degree-specific aggregation lists
            aggregated_data[degree]['shap_importances'].append(shap_df)
            aggregated_data[degree]['pfi_importances'].append(pfi_df)
            aggregated_data[degree]['lime_importances'].append(lime_df)

            # Define heatmap directory for this instance
            heatmap_dir = os.path.join('ils-split', degree, str(instance_number), 'heatmap')
            os.makedirs(heatmap_dir, exist_ok=True)

            # Plot SHAP heatmap
            shap_heatmap_path = os.path.join(heatmap_dir, 'shap-median.png')
            plot_and_save_heatmap(
                combined_shap_values,
                f'Model Performance with SHAP for ILS - Instance {instance_number} ({degree})',
                shap_heatmap_path
            )

            # Plot PFI heatmap
            pfi_heatmap_path = os.path.join(heatmap_dir, 'pfi-median.png')
            plot_and_save_heatmap(
                combined_pfi_values,
                f'Model Performance with PFI for ILS - Instance {instance_number} ({degree})',
                pfi_heatmap_path
            )

            # Plot LIME heatmap
            lime_heatmap_path = os.path.join(heatmap_dir, 'lime-median.png')
            plot_and_save_heatmap(
                combined_lime_values,
                f'Model Performance with LIME for ILS - Instance {instance_number} ({degree})',
                lime_heatmap_path
            )

            # save_raw_metrics_per_instance(
            #     aggregated_data[degree]['metrics_aggregated'],
            #     degree,
            #     instance_number,
            #     run_number
            # )

            logger.info(f"Instance {instance_number}: Raw metrics saved.")
            

def main():
    """
    Main function to execute the ILS and ML training process.

    Usage:
        python executed-separately-ils.py <termcond> <runs_per_instance> <perturbation_strength> --instances 1000-1050,1060-1100
    """
    parser = argparse.ArgumentParser(description="Process subsets of instances for ILS and ML model training.")
    parser.add_argument("termcond", type=int, help="Termination condition parameter.")
    parser.add_argument("runs_per_instance", type=int, help="Number of runs per instance.")
    parser.add_argument("perturbation_strength", type=int, help="Perturbation strength parameter.")
    parser.add_argument("--instances", type=str, default=None, 
                        help="Comma-separated list or ranges of instance numbers to process (e.g., '1000-1030,1050-1080'). If not provided, all instances are processed.")

    args = parser.parse_args()

    termcond, runs_per_instance, perturbation = args.termcond, args.runs_per_instance, args.perturbation_strength

    # Update global variables based on command-line arguments
    global RUNS_PER_INSTANCE
    global perturbation_strength
    RUNS_PER_INSTANCE = runs_per_instance
    perturbation_strength = perturbation

    output_dir = 'ils-split/mined_solutions'
    os.makedirs(output_dir, exist_ok=True)

    # Define degrees and their respective instance numbers
    degrees_info = {
        'deg2': range(1000, 1030),   # 30 instances
        'deg10': range(1050, 1080)  # 30 instances
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
        'deg10': {
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
        }
    }
    # # Initialize lists to store aggregated importances across all degrees
    # #  separate aggregated plots per degree)
    all_metric_records = []
    # all_pfi_importances = []
    # all_lime_importances = []

    # Call the processing function
    process_all_runs(
        degrees_info=degrees_info,
        path=output_dir,
        aggregated_data=aggregated_data,
        all_metric_records=all_metric_records,
        selected_instances=selected_instances
    )

    # After processing all runs, aggregate and plot per degree
    for degree in degrees_info.keys():
        # if not aggregated_data[degree]['metrics_aggregated']['RF']['SHAP']['ImportanceRatio']:
        #     logger.info(f"No importance data for RF Degree: {degree}. Skipping aggregation.")
        # if not aggregated_data[degree]['metrics_aggregated']['MLP']['SHAP']['ImportanceRatio']:
        #     logger.info(f"No importance data for RF Degree: {degree}. Skipping aggregation.")
        #     continue
        heatmap_dir = os.path.join('ils-split', degree, 'aggregated_heatmaps')
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
        #     variability_shap_heatmap_path = os.path.join(heatmap_dir, 'shap_variability_heatmap.png')
        #     plot_variability_heatmap(
        #         aggregated_data[degree]['shap_importances'],
        #         f'{degree}-SHAP',
        #         heatmap_dir
        #     )
        #     logger.info(f"Variability SHAP heatmap for {degree} saved to {variability_shap_heatmap_path}.")
        # if aggregated_data[degree]['pfi_importances']:
        #     variability_pfi_heatmap_path = os.path.join(heatmap_dir, 'pfi_variability_heatmap.png')
        #     plot_variability_heatmap(
        #         aggregated_data[degree]['pfi_importances'],
        #         f'{degree}-PFI',
        #         heatmap_dir
        #     )
        #     logger.info(f"Variability PFI heatmap for {degree} saved to {variability_pfi_heatmap_path}.")
        # if aggregated_data[degree]['lime_importances']:
        #     variability_lime_heatmap_path = os.path.join(heatmap_dir, 'lime_variability_heatmap.png')
        #     plot_variability_heatmap(
        #         aggregated_data[degree]['lime_importances'],
        #         f'{degree}-LIME',
        #         heatmap_dir
        #     )
        #     logger.info(f"Variability LIME heatmap for {degree} saved to {variability_lime_heatmap_path}.")

        # # Aggregate and save metrics
        # aggregated_metrics = aggregate_metrics(aggregated_data[degree]['metrics_aggregated'])
        # metrics_csv_path = os.path.join(heatmap_dir, 'aggregated_metrics.csv')
        # save_aggregated_metrics_to_csv(aggregated_metrics, metrics_csv_path)
        # logger.info(f"Aggregated metrics for {degree} saved to {metrics_csv_path}.")

        # After processing all instances, save the aggregated metrics
        save_aggregated_metrics(
        all_metrics_records=all_metric_records,
        output_dir='ils-split',
        aggregated_filename='sample-aggregated_metrics.csv'
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
        plt.ylabel("Average R² Score", fontsize=30)
        plt.xlabel("Models", fontsize=30)
        plt.title(f"{degree} Model Performance Across Runs", fontsize=40)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        r2_plot_path = os.path.join(heatmap_dir, 'model_performance.png')
        plt.savefig(r2_plot_path, dpi=200)
        plt.close()
        logger.info(f"Model performance plot for {degree} saved to {r2_plot_path}.")

        # Save R² scores to a CSV for further analysis
        r2_df = pd.DataFrame(aggregated_data[degree]['r2_scores'])
        r2_csv_path = os.path.join(heatmap_dir, 'r2_scores_median.csv')
        r2_df.to_csv(r2_csv_path, index=False)
        logger.info(f"R² scores for {degree} saved to {r2_csv_path}.")


    logger.info("Completed all ILS runs and ML model trainings.")

if __name__ == "__main__":
    main()
