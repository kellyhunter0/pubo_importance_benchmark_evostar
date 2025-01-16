#!/usr/bin/env python3
"""
Heatmap Generator Script

This script generates heatmaps for feature importances based on CSV files.
Each CSV file should correspond to a specific experiment, degree, and method.

Usage:
    python generate_heatmaps.py --files path/to/shap_aggregated.csv path/to/lime_aggregated.csv path/to/pfi_aggregated.csv

Example:
    python generate_heatmaps.py \
        puboi/deg2/aggregated_heatmaps/shap_aggregated.csv \
        puboi/deg2/aggregated_heatmaps/lime_aggregated.csv \
        puboi/deg2/aggregated_heatmaps/pfi_aggregated.csv \
        ils-split/deg2/aggregated_heatmaps/shap_aggregated.csv \
        ils-split/deg2/aggregated_heatmaps/lime_aggregated.csv \
        ils-split/deg2/aggregated_heatmaps/pfi_aggregated.csv
"""

import argparse
import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_aggregated_heatmap(explainer_values, method_name, heatmap_dir, title_suffix=''):
    """
    Generates and saves a heatmap based on the provided explainer values.

    Parameters:
    - explainer_values (pd.DataFrame): DataFrame containing feature importances.
    - method_name (str): Name of the explainer method (e.g., 'SHAP', 'LIME', 'PFI').
    - heatmap_dir (str): Directory where the heatmap image will be saved.
    - title_suffix (str): Additional string to append to the plot title.

    Returns:
    - None
    """
    logger = logging.getLogger(__name__)
    
    # Create a mapping from feature names to numbers
    feature_names = explainer_values.index.tolist()
    feature_numbers = list(range(1, len(feature_names) + 1))  # Start numbering from 1
    feature_mapping = dict(zip(feature_names, feature_numbers))
    explainer_values.index = explainer_values.index.map(feature_mapping)

    # Set figure size based on the number of features and models
    num_features = explainer_values.shape[0]
    num_models = explainer_values.shape[1]
    
    # Adjust figsize dynamically; feel free to tweak these multipliers
    figsize_x = max(50, num_models * 2)  # Width
    figsize_y = max(33, num_features * 0.3)  # Height
    
    plt.figure(figsize=(figsize_x, figsize_y))
    ax = sns.heatmap(
        explainer_values,
        annot=True,
        cmap='viridis',
        fmt='.3f',
        annot_kws={"size": 150},  # Adjust annotation font size
        cbar_kws={'label': 'Importance Score'}
    )
    
    # Set x-axis labels to model names
    ax.set_xticklabels(explainer_values.columns, rotation=45, ha='right', fontsize=150)
    
    # Set y-axis labels to feature numbers
    ax.set_yticklabels(explainer_values.index, rotation=0, fontsize=150)
    
    # Set axis labels and title
   # ax.set_ylabel('Variables in Class', fontsize=14)
   # ax.set_xlabel('Models', fontsize=14)
    #ax.set_title(f'{method_name} Feature Importance Across Models {title_suffix}', fontsize=16, pad=20)
    color_bar = ax.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=150)  # Adjust tick label size
    color_bar.ax.set_ylabel("Importance Score", fontsize=150)  # Set label and font size
    color_bar.ax.yaxis.label.set_size(150)  # Ensure label size
    plt.tight_layout()
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Save the heatmap as a high-resolution PNG
    save_path = os.path.join(heatmap_dir, f'{method_name.lower()}_aggregated_heatmap.png')
    plt.savefig(save_path, dpi=200)  # Increased DPI for better clarity
    plt.close()
    
    logger.info(f"Heatmap saved to {save_path}")

def setup_logger():
    """
    Sets up the logger for the script.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('heatmap_generator.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    if not logger.handlers:
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger

def parse_file_path(file_path):
    """
    Parses the file path to extract experiment, degree, and method.

    Expected file path format:
    experiment/degree/aggregated_heatmaps/method_aggregated.csv

    Example:
    puboi/deg2/aggregated_heatmaps/shap_aggregated.csv

    Returns:
    - experiment (str)
    - degree (str)
    - method (str)
    """
    parts = os.path.normpath(file_path).split(os.sep)
    try:
        experiment = parts[-4]  # Adjust based on directory depth
        degree = parts[-3]
        method_file = parts[-1]
        method = method_file.split('_')[0].upper()  # Extract method name and uppercase it
    except IndexError:
        raise ValueError(f"File path '{file_path}' does not match the expected format.")
    
    return experiment, degree, method

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate heatmaps from aggregated feature importance CSV files.")
    parser.add_argument('--files', nargs='+', required=True, help='List of CSV file paths to process.')
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Heatmap Generation Process...")
    
    # Process each file
    for file_path in args.files:
        logger.info(f"Processing file: {file_path}")
        
        # Check if file exists
        if not os.path.isfile(file_path):
            logger.error(f"File not found: {file_path}. Skipping.")
            continue
        
        # Parse file path to get experiment, degree, and method
        try:
            experiment, degree, method = parse_file_path(file_path)
            logger.info(f"Parsed Information - Experiment: {experiment}, Degree: {degree}, Method: {method}")
        except ValueError as ve:
            logger.error(ve)
            continue
        
        # Read the CSV file
        try:
            df = pd.read_csv(file_path, index_col=0)
            logger.info(f"Loaded CSV with shape: {df.shape}")
        except Exception as e:
            logger.error(f"Failed to read CSV file '{file_path}'. Error: {e}. Skipping.")
            continue
        
        # Define output directory
        heatmap_dir = os.path.join('heatmaps', experiment, degree)
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Generate a title suffix for additional context
        title_suffix = f"({degree})" if degree else ""
        
        # Plot and save the heatmap
        try:
            plot_aggregated_heatmap(
                explainer_values=df,
                method_name=method,
                heatmap_dir=heatmap_dir,
                title_suffix=title_suffix
            )
            logger.info(f"Successfully generated heatmap for {method} in {experiment}, {degree}.")
        except Exception as e:
            logger.error(f"Failed to generate heatmap for '{file_path}'. Error: {e}. Skipping.")
            continue
    
    logger.info("Heatmap Generation Process Completed.")

if __name__ == "__main__":
    main()
