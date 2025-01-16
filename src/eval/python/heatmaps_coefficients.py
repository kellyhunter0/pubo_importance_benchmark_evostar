#!/usr/bin/env python3
"""
Heatmap Generator Script for RF and LR Coefficients

This script generates a combined heatmap for Random Forest (RF) and Linear Regression (LR) coefficients based on provided CSV files.
It retains the styling and font sizes from the original configuration and remaps feature names to numerical identifiers for clarity.

Usage:
    python heatmaps.py --files puboi/aggregated_results/lr_aggregated_coefficients.csv puboi/aggregated_results/rf_aggregated_coefficients.csv

Example:
    python heatmaps.py \
        puboi/aggregated_results/lr_aggregated_coefficients.csv \
        puboi/aggregated_results/rf_aggregated_coefficients.csv
"""

import argparse
import pandas as pd
import os
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns

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
    Parses the file path to extract the model name.

    Expected file path format:
    path/to/model_aggregated_coefficients.csv

    Example:
    puboi/aggregated_results/lr_aggregated_coefficients.csv

    Returns:
        model (str)
    """
    filename = os.path.basename(file_path)
    if 'lr' in filename.lower():
        return 'LR'
    elif 'rf' in filename.lower():
        return 'RF'
    else:
        raise ValueError(f"Cannot determine model type from filename '{filename}'. Expected 'lr' or 'rf' in the filename.")

def read_coefficients(file_path, model):
    """
    Reads the coefficients CSV file into a DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.
    - model (str): Model name ('LR' or 'RF').

    Returns:
    - pd.DataFrame: DataFrame with features as index and a single column for the model.
    """
    try:
        df = pd.read_csv(file_path, index_col=0)
        # Ensure that the DataFrame has exactly one column
        if df.shape[1] != 1:
            raise ValueError(f"Expected exactly one column for coefficients in '{file_path}', but found {df.shape[1]}.")
        df.columns = [model]  # Rename the column to the model name
                # Apply absolute value to LR coefficients
        if model == 'LR':
            df[model] = df[model].abs()
        return df
    except Exception as e:
        raise ValueError(f"Error reading '{file_path}': {e}")

def remap_features_to_numbers(df):
    """
    Remaps feature names to numerical identifiers.

    Parameters:
    - df (pd.DataFrame): DataFrame with features as index.

    Returns:
    - pd.DataFrame: DataFrame with numerical feature identifiers as index.
    - dict: Mapping from numerical identifiers to original feature names.
    """
    feature_names = df.index.tolist()
    feature_numbers = list(range(1, len(feature_names) + 1))  # Start numbering from 1
    feature_mapping = dict(zip(feature_names, feature_numbers))
    df_numeric = df.copy()
    df_numeric.index = df_numeric.index.map(feature_mapping)
    return df_numeric, feature_mapping

def combine_coefficients(lr_df, rf_df):
    """
    Combines LR and RF coefficients into a single DataFrame.

    Parameters:
    - lr_df (pd.DataFrame): DataFrame containing LR coefficients.
    - rf_df (pd.DataFrame): DataFrame containing RF coefficients.

    Returns:
    - pd.DataFrame: Combined DataFrame with numerical feature identifiers as rows and models as columns.
    - dict: Mapping from numerical identifiers to original feature names.
    """
    # Remap features to numbers for both DataFrames
    lr_df_numeric, lr_mapping = remap_features_to_numbers(lr_df)
    rf_df_numeric, rf_mapping = remap_features_to_numbers(rf_df)
    
    # Ensure both mappings are identical
    if lr_mapping != rf_mapping:
        print("Feature mappings between LR and RF do not match. Aligning based on union of features.")
        # Combine mappings
        all_features = set(lr_mapping.keys()).union(set(rf_mapping.keys()))
        combined_mapping = {feature: idx for idx, feature in enumerate(sorted(all_features), 1)}
        
        # Remap both DataFrames
        lr_df_numeric = lr_df.copy()
        lr_df_numeric.index = lr_df_numeric.index.map(combined_mapping)
        
        rf_df_numeric = rf_df.copy()
        rf_df_numeric.index = rf_df_numeric.index.map(combined_mapping)
        
    else:
        combined_mapping = lr_mapping  # or rf_mapping
    
    # Combine the DataFrames
    combined_df = pd.concat([lr_df_numeric, rf_df_numeric], axis=1)
    
    # Handle missing features by filling with 0 or NaN
    combined_df = combined_df.fillna(0)  # Alternatively, use another method as appropriate
    
    return combined_df, combined_mapping

def plot_combined_heatmap(combined_df, feature_mapping, output_dir):
    """
    Generates and saves a combined heatmap for RF and LR coefficients.

    Parameters:
    - combined_df (pd.DataFrame): DataFrame with numerical feature identifiers as rows and models as columns.
    - feature_mapping (dict): Mapping from numerical identifiers to original feature names.
    - output_dir (str): Directory where the heatmap image will be saved.

    Returns:
    - None
    """
    logger = logging.getLogger(__name__)
    
    # Reverse mapping for display if needed
    reverse_mapping = {v: k for k, v in feature_mapping.items()}
    
    # Set figure size based on the number of features and models
    num_features = combined_df.shape[0]
    num_models = combined_df.shape[1]
    
    # Adjust figsize dynamically; feel free to tweak these multipliers
    figsize_x = max(50, num_models * 2)  # Width
    figsize_y = max(33, num_features * 0.3)  # Height
    
    plt.figure(figsize=(figsize_x, figsize_y))
    
    # Create a mapping from feature numbers to sorted order for better visualization
    sorted_features = combined_df.index.tolist()
    
    # Plot the heatmap
    ax = sns.heatmap(
        combined_df,
        annot=True,
        cmap='viridis',
        fmt='.3f',
        annot_kws={"size": 150},  # Adjust annotation font size
        cbar_kws={'label': 'Importance Score'}
    )
    
    # Set x-axis labels to model names with specified font size
    ax.set_xticklabels(combined_df.columns, rotation=45, ha='right', fontsize=150)
    
    # Set y-axis labels to numerical feature identifiers with specified font size
    ax.set_yticklabels(sorted_features, rotation=0, fontsize=150)
    
    # Set axis labels and title with specified font sizes
   # ax.set_xlabel('Model', fontsize=14)
   # ax.set_ylabel('Features', fontsize=14)
   # ax.set_title('RF vs. LR Coefficients Heatmap', fontsize=16, pad=20)
    
    # Customize color bar label font size
    color_bar = ax.collections[0].colorbar
    color_bar.ax.tick_params(labelsize=150)  # Adjust tick label size
    color_bar.ax.set_ylabel("Importance Score", fontsize=150)  # Set label and font size
    color_bar.ax.yaxis.label.set_size(150)  # Ensure label size
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the heatmap as a high-resolution PNG
    save_path = os.path.join(output_dir, 'RF_LR_coefficients_heatmap.png')
    plt.savefig(save_path, dpi=200)  # Increased DPI for better clarity
    plt.close()
    
    logger.info(f"Heatmap saved to {save_path}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a combined heatmap for RF and LR coefficients from CSV files.")
    parser.add_argument('--files', nargs=2, required=True, help='Paths to the LR and RF aggregated coefficients CSV files.')
    parser.add_argument('--output_dir', type=str, default='heatmaps', help='Directory to save the heatmap image.')
    args = parser.parse_args()
    
    # Set up logger
    logger = setup_logger()
    logger.info("Starting Heatmap Generation Process...")
    
    # Validate the number of files
    if len(args.files) != 2:
        logger.error("Exactly two CSV files must be provided: one for LR and one for RF.")
        sys.exit(1)
    
    # Initialize DataFrames
    lr_df = rf_df = None
    
    # Process each file
    for file_path in args.files:
        logger.info(f"Processing file: {file_path}")
        try:
            model = parse_file_path(file_path)
            logger.info(f"Identified model: {model}")
            df = read_coefficients(file_path, model)
            if model == 'LR':
                lr_df = df
            elif model == 'RF':
                rf_df = df
        except ValueError as ve:
            logger.error(ve)
            sys.exit(1)
    
    # Ensure both DataFrames are loaded
    if lr_df is None or rf_df is None:
        logger.error("Both LR and RF coefficients must be provided and correctly formatted.")
        sys.exit(1)
    
    # Combine the coefficients
    combined_df, feature_mapping = combine_coefficients(lr_df, rf_df)
    logger.info(f"Combined DataFrame shape: {combined_df.shape}")
    
    # Plot the heatmap
    try:
        plot_combined_heatmap(combined_df, feature_mapping, args.output_dir)
        logger.info(f"Heatmap successfully saved to '{os.path.join(args.output_dir, 'RF_LR_coefficients_heatmap.png')}'.")
    except Exception as e:
        logger.error(f"Failed to generate heatmap: {e}")
        sys.exit(1)
    
    logger.info("Heatmap Generation Process Completed Successfully.")

if __name__ == "__main__":
    main()
