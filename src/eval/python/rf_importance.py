import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from scipy.stats import iqr
import logging
import argparse
import os
import sys
from glob import glob
import matplotlib.pyplot as plt

# ----------------------------
# 1. Setup Logger
# ----------------------------
def setup_logger():
    """
    Set up the logger for debugging and information.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Remove existing handlers to prevent duplication
    if logger.hasHandlers():
        logger.handlers.clear()
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('model_importance.log')
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    # Create formatters and add to handlers
    c_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    return logger

# ----------------------------
# 2. Filename Parsing
# ----------------------------
def parse_filename(filename):
    """
    Parse the filename to extract degree and instance number.
    
    Expected format: puboi_deg2_1000.csv
    
    Returns:
        degree (str), instance_number (str)
    """
    parts = filename.replace('.csv', '').split('_')
    if len(parts) != 3:
        raise ValueError("Filename does not match the expected format 'puboi_degX_instance.csv'.")
    degree = parts[1]  # 'deg2' or 'deg10'
    instance_number = parts[2]  # e.g., '1000'
    return degree, instance_number

# ----------------------------
# 3. Metric Computation Functions
# ----------------------------

def compute_importance_ratio(importances, abs=False, epsilon=1e-4):
    """
    Compute the ratio of median importance of ground truth variables to median importance of remaining variables.
    Replaces zero values in remaining variables with epsilon to prevent division by zero.
    Assumes the first four variables are the ground truth.

    Parameters:
    - importances (pd.Series): Feature importances with variable names as index.
    - abs (bool): Whether to take absolute values of importances. Defaults to False.
    - epsilon (float): Small value to replace zero importances. Defaults to 1e-4.

    Returns:
    - float: Importance Ratio (median_top4 / median_remaining).
    """
    if abs:
        importances = importances.abs()

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
        print(f"Median of remaining importances ({median_remaining}) is below epsilon ({epsilon}). Using epsilon as denominator.")
        median_remaining_adjusted = epsilon
    else:
        median_remaining_adjusted = median_remaining
    # Compute ratio
    ratio = median_top / median_remaining_adjusted
    print(f"\nRatio: {ratio}")

    return ratio

# ----------------------------
# 4. Model Training Functions
# ----------------------------
def rf_regression_model(
    df,
    instance_number,
    degree,
    ground_truth_importances,
    aggregated_metrics,
    aggregated_importances  # List to store median importances
):
    """
    Train RandomForestRegressor models using K-fold cross-validation and collect feature importances and metrics.

    Parameters:
    - df (pd.DataFrame): DataFrame containing features and target variable.
    - instance_number (str): Instance identifier for logging or saving results.
    - degree (str): Degree identifier (e.g., 'deg2', 'deg10').
    - ground_truth_importances (pd.Series): Ground truth importances with feature names as index.
    - aggregated_metrics (dict): Dictionary to store aggregated metrics.
    - aggregated_importances (list): List to store aggregated feature importances.

    Returns:
    - median_importances_series (pd.Series): Median feature importances across folds.
    - median_r2 (float): Median R² score across folds.
    """
    logger = logging.getLogger(__name__)
    cv_scores = []
    feature_importances_list = []
    ratio_list = []
    
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger.info(f"\nComputing Random Forest models for Instance {instance_number}, Degree {degree}...")
    print(f"\nComputing Random Forest models for Instance {instance_number}, Degree {degree}...")

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = RandomForestRegressor(random_state=42, n_estimators=50)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        logger.info(f"RF R² score for fold {fold}: {r2:.4f}")
        print(f"RF R² score for fold {fold}: {r2:.4f}")
        
        # Collect feature importances from every fold
        feature_importances = model.feature_importances_
        feature_importances_list.append(feature_importances)
        logger.info(f"Fold {fold} - Feature importances collected.")
        print(f"Fold {fold} - Feature importances collected.")
        
        # Convert to Series
        importance_series = pd.Series(feature_importances, index=X.columns)
        
        # Compute Importance Ratio
        ratio = compute_importance_ratio(importance_series)
        ratio_list.append(ratio)
        aggregated_metrics['RF']['ImportanceRatio'].append({'value': ratio})
        
        logger.info(f"Fold {fold} - Importance Ratio: {ratio:.4f}")
        print(f"Fold {fold} - Importance Ratio: {ratio:.4f}")

    if feature_importances_list:
        # Aggregate feature importances across folds using median
        median_importances = np.median(feature_importances_list, axis=0)
        feature_names = X.columns
        median_importances_series = pd.Series(median_importances, index=feature_names)
        aggregated_importances.append(median_importances_series)  # Store in aggregated_importances
        logger.info(f"Aggregated Median Feature Importances for Instance {instance_number}, Degree {degree}:")
        logger.info(median_importances_series)
        print(f"Aggregated Median Feature Importances for Instance {instance_number}, Degree {degree} collected.")
    else:
        logger.warning(f"No feature importances collected for Instance {instance_number}, Degree {degree}.")
        print(f"No feature importances collected for Instance {instance_number}, Degree {degree}.")

    # Return the average R² score to be handled in main()
    average_r2 = np.median(cv_scores) if cv_scores else np.nan
    logger.info(f"Average Cross-Validation R² for RF: {average_r2:.4f}")
    print(f"Average Cross-Validation R² for RF: {average_r2:.4f}")

    return median_importances_series, average_r2

def lr_regression_model(
    df,
    instance_number,
    degree,
    ground_truth_importances,
    aggregated_metrics,
    aggregated_coefficients  # List to store median coefficients
):
    """
    Train LinearRegression models using K-fold cross-validation and collect coefficients and metrics.

    Parameters:
    - df (pd.DataFrame): DataFrame containing features and target variable.
    - instance_number (str): Instance identifier for logging or saving results.
    - degree (str): Degree identifier (e.g., 'deg2', 'deg10').
    - ground_truth_importances (pd.Series): Ground truth importances with feature names as index.
    - aggregated_metrics (dict): Dictionary to store aggregated metrics.
    - aggregated_coefficients (list): List to store aggregated coefficients.

    Returns:
    - median_coefficients_series (pd.Series): Median coefficients across folds.
    - average_r2 (float): Mean R² score across folds.
    """
    logger = logging.getLogger(__name__)
    cv_scores = []
    coefficients_list = []
    ratio_list = []
    
    X = df.drop(['WalshFitness'], axis=1)
    y = df['WalshFitness']
    n_splits = 3
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger.info(f"\nComputing Linear Regression models for Instance {instance_number}, Degree {degree}...")
    print(f"\nComputing Linear Regression models for Instance {instance_number}, Degree {degree}...")

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        r2 = r2_score(y_test, prediction)
        cv_scores.append(r2)
        logger.info(f"LR R² score for fold {fold}: {r2:.4f}")
        print(f"LR R² score for fold {fold}: {r2:.4f}")
        
        # Collect coefficients from every fold
        coefficients = model.coef_
        coefficients_list.append(coefficients)
        logger.info(f"Fold {fold} - Coefficients collected.")
        print(f"Fold {fold} - Coefficients collected.")
        
        # Convert to Series
        coefficient_series = pd.Series(coefficients, index=X.columns)
        
        # Compute Importance Ratio
        ratio = compute_importance_ratio(coefficient_series)
        ratio_list.append(ratio)
        aggregated_metrics['LR']['ImportanceRatio'].append({'value': ratio})
        
        logger.info(f"Fold {fold} - Importance Ratio: {ratio:.4f}")
        print(f"Fold {fold} - Importance Ratio: {ratio:.4f}")

    if coefficients_list:
        # Aggregate coefficients across folds using median
        median_coefficients = np.median(coefficients_list, axis=0)
        feature_names = X.columns
        median_coefficients_series = pd.Series(median_coefficients, index=feature_names)
        aggregated_coefficients.append(median_coefficients_series)  # Store in aggregated_coefficients
        logger.info(f"Aggregated Median Coefficients for Instance {instance_number}, Degree {degree}:")
        logger.info(median_coefficients_series)
        print(f"Aggregated Median Coefficients for Instance {instance_number}, Degree {degree} collected.")
    else:
        logger.warning(f"No coefficients collected for Instance {instance_number}, Degree {degree}.")
        print(f"No coefficients collected for Instance {instance_number}, Degree {degree}.")

    # Return the average R² score to be handled in main()
    average_r2 = np.median(cv_scores) if cv_scores else np.nan
    logger.info(f"Average Cross-Validation R² for LR: {average_r2:.4f}")
    print(f"Average Cross-Validation R² for LR: {average_r2:.4f}")

    return median_coefficients_series, average_r2

# ----------------------------
# 5. Aggregation Function
# ----------------------------
def aggregate_metrics(metrics_aggregated, degree):
    """
    Compute mean, median, standard deviation, and IQR for each metric and model.

    Parameters:
    - metrics_aggregated (dict): Dictionary containing lists of metric values.
    - degree (str): Degree identifier (e.g., 'deg2', 'deg10').

    Returns:
    - aggregated_results (list of dict): List containing aggregated statistics with degree included.
    """
    aggregated_results = []
    logger = logging.getLogger(__name__)
    for model, metrics in metrics_aggregated.items():
        for metric, values in metrics.items():
            if not values:
                logger.warning(f"No values for {model} - {metric} in {degree}. Skipping aggregation.")
                continue  # Skip this metric since there are no values
            
            # Extract 'value' from each dict
            main_values = [v['value'] for v in values if 'value' in v]
            
            if not main_values:
                logger.warning(f"No valid entries for {model} - {metric} in {degree}. Skipping aggregation.")
                continue
            
            # Compute aggregates
            mean_val = np.mean(main_values)
            median_val = np.median(main_values)
            std_val = np.std(main_values)
            iqr_val = iqr(main_values)
            
            # Append to aggregated_results
            aggregated_results.append({
                'Degree': degree,
                'Model': model,
                'Metric': metric,
                'Mean': mean_val,
                'Median': median_val,
                'StdDev': std_val,
                'IQR': iqr_val
            })
    return aggregated_results

# ----------------------------
# 6. Aggregation and Plotting Functions
# ----------------------------
def aggregate_metrics_series(aggregated_list, feature_names):
    """
    Aggregate feature importances or coefficients across all instances and degrees.

    Parameters:
    - aggregated_list (list of pd.Series): List containing median importances or coefficients.
    - feature_names (list of str): List of feature names.

    Returns:
    - pd.Series: Aggregated median importances or coefficients.
    """
    if not aggregated_list:
        return pd.Series(dtype=float)
    df_metrics = pd.DataFrame(aggregated_list)
    median_metrics = df_metrics.median()
    return median_metrics

def plot_importances(df, model_degree, output_path, top_n=10, logger=None):
    """
    Plot feature importances or coefficients.

    Parameters:
    - df (pd.Series): Series containing importances or coefficients.
    - model_degree (str): e.g., 'RF_deg2'.
    - output_path (str): Path to save the plot.
    - top_n (int): Number of top features to plot.
    - logger (logging.Logger): Logger instance.

    Returns:
    - None
    """
    if df.empty:
        if logger:
            logger.warning(f"No data available to plot for {model_degree}.")
        return

    plt.figure(figsize=(12, 8))
    sorted_importances = df.sort_values(ascending=False).head(top_n)
    sorted_importances.plot(kind='bar', color='skyblue')
    plt.title(f'Feature {"Importances" if "RF" in model_degree else "Coefficients"} for {model_degree}', fontsize=20)
    plt.xlabel('Features', fontsize=16)
    plt.ylabel('Median Importance/Coefficient', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    if logger:
        logger.info(f"Plot saved to {output_path}.")

# ----------------------------
# 7. Save Aggregated Metrics to CSV
# ----------------------------
def save_aggregated_metrics_to_csv(aggregated_results_all, output_dir):
    """
    Save aggregated metrics across all instances to CSV files per degree.

    Parameters:
    - aggregated_results_all (list of dict): List containing aggregated statistics.
    - output_dir (str): Directory to save the CSV files.

    Returns:
    - None
    """
    df = pd.DataFrame(aggregated_results_all)
    degrees = df['Degree'].unique()
    
    for degree in degrees:
        df_degree = df[df['Degree'] == degree]
        if df_degree.empty:
            continue
        csv_path = os.path.join(output_dir, f'aggregated_metrics_{degree}.csv')
        df_degree.to_csv(csv_path, index=False)
        logging.getLogger(__name__).info(f"Aggregated metrics for Degree {degree} saved to {csv_path}.")
        print(f"Aggregated metrics for Degree {degree} saved to {csv_path}.")

# ----------------------------
# 8. Main Function
# ----------------------------
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process CSV files to compute and aggregate Random Forest feature importances and Linear Regression coefficients."
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='puboi/model_data',
        help='Directory containing CSV files to process.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='puboi/aggregated_results',
        help='Directory to save aggregated importances, coefficients, and plots.'
    )
    args = parser.parse_args()

    # Set up logger
    logger = setup_logger()
    logger.info("Starting Feature Importance and Coefficient Aggregation...")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize aggregation containers
    degrees = ['deg2']
    aggregated_data = {
        'deg2':{
            'rf_importances': [],
            'lr_coefficients': [],
            'ground_truth_importances': [],  # To store ground truth importances
            'metrics_aggregated': {
                'RF': {'ImportanceRatio': []},
                'LR': {'ImportanceRatio': []}
            },
            'r2_scores': {'RF': [], 'LR': []}
        }

    }

    # Initialize lists to store all median importances and coefficients for plotting
    all_rf_importances = []
    all_lr_coefficients = []

    # List all CSV files in the data directory
    csv_files = glob(os.path.join(args.data_dir, '*.csv'))
    if not csv_files:
        logger.error(f"No CSV files found in directory {args.data_dir}. Exiting.")
        sys.exit(1)
    logger.info(f"Found {len(csv_files)} CSV files to process.")

    for file_path in csv_files:
        # Extract degree and instance number from filename
        # Assuming filename format: puboi_deg2_1000.csv or puboi_deg10_1050.csv
        filename = os.path.basename(file_path)
        try:
            degree, instance_number = parse_filename(filename)
            if degree not in degrees:
                logger.warning(f"Degree {degree} not recognized. Skipping file {filename}.")
                continue
        except ValueError as e:
            logger.warning(f"Filename {filename} does not match expected format. Skipping. Error: {e}")
            continue

        logger.info(f"Processing file: {filename}, Degree: {degree}, Instance: {instance_number}")

        # Load the dataset
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path} with shape {df.shape}.")
        except Exception as e:
            logger.error(f"Failed to read {file_path}. Error: {e}")
            continue

        # Check if 'WalshFitness' is present
        if 'WalshFitness' not in df.columns:
            logger.error(f"'WalshFitness' column not found in {filename}. Skipping.")
            continue

        # Load ground truth importances
        # Adjust the path as per your directory structure
        ground_truth_filepath = f'../../../instances/small/puboi_1000seed_{degree}/puboi_{instance_number}_vars.csv'
        if not os.path.exists(ground_truth_filepath):
            logger.error(f"Ground truth file {ground_truth_filepath} does not exist. Skipping Instance: {instance_number}.")
            continue  # Skip to the next instance

        try:
            ground_truth_df = pd.read_csv(ground_truth_filepath)
        except Exception as e:
            logger.error(f"Failed to read ground truth file {ground_truth_filepath}. Error: {e}")
            continue

        # Map 'vars_in_class' to 'VariableX'
        vars_in_class_to_feature = {i: f'Variable{i+1}' for i in range(len(ground_truth_df))}
        ground_truth_df['feature_name'] = ground_truth_df['vars_in_class'].map(vars_in_class_to_feature)
        ground_truth_importances = pd.Series(
            data=ground_truth_df['var_importance'].values,
            index=ground_truth_df['feature_name']
        )

        # Recode ground truth importances if needed (e.g., invert if necessary)
        # Adjust the mapping based on your specific requirement
        ground_truth_importances_recode = ground_truth_importances.copy()  # Keeping it the same as initial
        logger.info(f"Loaded ground truth importances for Instance {instance_number}.")
        print(f"Loaded ground truth importances for Instance {instance_number}.")

        # Train RF and collect feature importances and metrics
        median_importances_rf, average_r2_rf = rf_regression_model(
            df=df,
            instance_number=instance_number,
            degree=degree,
            ground_truth_importances=ground_truth_importances_recode,
            aggregated_metrics=aggregated_data[degree]['metrics_aggregated'],  # Pass RF metrics
            aggregated_importances=aggregated_data[degree]['rf_importances']        # Pass rf_importances list
        )
        if not median_importances_rf.empty:
            aggregated_data[degree]['rf_importances'].append(median_importances_rf)
            all_rf_importances.append(median_importances_rf)
            # Append R² score to 'r2_scores'
            aggregated_data[degree]['r2_scores']['RF'].append(average_r2_rf)

        # Train LR and collect coefficients and metrics
        median_coefficients_lr, average_r2_lr = lr_regression_model(
            df=df,
            instance_number=instance_number,
            degree=degree,
            ground_truth_importances=ground_truth_importances_recode,
            aggregated_metrics=aggregated_data[degree]['metrics_aggregated'],  # Pass LR metrics
            aggregated_coefficients=aggregated_data[degree]['lr_coefficients']      # Pass lr_coefficients list
        )
        if not median_coefficients_lr.empty:
            aggregated_data[degree]['lr_coefficients'].append(median_coefficients_lr)
            all_lr_coefficients.append(median_coefficients_lr)
            # Append R² score to 'r2_scores'
            aggregated_data[degree]['r2_scores']['LR'].append(average_r2_lr)

    # After processing all files, aggregate metrics per degree
    logger.info("Aggregating metrics across all models and degrees.")
    print("Aggregating metrics across all models and degrees.")

    aggregated_results_all = []

    for degree in degrees:
        # Aggregate metrics for the current degree
        aggregated_results_degree = aggregate_metrics(aggregated_data[degree]['metrics_aggregated'], degree)
        aggregated_results_all.extend(aggregated_results_degree)

    # Convert aggregated_results_all to a DataFrame for easier handling
    metrics_df = pd.DataFrame(aggregated_results_all)

    # Save aggregated metrics to CSV files per degree
    save_aggregated_metrics_to_csv(metrics_df, args.output_dir)

    # Save R² scores in separate CSV files per degree
    logger.info("Aggregating R² scores and saving to separate CSV files per degree.")
    for degree in degrees:
        if not aggregated_data[degree]['r2_scores']['RF'] and not aggregated_data[degree]['r2_scores']['LR']:
            logger.warning(f"No R² scores collected for Degree {degree}. Skipping.")
            print(f"No R² scores collected for Degree {degree}.")
            continue
        r2_df = pd.DataFrame({
            'RF_R2': aggregated_data[degree]['r2_scores']['RF'],
            'LR_R2': aggregated_data[degree]['r2_scores']['LR']
        })
        r2_csv_path = os.path.join(args.output_dir, f'r2_scores_{degree}.csv')
        r2_df.to_csv(r2_csv_path, index=False)
        logger.info(f"R² scores for Degree {degree} saved to {r2_csv_path}.")
        print(f"R² scores for Degree {degree} saved to {r2_csv_path}.")

    # Generate Aggregated Metrics Plots
    logger.info("Generating plots for aggregated feature importances and coefficients.")
    # Aggregate RF Importances
    if all_rf_importances:
        aggregated_rf_importances = aggregate_metrics_series(
            all_rf_importances,
            feature_names=df.drop(['WalshFitness'], axis=1).columns.tolist()
        )
        plot_path_rf = os.path.join(args.output_dir, 'RF_aggregated_importances.png')
        plot_importances(aggregated_rf_importances, 'RF', plot_path_rf, top_n=10, logger=logger)

    # Aggregate LR Coefficients
    if all_lr_coefficients:
        aggregated_lr_coefficients = aggregate_metrics_series(
            all_lr_coefficients,
            feature_names=df.drop(['WalshFitness'], axis=1).columns.tolist()
        )
        plot_path_lr = os.path.join(args.output_dir, 'LR_aggregated_coefficients.png')
        plot_importances(aggregated_lr_coefficients, 'LR', plot_path_lr, top_n=10, logger=logger)

    # Optionally, plot Top-10 Features for Each Model
    if all_rf_importances:
        top10_rf = aggregated_rf_importances.sort_values(ascending=False).head(10)
        plot_path_rf_top10 = os.path.join(args.output_dir, 'RF_top10_importances.png')
        plot_importances(top10_rf, 'RF', plot_path_rf_top10, top_n=10, logger=logger)
    if all_lr_coefficients:
        top10_lr = aggregated_lr_coefficients.sort_values(ascending=False).head(10)
        plot_path_lr_top10 = os.path.join(args.output_dir, 'LR_top10_coefficients.png')
        plot_importances(top10_lr, 'LR', plot_path_lr_top10, top_n=10, logger=logger)


    # ----------------------------
    # 9. Saving Aggregated Coefficients to DataFrames and CSV
    # ----------------------------
    logger.info("Saving aggregated coefficients to DataFrames and CSV files.")
    print("Saving aggregated coefficients to DataFrames and CSV files.")

    # Save RF Aggregated Importances
    if all_rf_importances:
        rf_coefficients_df = aggregated_rf_importances.to_frame(name='Median_Importance')
        rf_csv_path = os.path.join(args.output_dir, 'RF_aggregated_coefficients.csv')
        rf_coefficients_df.to_csv(rf_csv_path)
        logger.info(f"RF aggregated coefficients saved to {rf_csv_path}.")
        print(f"RF aggregated coefficients saved to {rf_csv_path}.")
    else:
        logger.warning("No RF importances available to save.")
        print("No RF importances available to save.")

    # Save LR Aggregated Coefficients
    if all_lr_coefficients:
        lr_coefficients_df = aggregated_lr_coefficients.to_frame(name='Median_Coefficient')
        lr_csv_path = os.path.join(args.output_dir, 'LR_aggregated_coefficients.csv')
        lr_coefficients_df.to_csv(lr_csv_path)
        logger.info(f"LR aggregated coefficients saved to {lr_csv_path}.")
        print(f"LR aggregated coefficients saved to {lr_csv_path}.")
    else:
        logger.warning("No LR coefficients available to save.")
        print("No LR coefficients available to save.")

    # Logging the total number of aggregated entries
    total_rf_metrics = sum(len(aggregated_data[degree]['metrics_aggregated']['RF']['ImportanceRatio']) for degree in degrees)
    total_lr_metrics = sum(len(aggregated_data[degree]['metrics_aggregated']['LR']['ImportanceRatio']) for degree in degrees)
    logger.info(f"Total RF Metrics collected: {total_rf_metrics}")
    logger.info(f"Total LR Metrics collected: {total_lr_metrics}")
    print(f"Total RF Metrics collected: {total_rf_metrics}")
    print(f"Total LR Metrics collected: {total_lr_metrics}")

    logger.info("Feature Importance and Coefficient Aggregation Completed Successfully.")
    print("Feature Importance and Coefficient Aggregation Completed Successfully.")

if __name__ == "__main__":
    main()
