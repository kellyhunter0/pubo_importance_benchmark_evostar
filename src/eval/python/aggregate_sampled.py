import pandas as pd
import glob
import os
import argparse
import sys

def read_all_files(base_dir, instance_start, instance_end, run_file_pattern):
    """
    Reads all CSV files matching the run_file_pattern within the specified instance directories.
    
    Parameters:
        base_dir (str): The base directory containing instance subdirectories.
        instance_start (int): Starting instance number.
        instance_end (int): Ending instance number (inclusive).
        run_file_pattern (str): Pattern to match run files.
        
    Returns:
        pd.DataFrame: Concatenated DataFrame containing all data.
    """
    df_list = []
    
    for instance in range(instance_start, instance_end + 1):
        instance_dir = os.path.join(base_dir, str(instance))
        
        if not os.path.isdir(instance_dir):
            print(f"Warning: Directory {instance_dir} does not exist. Skipping.")
            continue

        pattern = os.path.join(instance_dir, run_file_pattern)
        files = glob.glob(pattern)
        
        if not files:
            print(f"Warning: No files matching {run_file_pattern} found in {instance_dir}.")
            continue

        for file in files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                print(f"Successfully read {file}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
    
    if not df_list:
        print("No data files found. Please check the directory paths and file patterns.")
        return pd.DataFrame()  # Return empty DataFrame
    
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

def clean_data(df):
    """
    Cleans the DataFrame by ensuring 'Value' is numeric and dropping rows with NaN in 'Value'.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to clean.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    before_drop = df.shape[0]
    df.dropna(subset=['Value'], inplace=True)
    after_drop = df.shape[0]
    dropped = before_drop - after_drop
    if dropped > 0:
        print(f"Dropped {dropped} rows due to non-numeric 'Value' entries.")
    return df

def aggregate_statistics(df, group_cols):
    """
    Aggregates statistics for specified grouping columns.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to aggregate.
        group_cols (list): List of columns to group by.
        
    Returns:
        pd.DataFrame: Aggregated DataFrame.
    """
    grouped = df.groupby(group_cols)['Value']
    aggregated_df = grouped.agg(
        Mean='mean',
        Median='median',
        StandardDeviation='std',
        InterquartileRange=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).reset_index()
    
    # Round statistics for readability
    aggregated_df['Mean'] = aggregated_df['Mean'].round(4)
    aggregated_df['Median'] = aggregated_df['Median'].round(4)
    aggregated_df['StandardDeviation'] = aggregated_df['StandardDeviation'].round(4)
    aggregated_df['InterquartileRange'] = aggregated_df['InterquartileRange'].round(4)
    
    return aggregated_df

def save_aggregated_df(aggregated_df, output_file):
    """
    Saves the aggregated DataFrame to a CSV file.
    
    Parameters:
        aggregated_df (pd.DataFrame): The DataFrame to save.
        output_file (str): Path to save the CSV.
    """
    try:
        aggregated_df.to_csv(output_file, index=False)
        print(f"Aggregated statistics saved to {output_file}")
    except Exception as e:
        print(f"Error saving aggregated statistics to {output_file}: {e}")

def main():
    # -----------------------------
    # Argument Parsing
    # -----------------------------
    parser = argparse.ArgumentParser(description="Aggregate raw metrics across models and explainers.")
    parser.add_argument('--base_dir', type=str, required=True,
                        help="Path to the 'ils-split' directory.")
    parser.add_argument('--instance_start', type=int, default=1000,
                        help="Starting instance directory number (default: 1000).")
    parser.add_argument('--instance_end', type=int, default=1029,
                        help="Ending instance directory number (default: 1029).")
    parser.add_argument('--run_file_pattern', type=str, default='Run*-raw_metrics.csv',
                        help="Pattern to match run files (default: 'Run*-raw_metrics.csv').")
    parser.add_argument('--aggregate_by', type=str, nargs='+', choices=['model', 'model_explainer', 'explainer'],
                        default=['model', 'model_explainer', 'explainer'],
                        help="Specify which aggregations to perform. Choices: 'model', 'model_explainer', 'explainer'. Default: all.")
    parser.add_argument('--output_suffix', type=str, default='',
                        help="Suffix to append to output filenames for differentiation.")
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    instance_start = args.instance_start
    instance_end = args.instance_end
    run_file_pattern = args.run_file_pattern
    aggregations = args.aggregate_by
    output_suffix = args.output_suffix
    
    # -----------------------------
    # Read and Combine Data
    # -----------------------------
    print("Reading and combining data...")
    combined_df = read_all_files(base_dir, instance_start, instance_end, run_file_pattern)
    
    if combined_df.empty:
        print("No data to process. Exiting.")
        sys.exit()
    
    # -----------------------------
    # Data Cleaning
    # -----------------------------
    print("\nCleaning data...")
    cleaned_df = clean_data(combined_df)
    
    # -----------------------------
    # Perform Aggregations
    # -----------------------------
    for agg in aggregations:
        if agg == 'model':
            print("\nAggregating by Model...")
            aggregated_df = aggregate_statistics(cleaned_df, ['Model'])
            output_file = os.path.join(base_dir, f'aggregated_by_model{output_suffix}.csv')
            save_aggregated_df(aggregated_df, output_file)
        
        elif agg == 'model_explainer':
            print("\nAggregating by Model and Explainer...")
            aggregated_df = aggregate_statistics(cleaned_df, ['Model', 'Explainer'])
            output_file = os.path.join(base_dir, f'aggregated_by_model_explainer{output_suffix}.csv')
            save_aggregated_df(aggregated_df, output_file)
        
        elif agg == 'explainer':
            print("\nAggregating by Explainer...")
            aggregated_df = aggregate_statistics(cleaned_df, ['Explainer'])
            output_file = os.path.join(base_dir, f'aggregated_by_explainer{output_suffix}.csv')
            save_aggregated_df(aggregated_df, output_file)
    
    # -----------------------------
    # Summary
    # -----------------------------
    print("\nAggregation complete. Summary:")
    if 'model' in aggregations:
        print("\nAggregated by Model:")
        print(aggregated_df.head())
    if 'model_explainer' in aggregations:
        print("\nAggregated by Model and Explainer:")
        print(aggregated_df.head())
    if 'explainer' in aggregations:
        print("\nAggregated by Explainer:")
        print(aggregated_df.head())

if __name__ == "__main__":
    main()
