import os
import sys
import pandas as pd

from model_training import binary_combinations
sys.path.insert(0, '../../generator')
from walsh_expansion import WalshExpansion

def generate_and_save_combinations(degree, instance_id_start, instance_id_end, valid_instances_per_degree):
    """
    Generate all binary combinations for a given degree and save them to CSV files.

    Parameters:
    - degree: str, either 'deg2' or 'deg10'.
    - instance_id_start: int, starting instance number for the degree.
    - instance_id_end: int, ending instance number for the degree.
    - valid_instances_per_degree: int, number of valid instances to generate per degree.

    Returns:
    - None
    """
    bin_dir = f'puboi/model_data'
    os.makedirs(bin_dir, exist_ok=True)
    for instance_number in range(instance_id_start, instance_id_end):
        for valid_count in range(1, valid_instances_per_degree):
            # Construct the input file path based on degree
            input_name = f'../../../instances/small/puboi_1000seed_{degree}/puboi_{instance_number}.json'
            
            # Check if the input file exists
            if not os.path.exists(input_name):
                #print(f"Input file {input_name} does not exist. Skipping.")
                continue
            
            # Initialize WalshExpansion and load the instance
            f = WalshExpansion()
            f.load(input_name)
            print(f"Processing Instance {degree.upper()} - ID: {instance_number}, Valid Count: {valid_count}")
            
            # Generate all binary combinations of length 14 and compute fitness
            df = binary_combinations(f, 14)

            # Define the output CSV path based on degree
            output_csv = f'{bin_dir}/puboi_{degree}_{instance_number}.csv'
            df.to_csv(output_csv, index=False)
            print(f"Saved combinations to {output_csv}")

if __name__ == "__main__":
    # Set display option to suppress scientific notation globally
    pd.set_option('display.float_format', '{:.3f}'.format)
    
    # Define degrees to process
    degrees = {
        'deg2': {
            'instance_id_start': 1000,
            'instance_id_end': 1030  # Adjust based on your data
        },
        'deg10': {
            'instance_id_start': 1050,
            'instance_id_end': 1080  # Adjust based on your data
        }
    }
    
    # Define the number of valid instances to generate per degree
    valid_instances_per_degree = 30
    
    # Iterate over each degree and generate combinations
    for degree, params in degrees.items():
        print(f"\nStarting processing for {degree.upper()} instances...")
        generate_and_save_combinations(
            degree=degree,
            instance_id_start=params['instance_id_start'],
            instance_id_end=params['instance_id_end'],
            valid_instances_per_degree=valid_instances_per_degree
        )
        print(f"Completed processing for {degree.upper()} instances.\n")
    
    print("All binary combinations have been generated and saved.")
    # make sure to fined tune allthe model training, as this will now need to work with 60 instances, reduce RF param to 50, and neurons in MLP to 50 too to reduce computaiton