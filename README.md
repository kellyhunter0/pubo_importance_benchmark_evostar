# PUBOi: a tunable benchmark with variable importance (original work by Verel, Tari and Omidvar) - 2022

  Sara Tari, Sebastien Verel, and Mahmoud Omidvar. 
  "PUBOi: A Tunable Benchmark with Variable Importance." 
  In European Conference on Evolutionary Computation in Combinatorial Optimization (Part of EvoStar), pp. 175-190. Springer, Cham, 2022.

This benchmark is a set of instances from the problem: Polynomial Unconstrained Binary Optimization (PUBO) with variable importance. 





# Into the Black Box: Mining Variable Importance with XAI - 2025
Authors: Kelly Hunter, Sarah L. Thomson, Emma Hart

This work is in association with paper submitted to EvoStar 2025's special joint track by EvoAPPS and EuroGP on Evolutionary Machine Learning on 15/11/24 and accepted on 10/01/25. The present work utilises the PUBOi benchmark to conduct experiments to uncover if XAI can learn ground truth importances in important variables when we know these ahead of time. We conduct this in two setups: ENUM and SAMPLE where we compare learning algorithms, XAI methods and sample sizes used to train the models to better understand which techniques are promising in this context. 

# Full data and instances used for the SAMPLE and ENUM experiment
- This can be found at a [Zenodo Repository](https://zenodo.org/records/14664843)


# EvoStar Paper

## To Run the Code

### 1. Generate Instances
- Start by generating instances with the **`parameters_generator_evoCOP22.R`** file, located within the **`/instances`** directory.
- The generated instances will save in **`/instances/small`**. For experiments, we investigate degree 2 and 10 for importance across known variables but include only degree 2 since results for both degrees were the same (future work will investigate this further).
- Adjust parameters as needed for your experiments:
  - We used:
    - Random seed: `1000`
    - Number of instances: `50`
  - Instances were checked to ensure they are valid, containing important variables.
  - Valid instance check:
    - Navigate to the instances directory and run the following command in the terminal:
      ```bash
      python ../src/generator/make_instances.py -I2 small/puboi_param_1000seed_deg2.csv -D2 small/testd2 -I10 small/puboi_param_1000seed_deg10.csv -D10 small/testd10
      ```

### 2. Run Experiments
#### a. Generate Data
- Generate the full space for degree 2 and 10:
  - **`generate-full-space.py`**
- Generate the partial space for degree 2 and 10:
  - **`minemodelstest.py`**

#### b. Run Experiments
- Once the data is generated:
  - **Sampled Space Experiments:**
    - Navigate to the **`executed-separately-ils.py`** file.
  - **Full Space Experiments:**
    - Navigate to the **`model-training.py`** file.

#### c. Aggregate Data
- After experiments finish, aggregate the data:
  - Use **`aggregate_sampled.py`** for both the enum and sample data:
    ```bash
    python aggregate_sampled.py --base_dir puboi/deg2 --instance_start 1000 --instance_end 1029 --run_file_pattern raw_metrics.csv --aggregate_by model_explainer --output_suffix ME
    python aggregate_sampled.py --base_dir ils-split/deg2 --instance_start 1000 --instance_end 1029 --run_file_pattern Run*-raw_metrics.csv --aggregate_by model_explainer --output_suffix ME
    ```

#### d. Generate Visualisations
1. **Boxplots**:
   - Use **`boxplots.py`** with the R² scores for:
     - SAMPLE: `src/eval/python/ils-split/deg2/aggregated_heatmaps/r2_scores_median.csv`
       ```bash
        python boxplots.py ils-split\deg2\aggregated_heatmaps\r2_scores_median.csv --threshold 0.94 --output deg2ils-230125.png --stats_output d2-ils-stats-230125.csv --figsize (40,20)
       ```
     - ENUM: `src/eval/python/puboi/deg2/aggregated_heatmaps/r2_scores_median.csv`
       ```bash
        python boxplots.py puboi\deg2\aggregated_heatmaps\r2_scores_median.csv --threshold 0.94 --output deg2enum-230125.png --stats_output d2-enum-stats-230125.csv --figsize (40,20)
       ```

2. **Heatmaps**:
   - Plot the heatmaps for full model training using **`heatmaps.py`**:
     ```bash
     python generate_heatmaps.py \
       puboi/deg2/aggregated_heatmaps/shap_aggregated.csv \
       puboi/deg2/aggregated_heatmaps/lime_aggregated.csv \
       puboi/deg2/aggregated_heatmaps/pfi_aggregated.csv \
       ils-split/deg2/aggregated_heatmaps/shap_aggregated.csv \
       ils-split/deg2/aggregated_heatmaps/lime_aggregated.csv \
       ils-split/deg2/aggregated_heatmaps/pfi_aggregated.csv
     ```

3. **Coefficient Heatmaps**:
   - Generate internal feature importance scores with **`heatmaps_coefficients.py`**:
     ```bash
     python heatmaps.py \
       puboi/aggregated_results/lr_aggregated_coefficients.csv \
       puboi/aggregated_results/rf_aggregated_coefficients.csv
     ```
