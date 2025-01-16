import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import argparse
import os
import sys
from matplotlib import rcParams
import seaborn as sns
# ----------------------------
# 1. Configure Plot Aesthetics
# ----------------------------

# Apply a better theme/style before updating rcParams
plt.style.use('seaborn-colorblind')  # You can choose other styles like 'ggplot', 'fivethirtyeight', etc.

# Update rcParams to adjust font sizes and other parameters
rcParams.update({
    'font.size': 130,            # Default font size for all text elements
    'axes.titlesize': 130,       # Font size for the axes title
    'axes.labelsize': 130,       # Font size for the x and y labels
    'xtick.labelsize': 130,      # Font size for x tick labels
    'ytick.labelsize': 130,      # Font size for y tick labels
    'legend.fontsize': 130,      # Font size for the legend
    'figure.titlesize': 130,     # Font size for the figure title
    'lines.linewidth': 30,       # Default line width
    'boxplot.boxprops.linewidth': 10,       # Box line width
    'boxplot.whiskerprops.linewidth': 20,   # Whisker line width
    'boxplot.capprops.linewidth': 30,       # Cap line width
    'boxplot.medianprops.linewidth': 30,    # Median line width
    'boxplot.flierprops.markersize': 20,   # Flier (outlier) marker size
    'boxplot.flierprops.linewidth': 20,     # Flier (outlier) marker edge width
    'boxplot.flierprops.markeredgewidth': 20,
})

def plot_r2_scores(
    csv_filename: str,
    threshold: float = 0.00,
    output_filename: str = None,
    stats_output_filename: str = None,
    figsize_per_model: tuple = (6, 6),
    show_plot: bool = True
):
    """
    Generates boxplots of R² scores for different models from a CSV file.

    Parameters:
    - csv_filename (str): Path to the CSV file containing R² scores. Each column represents a model.
    - threshold (float, optional): Threshold to filter R² scores. Defaults to 0.98.
    - output_filename (str, optional): If provided, saves the plot to the specified file. Defaults to None.
    - figsize_per_model (tuple, optional): Size of each subplot (width, height). Defaults to (6, 6).
    - show_plot (bool, optional): Whether to display the plot. Set to False if saving the plot. Defaults to True.
    """

    # Check if the file exists
    if not os.path.isfile(csv_filename):
        print(f"Error: The file '{csv_filename}' does not exist.")
        sys.exit(1)

    # Read data
    try:
        df = pd.read_csv(csv_filename)
    except Exception as e:
        print(f"Error reading '{csv_filename}': {e}")
        sys.exit(1)

    if df.empty:
        print(f"Error: The file '{csv_filename}' is empty.")
        sys.exit(1)

    # Define threshold
    # threshold = 0.98  # Now an argument
    # Access the colorblind palette
    colorblind_palette = sns.color_palette("colorblind")
    colors = colorblind_palette.as_hex()
    print(colors)

    # For consistency, create a color cycle iterator
    color_cycle = iter(colorblind_palette)
    # List of models
    models = df.columns.tolist()

    summary_stats = []
    # Number of models
    num_models = len(models)

    # Create subplots - one for each model
    fig, axes = plt.subplots(
        1,
        num_models,
        figsize=(figsize_per_model[0] * num_models, figsize_per_model[1]),
        sharex=False,
        sharey=False
    )

    # If there's only one model, axes is not a list, so we make it a list
    if num_models == 1:
        axes = [axes]
    #color = next(color_cycle)
    # Iterate over each model and its corresponding axis
    for ax, model in zip(axes, models):
        model_data = df[model]
        # Filter values above threshold
        filtered_data = model_data[model_data >= threshold]

        if not filtered_data.empty:
            # Calculate mean and standard deviation
            # Calculate summary statistics
            mean_value = filtered_data.mean()
            median_value = filtered_data.median()
            std_dev = filtered_data.std()
            iqr = filtered_data.quantile(0.75) - filtered_data.quantile(0.25)
           # color = next(color_cycle)
            # Plot boxplot for the model with enhanced aesthetics

            #0173b2 - sample

            # Append statistics to the list
            summary_stats.append({
                'Model': model,
                'Mean': mean_value,
                'Median': median_value,
                'Standard Deviation': std_dev,
                'IQR': iqr
            })
            bp = ax.boxplot(
                filtered_data,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=colors[0], color=colors[0], linewidth=10),  # Thicker box lines
                medianprops=dict(linewidth=20, color='black'),                      # Thicker median lines
                whiskerprops=dict(linewidth=10, color='black'),                    # Thicker whiskers
                capprops=dict(linewidth=10),                        # Thicker caps
                flierprops=dict(marker='D', markersize=15, linewidth=15, markeredgecolor='black',linestyle='none', color='black'),      # Enhanced outliers
                showfliers=True                                                    # Show outliers
            )
            #color = next(color_cycle)
            # Overlay mean value
            ax.plot(1, mean_value, marker='o', label='Mean', color='black', markersize=40, markeredgewidth=20)

            # Add error bar for standard deviation
            ax.errorbar(
                1,
                mean_value,
                yerr=std_dev,
                fmt='none',
                ecolor='black',
                capsize=30,
                linewidth=25,
                label='Std Dev',
              #  fontsize=80
            )

            # Customize y-axis limits based on data
            data_min = filtered_data.min()
            data_max = filtered_data.max()
            data_range = data_max - data_min

            if data_range == 0:
                y_min = data_min - 0.01  # Small value to avoid zero range
                y_max = data_max + 0.01
            else:
                padding = data_range * 0.1  # 10% padding
                y_min = max(0, data_min - padding)
                y_max = min(1, data_max + padding)
            ax.set_ylim(0.94, 1)

            # Set title and labels
            ax.set_title(f'{model} Performance', fontsize=130)
            ax.set_ylabel('R² Score', fontsize=130)

            # Adjust y-axis formatter to avoid scientific notation if necessary
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)

            # Add legend
            ax.legend(loc='upper left', fontsize=60)
        else:
            # Plot an 'X' to indicate poor performance
            ax.text(
                0.5,
                0.5,
                'X',
                color='red',
                fontsize=50,
                ha='center',
                va='center',
                transform=ax.transAxes
            )
            ax.set_ylim(0, 1)  # Set a default y-axis range
            # Set title and labels
            ax.set_title(f'{model} Performance', fontsize=50)
            ax.set_ylabel('R² Score', fontsize=40)

        # Remove x-axis ticks
        ax.set_xticks([])

    # Convert summary statistics list to DataFrame
    summary_df = pd.DataFrame(summary_stats)

    # Save summary statistics to CSV if filename is provided
    if stats_output_filename:
        try:
            summary_df.to_csv(stats_output_filename, index=False)
            print(f"Summary statistics saved as '{stats_output_filename}'.")
        except Exception as e:
            print(f"Error saving summary statistics: {e}")
    # Adjust layout with space between subplots
    plt.tight_layout(pad=5.0)

    # Save the plot if output filename is provided
    if output_filename:
        try:
            plt.savefig(output_filename, bbox_inches='tight', dpi=300)
            print(f"Plot saved as '{output_filename}'.")
        except Exception as e:
            print(f"Error saving plot: {e}")

    # Display the plot if required
    if show_plot:
        plt.show()
    else:
        plt.close()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate boxplots of R² scores for different models from a CSV file."
    )
    parser.add_argument(
        'csv_filename',
        type=str,
        help='Path to the CSV file containing R² scores. Each column represents a model.'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.98,
        help='Threshold to filter R² scores. Default is 0.98.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Filename to save the plot (e.g., plot.png). If not provided, the plot will not be saved.'
    )
    parser.add_argument(
        '--stats_output',
        type=str,
        default=None,
        help='Filename to save the summary statistics (e.g., stats.csv). If not provided, statistics will not be saved.'
    )
    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=(6, 6),
        help='Figure size per model (width height). Default is 6 6.'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot. Useful when saving the plot without displaying it.'
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the plotting function with parsed arguments
    plot_r2_scores(
        csv_filename=args.csv_filename,
        threshold=args.threshold,
        output_filename=args.output,
        stats_output_filename=args.stats_output,  # Pass the stats output filename
        figsize_per_model=tuple(args.figsize),
        show_plot=not args.no_show
    )

if __name__ == "__main__":
    main()
