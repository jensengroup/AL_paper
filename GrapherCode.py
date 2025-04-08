import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import re
from matplotlib import cm
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import glob
import matplotlib

# Set this variable to either 'std_dev' or 'std_error' to control which metric is used globally.
ERROR_TYPE = 'std_error'  # 'std_dev' or 'std_error'

DARK_MODE = False

if DARK_MODE:
    plt.rcParams['figure.facecolor'] = 'grey'   # Background of the figure
    plt.rcParams['axes.facecolor'] = 'darkgrey'  # Background of the plot (axes)
    plt.rcParams['axes.edgecolor'] = 'black'     # Edge color of the plot
    plt.rcParams['grid.color'] = 'white'         # Grid line color
    plt.rcParams['axes.labelcolor'] = 'black'    # Label color
    plt.rcParams['xtick.color'] = 'black'        # X-tick color
    plt.rcParams['ytick.color'] = 'black'        # Y-tick color
    plt.rcParams['legend.edgecolor'] = 'black'   # Legend edge color
else:
    plt.rcParams['figure.facecolor'] = 'white'   # Background of the figure
    plt.rcParams['axes.facecolor'] = 'white'     # Background of the plot (axes)
    plt.rcParams['axes.edgecolor'] = 'black'     # Edge color of the plot
    plt.rcParams['grid.color'] = 'black'         # Grid line color
    plt.rcParams['axes.labelcolor'] = 'black'    # Label color
    plt.rcParams['xtick.color'] = 'black'        # X-tick color
    plt.rcParams['ytick.color'] = 'black'        # Y-tick color
    plt.rcParams['legend.edgecolor'] = 'black'   # Legend edge color


def get_files(file_path, recursive=False):
    if recursive:
        files = sorted(glob.glob(file_path + '/**/*.csv', recursive=recursive))
    else:
        files = sorted(glob.glob(file_path + '/*.csv'))

    return files


def set_dynamic_xticks(ax, max_iterations, desired_ticks=10):
    """
    Dynamically set x-ticks for a given axis.

    Parameters:
    - ax: The axis on which to set the x-ticks.
    - max_iterations: The maximum number of iterations/data points on the x-axis.
    - desired_ticks: The desired maximum number of tick marks on the x-axis (default is 10).
    """
    tick_interval = max_iterations / desired_ticks
    tick_interval = math.ceil(tick_interval)
    ax.set_xticks(np.arange(0, max_iterations + 1, tick_interval))


def format_title(filename):
    # Remove the prefix digits and underscore, and strip off the '.csv' extension
    clean_name = filename.split('_')[1].replace('.csv', '')
    
    # Split the name into model and descriptor parts
    model, descriptor = clean_name.rsplit('-', 1)
    
    # Return the formatted title
    return f'{model}({descriptor})'


def get_data(files):

    data_list = {}
    for file in files:
        data = pd.read_csv(file)
        columns_after_rank = data.columns[3:]
        name = file.split('\\')[-1].split('.')[0]
        for column in columns_after_rank:
            multiplier = int(re.search(r'\d+', columns_after_rank[0]).group())
            data[column] *= multiplier
            pivot_data = data.pivot(index='rank', columns='replicate', values=column)
            pivot_data['Avg.'] = pivot_data.mean(axis=1)
            pivot_data['Std. dev.'] = pivot_data.std(axis=1)
            replicates = pivot_data.shape[1] - 1
            pivot_data['Std. error'] = pivot_data['Std. dev.'] / (replicates ** 0.5)
            data_list[name] = pivot_data
    return data_list


def plot_all_graphs(data, graphs_per_row=4):
    """
    Plots graphs for each dataset stored in the data dictionary.

    Parameters:
    - data: Dictionary of dataframes where keys are dataset names.
    - graphs_per_row: Number of graphs to display in each row.
    """
    rows_for_files = int(np.ceil(len(data) / graphs_per_row))
    fig, axes = plt.subplots(rows_for_files, graphs_per_row, figsize=(15, 5 * rows_for_files), squeeze=False)

    error_col = 'Std. dev.' if ERROR_TYPE == 'std_dev' else 'Std. error'

    for i, (dataset_name, pivot_data) in enumerate(data.items()):
        # Determine subplot location
        row, col = divmod(i, graphs_per_row)
        ax = axes[row, col]

        # Plotting logic for individual runs
        for col_name in pivot_data.columns[:-2]:  # Ignore 'Avg.', 'Std. dev.', and 'Std. error'
            if col_name not in ['Avg.', 'Std. dev.', 'Std. error']:
                ax.plot(pivot_data.index, pivot_data[col_name], alpha=0.1, marker='o', linestyle='--', label='Individual runs' if col_name == pivot_data.columns[0] else "")

        # Plot average
        ax.plot(pivot_data.index, pivot_data['Avg.'], label='Avg.', color='black', linestyle='-', linewidth=2, marker='s')
        # Fill with selected error metric
        ax.fill_between(pivot_data.index, pivot_data['Avg.'] - pivot_data[error_col], pivot_data['Avg.'] + pivot_data[error_col], color='gray', alpha=0.2, label=error_col.capitalize())

        ax.set_title(dataset_name, fontsize=12)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of top 1% acquired')
        ax.set_ylim(0, 100)
        set_dynamic_xticks(ax, int(pivot_data.index.max()))

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
        if i == len(data) - 1:  # Add legend only to the last plot
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to make room for plot labels and titles
    plt.tight_layout()
    plt.show()


def plot_compact_graphs(
    data,
    graphs_per_row=4,
    dpi=300,
    title_size=10,
    label_size=9,
    tick_size=8,
    legend_size=7,
    ylim=(-3,60),
    alpha_of_runs=0.2,
    dont_sort=True,
):
    error_col = 'Std. dev.' if ERROR_TYPE == 'std_dev' else 'Std. error'
    # Exclude the unwanted dataset
    data_to_plot = {k: v for k, v in data.items() if k != 'Morgan Tanimoto/Tanimoto (Morgan)'}

    total_plots = len(data_to_plot)
    rows_for_files = int(np.ceil(total_plots / graphs_per_row))
    fig, axes = plt.subplots(rows_for_files, graphs_per_row, figsize=(10, 2.5 * rows_for_files), dpi=dpi, squeeze=False)
    
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    data_items = sorted(data_to_plot.items()) if dont_sort else data_to_plot.items()
    for i, (dataset_name, pivot_data) in enumerate(data_items):
        ax = axes[i]

        if graphs_per_row > 1: # if there are a grid of graphs, annotate each graph with a letter.
            label = chr(65 + i)
            ax.annotate(
                label,
                xy=(0, 1), xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize='medium', verticalalignment='top',
                # bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0)
            )

        row, col = divmod(i, graphs_per_row)  # Calculate row and column index for the subplot

        for col_name in pivot_data.columns:
            if col_name not in ['Avg.', 'Std. dev.', 'Std. error']:
                ax.plot(pivot_data.index, pivot_data[col_name], alpha=alpha_of_runs, marker='o', linestyle='--', label='Individual runs' if col_name == pivot_data.columns[0] else "")

        ax.plot(pivot_data.index, pivot_data['Avg.'], label='Avg.', color='black', linestyle='-', linewidth=1.5, marker='s')
        ax.fill_between(pivot_data.index, pivot_data['Avg.'] - pivot_data[error_col], pivot_data['Avg.'] + pivot_data[error_col], color='gray', alpha=0.3, label=error_col.capitalize())

        if '/' in dataset_name:
            no_of_splits = dataset_name.count('/')
            title_name = dataset_name.split('/')[no_of_splits]
        else:
            title_name = dataset_name
        ax.set_title(title_name, fontsize=title_size)

        # Only set the xlabel if the subplot is in the last row
        if row == (rows_for_files - 1):  
            ax.set_xlabel('Iteration', fontsize=label_size)

        ax.set_ylim(ylim)
        set_dynamic_xticks(ax, int(pivot_data.index.max()))
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')
        if i == total_plots - 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_size)

    if graphs_per_row > 1: # if there are a grid of graph, make a common y-axis title.
        for j in range(total_plots, rows_for_files * graphs_per_row):
            axes[j].axis('off')

        # Set a common y-label for the entire figure
        fig.text(0.04, 0.5, 'Number of top 1% acquired', va='center', rotation='vertical', fontsize=label_size)
    else: # if there is only one graph, set the y-axis label to the graph.
        axes[0].set_ylabel('Number of top 1% acquired', fontsize=label_size)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.3, hspace=0.45)
    return fig


def plot_single_graph(
    data,
    ax=None,
    title="",
    title_size=12,
    label_size=10,
    tick_size=10,
    legend_size=10,
    ylim=(-2, 50),
    z_upper="EI"
):
    error_col = "Std. dev." if ERROR_TYPE == "std_dev" else "Std. error"

    # If no axis is provided, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed

    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "H", "+", "x"]
    line_styles = [
        "-",
        "--",
        "-.",
        ":",
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1)),
        (0, (5, 10)),
    ]
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(data)))

    for i, (dataset_name, pivot_data) in enumerate(data.items()):
        color = colors[i]
        marker = markers[i % len(markers)]
        line_style = line_styles[i % len(line_styles)]
        no_of_splits = len(dataset_name.split("/")) - 1
        if "/" in dataset_name:
            dataset_name = dataset_name.split("/")[no_of_splits]

        z = 10 if dataset_name == z_upper else i
        ax.plot(
            pivot_data.index,
            pivot_data["Avg."],
            marker=marker,
            label=dataset_name,
            linestyle=line_style,
            color=color,
            zorder=z
        )
        ax.fill_between(
            pivot_data.index,
            pivot_data["Avg."] - pivot_data[error_col],
            pivot_data["Avg."] + pivot_data[error_col],
            alpha=0.10,
            color=color,
            zorder=z
        )

        # Solid lines at the boundaries
        ax.plot(
            pivot_data.index,
            pivot_data["Avg."] - pivot_data[error_col],
            linestyle="-",
            color=color,
            alpha=0.05,
            zorder=z
        )
        ax.plot(
            pivot_data.index,
            pivot_data["Avg."] + pivot_data[error_col],
            linestyle="-",
            color=color,
            alpha=0.05,
            zorder=z
        )

        # Font size and decoration
        if title != "":
            ax.set_title(title, fontsize=title_size)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)

    # Add grid, labels, limits, and legend
    ax.grid(True, linestyle="--", linewidth=0.5, color="lightgrey")
    ax.set_xlabel("Iteration", fontsize=label_size)
    ax.set_ylabel("Number of top 1% acquired", fontsize=label_size)
    ax.set_xlim(0, 5.1)
    ax.set_ylim(ylim)
    ax.legend(fontsize=legend_size)

    return ax


def get_probability_data(files, thresholds=[1, 5, 20]):
    data_list = {}
    for file in files:
        data = pd.read_csv(file)
        columns_after_rank = data.columns[3:]  # Assuming 'top-XXXX acquired' is the 4th column
        name = file.split('\\')[-1].split('.')[0]
        no_of_splits = name.count('/')
        name = name.split('/')[no_of_splits]
        multiplier = int(re.search(r'\d+', columns_after_rank[0]).group())
        for column in columns_after_rank:
            data[column] *= multiplier  # Multiply to get actual counts
            pivot_data = data.pivot(index='rank', columns='replicate', values=column).round(6)
            replicates = pivot_data.shape[1]

            probability_df = pd.DataFrame(index=pivot_data.index)
            for threshold in thresholds:
                # Indicator function
                indicator = (pivot_data >= threshold).astype(int)
                # Sum over replicates and divide by N
                prob = indicator.sum(axis=1) / replicates
                probability_df[f'Prob >= {threshold}'] = prob
            data_list[name] = probability_df
    return data_list


def plot_probability_graphs(
    data,
    graphs_per_row=3,
    dpi=150,
    title_size=13,
    label_size=11,
    tick_size=11,
    legend_size=11,
    legend_on_plot=False,
    n=30,
    emph_data={},
    error_style="fill",
    ylim=(-0.05, 1.05),
    custom_ylabel=True,
    offset=0.25,
    annotate_pos=(0, 1),
    show_baseline=True,
):
    # Define markers and line styles
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'H', '+', 'x']
    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 10))]

    data_to_plot = data
    total_plots = len(data_to_plot)
    rows_for_files = int(np.ceil(total_plots / graphs_per_row))
    fig, axes = plt.subplots(rows_for_files, graphs_per_row, figsize=(12, 3 * rows_for_files), dpi=dpi, squeeze=False)

    colormap = get_cmap('tab10')  
    colors = [colormap(i % 10) for i in range(10)]
    axes = axes.flatten()  # Flatten the axes array for easier indexing
    z = 1.96  # for ~95% CI for ~95% confidence

    for i, (dataset_name, prob_df) in enumerate(sorted(data_to_plot.items())):
        if show_baseline:
            threshold = int(dataset_name.split(' ')[2])
            from math import comb
            def binomial_p_exact(n, k, p=0.01):
                """Probability of exactly k 'successes' in n draws with success probability p."""
                return comb(n, k) * (p**k) * ((1 - p)**(n - k))
            # Add random baseline to every plot

            def binomial_p_at_least(n, k, p=0.01):
                """Probability of k or more 'successes' out of n."""
                return 1 - sum(binomial_p_exact(n, x, p) for x in range(k))
            
            N_list = np.array([10,30,50,70,90,110])
            baseline_list =  [binomial_p_at_least(N, threshold, 0.01) if N >= threshold else 0 for N in N_list]
            prob_df['Random probability'] = baseline_list
            
        ax = axes[i]

        if graphs_per_row > 1: # if there are a grid of graphs, annotate each graph with a letter.
            label = chr(65 + i)
            ax.annotate(
                label,
                xy=annotate_pos, xycoords='axes fraction',
                xytext=(+0.5, -0.5), textcoords='offset fontsize',
                fontsize='medium', verticalalignment='top',
                # bbox=dict(''' facecolor='0.7' '''edgecolor='none', pad=3.0)
            )

        row, col = divmod(i, graphs_per_row)

        offset = offset  # Adjust the offset value as needed
        num_columns = len(prob_df.columns)

        for j, column in enumerate(prob_df.columns):
            if '/' in column:
                col_len = len(column.split('/')) - 1 
                column_name = column.split('/')[col_len]
            else:
                column_name = column

            p = prob_df[column].values  # probabilities at each iteration

            se = np.sqrt(p * (1 - p) / n)
            lower_ci = p - z * se
            upper_ci = p + z * se

            lower_ci = np.clip(lower_ci, 0, 1)
            upper_ci = np.clip(upper_ci, 0, 1)

            marker = markers[j % len(markers)]
            line_style = line_styles[j % len(line_styles)]
            color = colors[j % len(colors)]

            # Decide alpha based on emphasis
            if emph_data == {}:
                line_alpha = 1.0
                fill_alpha = 0.2
                line_width = 1.5
            elif column_name in emph_data:
                line_alpha = 1.0
                fill_alpha = 0.3
                line_width = 2.2
            else:
                line_alpha = 0.4
                fill_alpha = 0.2
                line_width = 1.5

            x_values = prob_df.index + (j - num_columns / 2) * offset / num_columns

            if error_style == 'fill':
                # Plot main line
                line = ax.plot(
                    prob_df.index,
                    p,
                    marker=marker,
                    linestyle=line_style,
                    color=color,
                    label=column_name,
                    alpha=line_alpha,
                    linewidth=line_width,
                )
                line_color = line[0].get_color()
                # Fill the confidence interval
                ax.fill_between(prob_df.index, lower_ci, upper_ci, color=line_color, alpha=fill_alpha)

            elif error_style == 'bar':
                # yerr is given as a 2D-like array [lower_errors, upper_errors]
                yerr = np.vstack((p - lower_ci, upper_ci - p))
                # Use separate marker and linestyle params
                ax.errorbar(
                    x_values,
                    p,
                    yerr=yerr,
                    marker=marker,
                    linestyle=line_style,
                    color=color,
                    label=column_name,
                    alpha=line_alpha,
                    linewidth=line_width,                    
                )

        title_name = "$\geq$ " + dataset_name.split(' ')[2]
        ax.set_title(title_name, fontsize=title_size)
        if row == (rows_for_files - 1):
            ax.set_xlabel('Iteration', fontsize=label_size)
        ax.set_ylim(ylim)

        max_iter = int(prob_df.index.max())
        ax.set_xticks(range(max_iter + 1))

        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')

        if i == total_plots - 1 and legend_on_plot == False:
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_size)
        elif legend_on_plot == True:
            ax.legend(loc='upper left', fontsize=legend_size)

        if custom_ylabel == False:
            ax.set_ylabel('Probability', fontsize=label_size)


    # Set a common y-label for the entire figure
    if custom_ylabel:
        fig.text(0.06, 0.5, 'Probability', va='center', rotation='vertical', fontsize=label_size)

    # Hide unused subplots
    for j in range(total_plots, rows_for_files * graphs_per_row):
        axes[j].axis('off')

    return fig


def reorganize_data(data):
    # Initialize a dictionary to hold the new data
    new_data = {}
    
    # Get the list of thresholds from one of the DataFrames
    sample_df = next(iter(data.values()))
    thresholds = sample_df.columns
    
    # Loop over each threshold
    for threshold in thresholds:
        # Create a DataFrame to hold data for this threshold
        threshold_df = pd.DataFrame()
        # Loop over each method and extract the corresponding threshold column
        for method_name, df in data.items():
            series = df[threshold]
            series.name = method_name
            threshold_df = pd.concat([threshold_df, series], axis=1)
            threshold_df = threshold_df[sorted(threshold_df.columns)]
        # Add the DataFrame to the new data dictionary
        new_data[threshold] = threshold_df
    return new_data

# def reorganize_data(data):
#     """
#     Reorganizes a nested dictionary of DataFrames by threshold.
    
#     The input `data` is assumed to be a dictionary where each key (e.g. a method name)
#     maps to another dictionary whose values are DataFrames (each containing columns
#     such as "Prob >= 1", "Prob >= 5", etc.). This function first flattens the nested
#     structure so that each DataFrame gets a unique key (using "outer/inner" naming)
#     and then creates a new dictionary where each threshold is a key and its value is 
#     a DataFrame with rows corresponding to 'rank' and columns corresponding to the method.
#     """
#     # Flatten the nested dictionary if needed.
#     # If the first value is a dict, assume that every value is a dict.
#     first_val = next(iter(data.values()))
#     if isinstance(first_val, dict):
#         flat_data = {}
#         for method_name, inner_dict in data.items():
#             for inner_key, df in inner_dict.items():
#                 flat_key = f"{method_name}/{inner_key}"
#                 flat_data[flat_key] = df
#         data = flat_data

#     # Now assume data is a flat dictionary mapping method names to DataFrames.
#     # Get the list of thresholds from one of the DataFrames.
#     sample_df = next(iter(data.values()))
#     thresholds = sample_df.columns  # e.g., ["Prob >= 1", "Prob >= 5", "Prob >= 20"]

#     new_data = {}
#     # For each threshold, collect the corresponding series from each DataFrame.
#     for threshold in thresholds:
#         threshold_df = pd.DataFrame()
#         for method_name, df in data.items():
#             # Extract the series corresponding to the current threshold.
#             series = df[threshold].copy()
#             series.name = method_name
#             threshold_df = pd.concat([threshold_df, series], axis=1)
#         # Optionally, sort the columns alphabetically.
#         threshold_df = threshold_df[sorted(threshold_df.columns)]
#         new_data[threshold] = threshold_df

#     return new_data


def table_data(files, thresholds=[1, 5, 20]):
    results = []
    
    for file in files:
        # Read the file
        data = pd.read_csv(file)
        
        # Identify relevant columns
        columns_after_rank = data.columns[3:]  # Adjust as needed
        name = file.split('\\')[-1].split('.')[0]
        no_of_splits = name.count('/')
        name = name.split('/')[no_of_splits]
        
        # Extract the multiplier from the column name
        multiplier = int(re.search(r'\d+', columns_after_rank[0]).group())
        
        for column in columns_after_rank:
            data[column] *= multiplier  # Multiply to get actual counts
            
            # Pivot the data
            pivot_data = data.pivot(index='rank', columns='replicate', values=column)
            replicates = pivot_data.shape[1]
            
            # Calculate statistics
            pivot_data['Avg.'] = pivot_data.mean(axis=1)
            pivot_data['Std. dev.'] = pivot_data.std(axis=1)
            
            # Calculate probabilities for each threshold using the provided formula
            probabilities = {}
            for threshold in thresholds:
                # Indicator function
                indicator = (pivot_data.iloc[:, :replicates] >= threshold).astype(int)
                # Sum over replicates and divide by N
                prob = indicator.sum(axis=1) / replicates
                pivot_data[f'Prob >= {threshold}'] = prob
                probabilities[f'Prob >= {threshold}'] = prob.iloc[-1]
            
            # Collect results for the last row
            avg = pivot_data.iloc[-1]['Avg.']
            std = pivot_data.iloc[-1]['Std. dev.']
            
            # Prepare the result dictionary
            result = {
                'Name': name,
                'Avg.': avg,
                'Std. dev.': std,
            }
            result.update(probabilities)
            results.append(result)
    
    # Create the final DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort probability columns numerically
    probability_cols = [col for col in result_df.columns if col.startswith('Prob >=')]
    probability_cols_sorted = sorted(probability_cols, key=lambda x: int(re.search(r'\d+', x).group()))
    
    # Reorder columns
    cols = ['Name', 'Avg.', 'Std. dev.'] + probability_cols_sorted
    result_df = result_df[cols]
    
    # Set 'Name' as index
    result_df.set_index('Name', inplace=True)
    
    return result_df.round(2)


def combine_plot_single_graph(plot_funcs, ncols=2):
    """
    Combine multiple single-graph plotting functions into one figure with subplots.
    plot_funcs: A list of functions, each function creates a single plot on a given axis.
                Each function should have a signature like: plot_func(ax)
    ncols: Number of columns in the subplot layout.
    """
    n = len(plot_funcs)
    nrows = (n + ncols - 1) // ncols  # Compute the needed number of rows
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()
    
    for i, plot_func in enumerate(plot_funcs):
        plot_func(axes[i])  # Each plot function draws on the provided axis
    
    # Hide unused subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig


def padre_vs_normal_table(table_norm, DtpRfPadreTable, column_names=['Normal', 'PADRE']):
    # Convert probability columns into a single string "p1, p5, p20" for each row
    def format_probs(df):
        return df.apply(lambda row: f"{row['Prob >= 1']:.2f}, {row['Prob >= 5']:.2f}, {row['Prob >= 20']:.2f}", axis=1)

    normal_probs = format_probs(table_norm)
    padre_probs = format_probs(DtpRfPadreTable)

    col_1 = column_names[0]
    col_2 = column_names[1]

    latex_str = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        f"& \\multicolumn{{3}}{{c}}{{${{\\text{{{col_1}}}}}$}} & \\multicolumn{{3}}{{c}}{{${{\\text{{{col_2}}}}}$}} \\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Name & Avg. & Std. dev. & Prob $\\geq$ & Avg. & Std. dev. & Prob $\\geq$ \\\\\n"
        "& & & \\{1, 5, 20\\} & & & \\{1, 5, 20\\} \\\\ \n"  # Note the space and newline
        "\\midrule\n"
    )

    # Now add the data rows
    # Ensure exact format: Name, Avg., Std. dev., Prob, Avg., Std. dev., Prob
    # Each row has 7 entries separated by '&'.
    for name in table_norm.index:
        normal_row = table_norm.loc[name]
        padre_row = DtpRfPadreTable.loc[name]

        # Mimic the spacing as in the correct snippet (extra spaces after the Name)
        # This spacing is cosmetic and optional, but included for similarity.
        line = (
            f"{name:<10} & "  # Left-justify name in a field of width 10 for spacing
            f"{normal_row['Avg.']:.2f} & "
            f"{normal_row['Std. dev.']:.2f} & "
            f"{normal_probs[name]} & "
            f"{padre_row['Avg.']:.2f} & "
            f"{padre_row['Std. dev.']:.2f} & "
            f"{padre_probs[name]} \\\\\n"
        )
        latex_str += line

    latex_str += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Comparison of Normal and Padre datasets across key metrics.}\n"
        "\\label{tab:normal_padre_updated}\n"
        "\\end{table}\n"
    )

    return latex_str


def generate_comparison_table(tables):
    """
    Generates a comparison table from a list of DataFrames.
    Each DataFrame is expected to have columns: ['Avg.', 'Std. dev.', 'Prob >= 1', 'Prob >= 5', 'Prob >= 20'].
    The index in each DataFrame should contain rows that include 'No noise', 'noise 1', 'noise 2', 'noise 3', etc.

    Returns a string containing LaTeX code for the table.
    """

    # Mapping from the internal "noise x" labels to the displayed LaTeX labels
    noise_mapping = {
        'noise 1': 'Noise 1$\\sigma$',
        'noise 2': 'Noise 2$\\sigma$',
        'noise 3': 'Noise 3$\\sigma$',
    }

    # These are the row labels we'll look for in each DataFrame index
    row_order = ['No noise', 'noise 1', 'noise 2', 'noise 3']

    labels = []
    table_dicts = []
    formatted_probs_list = []

    # Process each DataFrame
    for df in tables:
        # Extract label from the first row's index name
        first_row_name = df.index[0]  # e.g., "MLP CDDD Greedy Padre"
        labels.append(first_row_name)
        
        # Rename the first row's index to 'No noise'
        df = df.copy()
        df.rename(index={first_row_name: 'No noise'}, inplace=True)
        
        # Format probabilities for the row "No noise", "noise 1", etc.
        formatted_probs = df.apply(
            lambda row: f"{row['Prob >= 1']:.2f}, {row['Prob >= 5']:.2f}, {row['Prob >= 20']:.2f}",
            axis=1
        )
        formatted_probs_list.append(formatted_probs.to_dict())
        table_dicts.append(df.to_dict(orient='index'))

    # Collect all unique row labels from all DataFrames
    all_names = set()
    for df_dict in table_dicts:
        all_names.update(df_dict.keys())

    # Calculate total number of columns:
    # 1 column for the row label + 3 columns (Avg., Std. dev., Prob >= {1,5,20}) per DataFrame
    num_columns = 1 + len(tables) * 3

    # Start constructing the LaTeX table
    latex_str = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Comparison across datasets.}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        f"\\begin{{tabular}}{{l{'c' * (num_columns - 1)}}}\n"
        "\\toprule\n"
    )

    # Header row: each 'label' spans 3 columns
    header = " & " + " & ".join(
        [f"\\multicolumn{{3}}{{c}}{{{label}}}" for label in labels]
    ) + " \\\\\n"
    latex_str += header

    # Midrule to visually separate top header from sub-headers
    column_ranges = []
    col_start = 2  # 1st column is 'Name', so data starts at column 2
    for _ in range(len(tables)):
        col_range = f"{col_start}-{col_start + 2}"
        column_ranges.append(col_range)
        col_start += 3
    midrule = "".join([f"\\cmidrule(lr){{{cr}}}" for cr in column_ranges]) + "\n"
    latex_str += midrule

    # Sub-headers
    # We'll bundle (Prob >= 1, 5, 20) as a single column to match your example
    sub_headers = []
    for _ in tables:
        sub_headers.extend(["Avg.", "Std. dev.", "Prob $\\geq$ \\{1, 5, 20\\}"])
    latex_str += " & " + " & ".join(sub_headers) + " \\\\\n"
    latex_str += "\\midrule\n"

    # Iterate over the defined row order
    for name in row_order:
        # Convert (e.g.) 'noise 1' to '1$\\sigma$ Noise' for final display
        display_name = noise_mapping.get(name, name)

        if name in all_names:
            line_elements = [display_name]  # First column in LaTeX
            for idx in range(len(tables)):
                table = table_dicts[idx]
                probs = formatted_probs_list[idx]
                if name in table:
                    row = table[name]
                    line_elements.extend([
                        f"{row['Avg.']:.2f}",
                        f"{row['Std. dev.']:.2f}",
                        f"{probs[name]}"
                    ])
                else:
                    # If this label doesn't exist in the current table's index, insert placeholders
                    line_elements.extend(["---"] * 3)
            # Join the elements with '&' and end with '\\'
            latex_str += " & ".join(line_elements) + " \\\\\n"
        # else skip row if not in current table set

    latex_str += (
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\label{tab:comparison_table}\n"
        "\\end{table}\n"
    )

    return latex_str


def create_probability_table_leq(df_le1, df_le5, df_le20):
    # Define the column subset we want to show (0 through 5)
    cols = [0, 1, 2, 3, 4, 5]

    latex_str = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\scriptsize % or \\footnotesize, \\tiny if needed\n"
        "\\setlength{\\tabcolsep}{3pt} % adjust as needed to reduce horizontal space\n"
        "\\begin{tabular}{l cccccc c cccccc c cccccc}\n"
        "\\toprule\n"
        "& \\multicolumn{6}{c}{$\\leq 1$ hit} & & \\multicolumn{6}{c}{$\\leq 5$ hits} & & \\multicolumn{6}{c}{$\\leq 20$ hits}\\\\\n"
        "\\cmidrule(lr){2-7}\\cmidrule(lr){9-14}\\cmidrule(lr){16-21}\n"
        "& 0 & 1 & 2 & 3 & 4 & 5 & & 0 & 1 & 2 & 3 & 4 & 5 & & 0 & 1 & 2 & 3 & 4 & 5 \\\\\n"
        "\\midrule\n"
    )

    # Iterate through the rows (methods) of the first DataFrame (assuming all have the same rows)
    for name in df_le1.index:
        # Extract values for each DataFrame, all 6 columns
        vals_le1 = df_le1.loc[name, cols]
        vals_le5 = df_le5.loc[name, cols]
        vals_le20 = df_le20.loc[name, cols]

        # Format row
        # We have:
        # name | 6 vals from df_le1 | ' & ' | 6 vals from df_le5 | ' & ' | 6 vals from df_le20
        # Ensure alignment by adding an extra '&' after the second group
        line = f"{name:<20}"  # Left-align name in a wider field for readability
        line += " & " + " & ".join(f"{v:.2f}" for v in vals_le1) + " & & "  # Add '& &' for spacing
        line += " & ".join(f"{v:.2f}" for v in vals_le5) + " & & "          # Add '& &' for spacing
        line += " & ".join(f"{v:.2f}" for v in vals_le20) + " \\\\\n"       # Last group

        latex_str += line

    latex_str += (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Probabilities over iterations for obtaining at least 1, 5, and 20 hits during 5 active learning cycles.}\n"
        "\\label{tab:compact}\n"
        "\\end{table}\n"
    )

    return latex_str


def generate_comparison_table_enrichment(tables, labels=['header1', 'header2']):
    table_dicts = []
    formatted_probs_list = []

    # Process each dataframe
    for df in tables:
        # Format probabilities
        formatted_probs = df.apply(
            lambda row: f"{row['Prob >= 1']:.2f}, {row['Prob >= 5']:.2f}, {row['Prob >= 20']:.2f}",
            axis=1
        )
        formatted_probs_list.append(formatted_probs.to_dict())
        table_dicts.append(df.to_dict(orient='index'))

    # Collect all unique names from all dataframes
    all_names = set()
    for df_dict in table_dicts:
        all_names.update(df_dict.keys())

    # Define the order of rows
    row_order = sorted(all_names, key=lambda x: (
        int(x.split()[0]) if x[0].isdigit() else float('inf'), x
    ))

    # Calculate total number of columns (Name + 3 columns per table)
    num_columns = 1 + len(tables) * 3

    # Start constructing the LaTeX table
    latex_str = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{l" + "c" * (num_columns - 1) + "}\n"
        "\\toprule\n"
    )

    # Header for labels
    header = " & " + " & ".join(
        [f"\\multicolumn{{3}}{{c}}{{{label}}}" for label in labels]
    ) + " \\\\\n"
    latex_str += header

    # Midrule for labels
    column_ranges = []
    col_start = 2
    for _ in range(len(tables)):
        col_range = f"{col_start}-{col_start+2}"
        column_ranges.append(col_range)
        col_start += 3
    midrule = "".join([f"\\cmidrule(lr){{{cr}}}" for cr in column_ranges]) + "\n"
    latex_str += midrule

    # Sub-headers
    sub_headers = []
    for _ in tables:
        sub_headers.extend(["Avg.", "Std. dev.", "Prob $\\geq$ \\{1, 5, 20\\}"])
    latex_str += " & " + " & ".join(sub_headers) + " \\\\\n"
    latex_str += "\\midrule\n"

    # Iterate over the specified row order
    for name in row_order:
        line_elements = [name]
        for idx in range(len(tables)):
            table = table_dicts[idx]
            probs = formatted_probs_list[idx]
            if name in table:
                row = table[name]
                line_elements.extend([
                    f"{row['Avg.']:.2f}",
                    f"{row['Std. dev.']:.2f}",
                    f"{probs[name]}"
                ])
            else:
                line_elements.extend(["---"] * 3)
        line = " & ".join(line_elements) + " \\\\\n"
        latex_str += line

    latex_str += (
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\caption{Comparison across datasets.}\n"
        "\\label{tab:comparison_table}\n"
        "\\end{table}\n"
    )

    return latex_str


def combine_probability_graphs(graphs_list, graphs_per_row=3, dpi=150):
    """
    Combine multiple matplotlib figures with subplots into one large figure.
    
    Parameters:
    - graphs_list: List of matplotlib figures (each containing subplots).
    - graphs_per_row: Number of subplots per row in the combined figure.
    - dpi: DPI for the combined figure.
    
    Returns:
    - combined_fig: The combined matplotlib figure.
    """
    # Extract all axes from each figure in the list
    all_axes = []
    for fig in graphs_list:
        all_axes.extend(fig.axes)  # Collect all subplot axes

    # Determine the total number of subplots needed
    total_plots = len(all_axes)
    rows = int(np.ceil(total_plots / graphs_per_row))

    # Create the combined figure
    combined_fig, combined_axes = plt.subplots(
        rows, graphs_per_row, figsize=(graphs_per_row * 5, rows * 4), dpi=dpi
    )
    combined_axes = combined_axes.flatten()

    # Loop through each axis from the original figures and copy its content
    for original_ax, combined_ax in zip(all_axes, combined_axes):
        # Copy lines
        for line in original_ax.get_lines():
            combined_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                label=line.get_label(),
                linestyle=line.get_linestyle(),
                color=line.get_color(),
                marker=line.get_marker(),
                alpha=line.get_alpha(),
            )

        # Copy shaded areas (confidence intervals)
        for collection in original_ax.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):  # For fill_between
                path = collection.get_paths()[0]
                vertices = path.vertices
                combined_ax.fill_between(
                    vertices[:, 0], vertices[:, 1], alpha=collection.get_alpha(), color=collection.get_facecolor()[0]
                )

        # Copy titles, labels, and legends
        combined_ax.set_title(original_ax.get_title())
        combined_ax.set_xlabel(original_ax.get_xlabel())
        combined_ax.set_ylabel(original_ax.get_ylabel())
        combined_ax.legend(loc='upper left')

    # Hide any unused subplots in the combined figure
    for ax in combined_axes[len(all_axes):]:
        ax.axis('off')

    plt.tight_layout()
    return combined_fig
