import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# script to plot the polysemantic_count.csv file, which contains activations for
# feature 1 of Whisper Tiny block2.mlp.1


# Read the CSV file
def read_polysemantic_data(filename):
    # Read CSV and set first column as index
    df = pd.read_csv(filename, header=None)
    # First column contains row labels
    df = df.set_index(0)
    # Remove empty columns (those with all NaN values)
    df = df.dropna(axis=1, how="all")
    return df


# Create side-by-side histogram plot
def plot_histograms(df):
    plt.figure(figsize=(12, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Define common bins for all histograms
    all_values = pd.concat([row.dropna() for _, row in df.iterrows()])
    bins = np.linspace(min(all_values), max(all_values), 20)

    n_rows = len(df)
    bar_width = (bins[1] - bins[0]) / (n_rows + 1)  # Add 1 for spacing

    for i, ((idx, row), color) in enumerate(zip(df.iterrows(), colors)):
        data = row.dropna()
        # Calculate histogram data
        counts, bin_edges = np.histogram(data, bins=bins)
        # Shift the bars for each row
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        shifted_centers = centers + (i - n_rows / 2) * bar_width

        plt.bar(
            shifted_centers, counts, width=bar_width, label=idx, color=color, alpha=0.8
        )

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Distribution of Values for Each Row")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# Create means plot with confidence intervals
def plot_means_with_ci(df):
    plt.figure(figsize=(10, 6))

    means = []
    cis = []
    labels = []

    for idx, row in df.iterrows():
        data = row.dropna()
        mean = np.mean(data)
        ci = stats.t.interval(0.95, len(data) - 1, loc=mean, scale=stats.sem(data))

        means.append(mean)
        cis.append(ci)
        labels.append(idx)

    # Convert to numpy arrays for easier manipulation
    means = np.array(means)
    cis = np.array(cis)

    # Create error bars
    yerr = np.array([(means - cis[:, 0]), (cis[:, 1] - means)])

    # Plot
    plt.errorbar(
        range(len(means)),
        means,
        yerr=yerr,
        fmt="o",
        capsize=5,
        capthick=2,
        elinewidth=2,
        markersize=8,
    )

    plt.xticks(range(len(means)), labels)
    plt.xlabel("Row")
    plt.ylabel("Mean Value")
    plt.title("Mean Values with 95% Confidence Intervals")
    plt.grid(True, alpha=0.3)
    plt.show()


# Main execution
def main():
    # Read the data
    data_path = os.path.join(
        os.path.dirname(__file__), "../assets/polyesemantic_count.csv"
    )
    df = read_polysemantic_data(data_path)

    # Create both plots
    plot_histograms(df)
    plot_means_with_ci(df)


if __name__ == "__main__":
    main()
