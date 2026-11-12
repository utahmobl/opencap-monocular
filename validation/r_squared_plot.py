"""
This script plots a specified column from two .mot files and calculates the R2 score between them.

Example usage:
python validation/plot.py path/to/file1.mot path/to/file2.mot knee_angle_r
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import argparse
import numpy as np
from io import StringIO


def load_mot_file(filepath):
    """
    Loads a .mot file into a pandas DataFrame.
    It skips the header and uses the line after 'endheader' as column names.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end_index = -1
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_end_index = i
            break

    if header_end_index == -1:
        raise ValueError("Could not find 'endheader' in the file.")

    # Column names are on the next line after 'endheader'
    columns = lines[header_end_index + 1].strip().split("\t")

    # Data starts two lines after 'endheader'
    data_lines = lines[header_end_index + 2 :]

    # Use pandas to read the data
    data_io = StringIO("".join(data_lines))

    # read_csv with sep='\s+' is more robust to spacing inconsistencies
    df = pd.read_csv(data_io, sep="\s+", header=None, names=columns, index_col=False)

    # Clean up any extra columns that might be created
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(axis=1, how="all")

    return df


def plot_comparison(ax_scatter, ax_timeseries, df1, df2, column):
    """
    Generates a scatter and time-series plot for a given column.
    """
    # Check if the column exists in both files
    if column not in df1.columns or column not in df2.columns:
        print(f"Warning: Column '{column}' not found in one or both files. Skipping.")
        if column not in df1.columns:
            print(f"'{column}' not in file 1")
        if column not in df2.columns:
            print(f"'{column}' not in file 2")
        ax_scatter.set_visible(False)
        ax_timeseries.set_visible(False)
        return

    # Time Alignment
    merged_df = pd.merge_ordered(
        df1[["time", column]],
        df2[["time", column]],
        on="time",
        how="outer",
        suffixes=("_1", "_2"),
    )

    # Interpolate missing values and drop any remaining NaNs
    merged_df = merged_df.interpolate(method="linear").dropna()

    if merged_df.empty:
        print(
            f"Warning: The time ranges for '{column}' do not overlap, or no data remains after alignment."
        )
        ax_scatter.set_visible(False)
        ax_timeseries.set_visible(False)
        return

    y_true = merged_df[f"{column}_1"]
    y_pred = merged_df[f"{column}_2"]
    time = merged_df["time"]

    # Calculate R2 score
    r2 = r2_score(y_true, y_pred)

    # Scatter plot
    ax_scatter.scatter(y_true, y_pred, alpha=0.5)

    # Add a 1:1 line for reference
    lims = [
        np.min([y_true.min(), y_pred.min()]),
        np.max([y_true.max(), y_pred.max()]),
    ]
    # Add some padding to the limits
    lims = [lims[0] - (lims[1] - lims[0]) * 0.05, lims[1] + (lims[1] - lims[0]) * 0.05]
    ax_scatter.plot(lims, lims, "k-", alpha=0.75, zorder=0)
    ax_scatter.set_xlim(lims)
    ax_scatter.set_ylim(lims)

    ax_scatter.set_xlabel(f"{column} (File 1)")
    ax_scatter.set_ylabel(f"{column} (File 2)")
    ax_scatter.set_title(f"Correlation\nR2 Score: {r2:.4f}")
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.grid(True)

    # Time series plot
    ax_timeseries.plot(time, y_true, label=f"File 1")
    ax_timeseries.plot(time, y_pred, label=f"File 2", linestyle="--")
    ax_timeseries.set_xlabel("Time (s)")
    ax_timeseries.set_ylabel(column)
    ax_timeseries.set_title(f"Time Series Comparison for {column}")
    ax_timeseries.legend()
    ax_timeseries.grid(True)


def main():
    parser = argparse.ArgumentParser(description="Plot and compare two .mot files.")

    parser.add_argument(
        "--file1",
        type=str,
        help="Path to the first .mot file.",
        default="output/case_new/subject9/Session1/Cam3/walking1/mocap/walking1.mot",
    )
    parser.add_argument(
        "--file2",
        type=str,
        help="Path to the second .mot file.",
        default="output/case_new/subject9/Session1/Cam3/walking1/OpenSim/IK/shiftedIK/walking1_5_sync.mot",
    )
    parser.add_argument(
        "--column_base",
        type=str,
        help="The base name of the column to plot (e.g., knee_angle).",
        default="knee_angle",
    )
    args = parser.parse_args()

    # Load the data
    try:
        df1 = load_mot_file(args.file1)
        df2 = load_mot_file(args.file2)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except ValueError as e:
        print(f"Error parsing file: {e}")
        return

    column_l = f"{args.column_base}_l"
    column_r = f"{args.column_base}_r"

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    title = (
        f"Comparison for {args.column_base}\n"
        f"File 1: {args.file1}\n"
        f"File 2: {args.file2}"
    )
    fig.suptitle(title, fontsize=14)

    plot_comparison(axes[0, 0], axes[0, 1], df1, df2, column_l)
    plot_comparison(axes[1, 0], axes[1, 1], df1, df2, column_r)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()


if __name__ == "__main__":
    main()
