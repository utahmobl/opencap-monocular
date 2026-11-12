import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_mot import load_mot


def run_ik_analysis(
    subject,
    session,
    cam,
    movement,
    exisiting_results_path_csv=None,
    output_path_csv=None,
    trimmed=False,
    output_case_path=None,
    run_wham=False,
    run_2cams=False,
):
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if not trimmed:
        mono_path_folder = os.path.join(
            root_path,
            f"output/{subject}/{session}/{cam}/{movement}/{movement}/OpenSim/IK/shiftedIK",
        )
        if output_case_path is not None:
            mono_path_folder = os.path.join(
                output_case_path, "OpenSim", "IK", "shiftedIK"
            )
    else:
        mono_path_folder = os.path.join(
            root_path,
            f"output/{subject}/{session}/{cam}/{movement}/{movement}_trimmed/OpenSim/IK/shiftedIK",
        )
        if output_case_path is not None:
            mono_path_folder = os.path.join(
                output_case_path, "OpenSim", "IK", "shiftedIK"
            )
    mono_path = None
    if exisiting_results_path_csv is not None:
        if os.path.exists(exisiting_results_path_csv):
            output_df = pd.read_csv(exisiting_results_path_csv)
        else:
            output_df = pd.DataFrame()
            exisiting_results_path_csv = None
    else:
        output_df = pd.DataFrame()

    for file in os.listdir(mono_path_folder):
        if file.endswith("_sync.mot"):
            mono_path = os.path.join(mono_path_folder, file)
            break

    wham_path = None
    if run_wham:
        for file in os.listdir(mono_path_folder):
            if file.endswith("_sync_wham.mot"):
                wham_path = os.path.join(mono_path_folder, file)
                break

    # Add 2cams path search similar to wham
    twocams_path = None
    if run_2cams:
        twocam_folder = os.path.join(
            root_path,
            f"LabValidation_withVideos1/{subject}/OpenSimData/Video/HRNet/2-cameras/IK",
        )
        twocam_ik_files = []
        for root, dirs, files in os.walk(twocam_folder):
            for file in files:
                if file.endswith(".mot") and movement in file:
                    twocam_ik_files.append(os.path.join(root, file))

        assert (
            len(twocam_ik_files) == 1
        ), f"Expected 1 2-camera IK file, found {len(twocam_ik_files)} for {subject} {trial}"
        twocams_path = twocam_ik_files[0]

    mocap_path = os.path.join(
        root_path,
        f"LabValidation_withVideos1/{subject}/OpenSimData/Mocap/IK/{movement}.mot",
    )

    mot_paths = {
        "mocap": mocap_path,
        "mono": mono_path,
    }

    if wham_path is not None:
        mot_paths["wham"] = wham_path

    if twocams_path is not None:
        mot_paths["2cams"] = twocams_path

    output_path = os.path.join(mono_path_folder)
    # output_path = os.path.join(root_path, 'output/subject3/Session0/Cam1/STS1/STS1/OpenSim/IK/shiftedIK/')

    mot_data = {
        key: load_mot(os.path.join(root_path, mot_paths[key]))
        for key in mot_paths.keys()
    }

    mocap = mot_data["mocap"]
    mono = mot_data["mono"]

    if run_wham:
        wham = mot_data["wham"]
        wham = pd.DataFrame(data=wham[0], columns=wham[1])
        wham_time = wham["time"]

    if run_2cams:
        twocams = mot_data["2cams"]
        twocams = pd.DataFrame(data=twocams[0], columns=twocams[1])
        twocams_time = twocams["time"]

    # convert to df
    mocap = pd.DataFrame(data=mocap[0], columns=mocap[1])
    mono = pd.DataFrame(data=mono[0], columns=mono[1])

    mocap_time = mocap["time"]
    mono_time = mono["time"]

    min_time = max(mocap_time.iloc[0], mono_time.iloc[0])
    max_time = min(mocap_time.iloc[-1], mono_time.iloc[-1])

    mocap = mocap[(mocap["time"] >= min_time) & (mocap["time"] <= max_time)]
    mono = mono[(mono["time"] >= min_time) & (mono["time"] <= max_time)]

    if run_wham:
        wham = wham[(wham["time"] >= min_time) & (wham["time"] <= max_time)]

    if run_2cams:
        twocams = twocams[(twocams["time"] >= min_time) & (twocams["time"] <= max_time)]

    # single string for the time
    min_time = str(min_time)
    max_time = str(max_time)
    time = f"{min_time} to {max_time}"

    # reset the index
    mocap.reset_index(drop=True, inplace=True)
    mono.reset_index(drop=True, inplace=True)
    if run_wham:
        wham.reset_index(drop=True, inplace=True)
    if run_2cams:
        twocams.reset_index(drop=True, inplace=True)

    # find the common columns between the two dataframes
    common_columns = list(set(mocap.columns).intersection(mono.columns))
    common_columns.remove("time")

    if exisiting_results_path_csv is None:
        output_df["subject"] = [subject]
        output_df["session"] = [session]
        output_df["cam"] = [cam]
        output_df["movement"] = [movement]
        output_df["time"] = [time]
    # else append
    else:
        # keys = read columns
        keys = output_df.keys()
        new_row = pd.DataFrame(columns=keys)
        new_row["subject"] = [subject]
        new_row["session"] = [session]
        new_row["cam"] = [cam]
        new_row["movement"] = [movement]
        new_row["time"] = [time]

    assert mocap.shape[0] == mono.shape[0], "Lengths do not match"

    measurements = {
        "pelvis_tilt": ["pelvis_tilt"],
        "pelvis_list": ["pelvis_list"],
        "pelvis_rotation": ["pelvis_rotation"],
        "pelvis_tx": ["pelvis_tx"],
        "pelvis_ty": ["pelvis_ty"],
        "pelvis_tz": ["pelvis_tz"],
        "hip_flexion": ["hip_flexion_l", "hip_flexion_r"],
        "hip_adduction": ["hip_adduction_l", "hip_adduction_r"],
        "hip_rotation": ["hip_rotation_l", "hip_rotation_r"],
        "knee_angle": ["knee_angle_l", "knee_angle_r"],
        "ankle_angle": ["ankle_angle_l", "ankle_angle_r"],
        "subtalar_angle": ["subtalar_angle_l", "subtalar_angle_r"],
        "lumbar_extension": ["lumbar_extension"],
        "lumbar_bending": ["lumbar_bending"],
        "lumbar_rotation": ["lumbar_rotation"],
    }

    translations = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
    # rotations are everything else from measurements
    rotations = [
        col
        for key, cols in measurements.items()
        for col in cols
        if col not in translations
    ]

    # Define the data sources to compare
    comparisons = [("mono", mono)]
    if run_wham:
        comparisons.append(("wham", wham))
    if run_2cams:
        comparisons.append(("2cams", twocams))

    # Initialize counters and global MAE variables
    mae_global_degrees = 0
    mae_global_mm = 0

    if run_wham:
        mae_global_degrees_wham = 0
        mae_global_mm_wham = 0

    if run_2cams:
        mae_global_degrees_2cams = 0
        mae_global_mm_2cams = 0

    for source_name, source_data in comparisons:
        source_prefix = "" if source_name == "mono" else f"{source_name}_"
        current_mae_degrees = 0
        current_mae_mm = 0
        current_rotation_count = 0
        current_translation_count = 0
        sum_mae_degrees = 0
        sum_mae_mm = 0

        for key, cols in measurements.items():
            mae_list = []
            unit = None
            for col in cols:
                if col in mocap.columns and col in source_data.columns:
                    mae = (mocap[col] - source_data[col]).abs().mean()

                    unit = (
                        "mm" if "tx" in col or "ty" in col or "tz" in col else "degrees"
                    )
                    if "tx" in col or "ty" in col or "tz" in col:
                        # convert to mm
                        mae *= 1000
                        mae = np.round(mae, 2)
                        current_translation_count += 1
                        sum_mae_mm += mae
                    else:
                        mae = np.round(mae, 2)
                        current_rotation_count += 1
                        sum_mae_degrees += mae
                    mae_list.append(mae)
                    # print(f"{source_name} {col}: {mae} {unit}.")
                    if exisiting_results_path_csv is None:
                        output_df[f"{source_prefix}{col}"] = [mae]
                    else:
                        new_row[f"{source_prefix}{col}"] = [mae]

            if sum_mae_degrees > 0:
                current_mae_degrees = round(sum_mae_degrees / current_rotation_count, 2)
            if sum_mae_mm > 0:
                current_mae_mm = round(sum_mae_mm / current_translation_count, 2)

        # Store the metrics and counts for this source
        if source_name == "mono":
            mae_global_degrees = current_mae_degrees
            mae_global_mm = current_mae_mm

        elif source_name == "wham":
            mae_global_degrees_wham = current_mae_degrees
            mae_global_mm_wham = current_mae_mm

        elif source_name == "2cams":
            mae_global_degrees_2cams = current_mae_degrees
            mae_global_mm_2cams = current_mae_mm

        # Print intermediate results with counts
        print(
            f"{source_name.capitalize()} Global MAE degrees: {current_mae_degrees} (from {current_rotation_count} rotations)."
        )
        print(
            f"{source_name.capitalize()} Global MAE mm: {current_mae_mm} (from {current_translation_count} translations)."
        )

    # Define file paths for results
    results_path = os.path.join(output_path, "translation_error.txt")

    if run_wham:
        wham_results_path = os.path.join(output_path, "translation_error_wham.txt")

    # Write mono results to a file
    with open(results_path, "w") as f:
        f.write(f"Global MAE degrees: {mae_global_degrees}.\n")
        f.write(f"Global MAE mm: {mae_global_mm}.\n")

        for col in common_columns:
            if col in output_df.columns:
                f.write(f"{col}: {output_df[col].values[0]}.\n")

    # Write wham results to a separate file if needed
    if run_wham:
        with open(wham_results_path, "w") as f:
            f.write(f"WHAM Global MAE degrees: {mae_global_degrees_wham}.\n")
            f.write(f"WHAM Global MAE mm: {mae_global_mm_wham}.\n")

            for col in common_columns:
                wham_col = f"wham_{col}"
                if wham_col in output_df.columns:
                    f.write(f"{col}: {output_df[wham_col].values[0]}.\n")

    # Prepare mono CSV output
    if exisiting_results_path_csv is not None:
        # Create a clean copy for mono results only - exclude any method-specific columns
        mono_columns = [
            col
            for col in new_row.columns
            if not col.startswith("wham_") and not col.startswith("2cams_")
        ]
        mono_new_row = new_row[mono_columns].copy()

        # Add global metrics to the mono new row
        mono_new_row["global_mae_degrees"] = mae_global_degrees
        mono_new_row["global_mae_mm"] = mae_global_mm

        # Add the new row to the existing DataFrame
        output_df_mono = pd.concat([output_df, mono_new_row], ignore_index=True)
        output_csv_path = exisiting_results_path_csv
    else:
        # Filter out wham and 2cams columns from output_df
        mono_columns = [
            col
            for col in output_df.columns
            if not col.startswith("wham_") and not col.startswith("2cams_")
        ]
        output_df_mono = output_df[mono_columns].copy()

        # Add global metrics to output_df_mono
        output_df_mono["global_mae_degrees"] = mae_global_degrees
        output_df_mono["global_mae_mm"] = mae_global_mm

        output_csv_path = os.path.join(output_path, "translation_error.csv")

    if output_path_csv is not None:
        output_csv_path = output_path_csv

    output_df_mono.to_csv(output_csv_path, index=False)

    # Create and save wham CSV if needed
    if run_wham:
        # Create a DataFrame with only the wham columns, renaming them to remove the 'wham_' prefix
        if exisiting_results_path_csv is not None:
            wham_columns = [col for col in new_row.columns if col.startswith("wham_")]
            wham_new_row = new_row[wham_columns].copy()

            # Rename columns to remove 'wham_' prefix
            wham_new_row.columns = [
                col.replace("wham_", "") for col in wham_new_row.columns
            ]

            # Add the basic info columns
            for col in ["subject", "session", "cam", "movement", "time"]:
                if col in new_row:
                    wham_new_row[col] = new_row[col]

            # Add global metrics to wham_new_row
            wham_new_row["global_mae_degrees"] = mae_global_degrees_wham
            wham_new_row["global_mae_mm"] = mae_global_mm_wham

            # Read existing wham CSV if it exists
            wham_csv_path = output_csv_path.replace(".csv", "_wham.csv")
            if os.path.exists(wham_csv_path):
                existing_wham_df = pd.read_csv(wham_csv_path)
                output_df_wham = pd.concat(
                    [existing_wham_df, wham_new_row], ignore_index=True
                )
            else:
                output_df_wham = wham_new_row
        else:
            wham_columns = [col for col in output_df.columns if col.startswith("wham_")]
            output_df_wham = output_df[wham_columns].copy()

            # Rename columns to remove 'wham_' prefix
            output_df_wham.columns = [
                col.replace("wham_", "") for col in output_df_wham.columns
            ]

            # Add the basic info columns
            for col in ["subject", "session", "cam", "movement", "time"]:
                if col in output_df:
                    output_df_wham[col] = output_df[col]

            # Add global metrics to output_df_wham
            output_df_wham["global_mae_degrees"] = mae_global_degrees_wham
            output_df_wham["global_mae_mm"] = mae_global_mm_wham

            wham_csv_path = output_csv_path.replace(".csv", "_wham.csv")

        # Save the wham CSV
        output_df_wham.to_csv(wham_csv_path, index=False)

    # Create and save 2cams CSV if needed
    if run_2cams:
        # Write 2cams results to a file
        twocams_results_path = os.path.join(output_path, "translation_error_2cams.txt")

        with open(twocams_results_path, "w") as f:
            f.write(
                f"2CAMS Global MAE degrees: {round(mae_global_degrees_2cams, 2)}.\n"
            )
            f.write(f"2CAMS Global MAE mm: {round(mae_global_mm_2cams, 1)}.\n")

            for col in common_columns:
                twocams_col = f"2cams_{col}"
                if twocams_col in output_df.columns:
                    f.write(f"{col}: {output_df[twocams_col].values[0]}.\n")

        # Create a DataFrame with only the 2cams columns, renaming them to remove the '2cams_' prefix
        if exisiting_results_path_csv is not None:
            twocams_columns = [
                col for col in new_row.columns if col.startswith("2cams_")
            ]
            twocams_new_row = new_row[twocams_columns].copy()

            # Rename columns to remove '2cams_' prefix
            twocams_new_row.columns = [
                col.replace("2cams_", "") for col in twocams_new_row.columns
            ]

            # Add the basic info columns
            for col in ["subject", "session", "cam", "movement", "time"]:
                if col in new_row:
                    twocams_new_row[col] = new_row[col]

            # Add global metrics to twocams_new_row
            twocams_new_row["global_mae_degrees"] = mae_global_degrees_2cams
            twocams_new_row["global_mae_mm"] = mae_global_mm_2cams

            # Read existing 2cams CSV if it exists
            twocams_csv_path = output_csv_path.replace(".csv", "_2cams.csv")
            if os.path.exists(twocams_csv_path):
                existing_twocams_df = pd.read_csv(twocams_csv_path)
                output_df_twocams = pd.concat(
                    [existing_twocams_df, twocams_new_row], ignore_index=True
                )
            else:
                output_df_twocams = twocams_new_row
        else:
            twocams_columns = [
                col for col in output_df.columns if col.startswith("2cams_")
            ]
            output_df_twocams = output_df[twocams_columns].copy()

            # Rename columns to remove '2cams_' prefix
            output_df_twocams.columns = [
                col.replace("2cams_", "") for col in output_df_twocams.columns
            ]

            # Add the basic info columns
            for col in ["subject", "session", "cam", "movement", "time"]:
                if col in output_df:
                    output_df_twocams[col] = output_df[col]

            # Add global metrics to output_df_twocams
            output_df_twocams["global_mae_degrees"] = mae_global_degrees_2cams
            output_df_twocams["global_mae_mm"] = mae_global_mm_2cams

            twocams_csv_path = output_csv_path.replace(".csv", "_2cams.csv")

        # Save the 2cams CSV
        output_df_twocams.to_csv(twocams_csv_path, index=False)

    # Create appropriate comparison plots based on available methods
    if run_wham and run_2cams:
        # When all three methods are available, only create the three-method comparison
        create_comparison_barplots_three_methods(
            mae_global_mm,
            mae_global_mm_wham,
            mae_global_mm_2cams,
            mae_global_degrees,
            mae_global_degrees_wham,
            mae_global_degrees_2cams,
            output_path,
        )
    elif run_wham:
        # Only create WHAM vs Mono comparison when 2CAMS is not enabled
        create_comparison_barplots(
            mae_global_mm,
            mae_global_mm_wham,
            mae_global_degrees,
            mae_global_degrees_wham,
            output_path,
        )
    elif run_2cams:
        # Only create 2CAMS vs Mono comparison when WHAM is not enabled
        create_comparison_barplots(
            mae_global_mm,
            mae_global_mm_2cams,
            mae_global_degrees,
            mae_global_degrees_2cams,
            output_path,
            methods=["Mono", "2CAMS"],
            filename_suffix="_2cams",
        )

    # Return appropriate values based on which methods were run
    if run_wham and run_2cams:
        return (
            mae_global_degrees,
            mae_global_mm,
            mae_global_degrees_wham,
            mae_global_mm_wham,
            mae_global_degrees_2cams,
            mae_global_mm_2cams,
            results_path,
            output_csv_path,
        )
    elif run_wham:
        return (
            mae_global_degrees,
            mae_global_mm,
            mae_global_degrees_wham,
            mae_global_mm_wham,
            results_path,
            output_csv_path,
        )
    elif run_2cams:
        return (
            mae_global_degrees,
            mae_global_mm,
            mae_global_degrees_2cams,
            mae_global_mm_2cams,
            results_path,
            output_csv_path,
        )
    else:
        return mae_global_degrees, mae_global_mm, results_path, output_csv_path


def create_comparison_barplots(
    mono_mm,
    other_mm,
    mono_degrees,
    other_degrees,
    output_path,
    methods=["Mono", "WHAM"],
    filename_suffix="",
):
    """
    Create barplots comparing two methods for both mm and degrees metrics.

    Args:
        mono_mm: Global MAE in mm for mono approach
        other_mm: Global MAE in mm for the second approach
        mono_degrees: Global MAE in degrees for mono approach
        other_degrees: Global MAE in degrees for the second approach
        output_path: Directory to save the plots
        methods: List of method names to display on the x-axis
        filename_suffix: Optional suffix to add to output filenames
    """
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reorder methods and values if methods are the default ['Mono', 'WHAM']
    if set(methods) == set(["Mono", "WHAM"]):
        # Change order to ['WHAM', 'Mono']
        methods = ["WHAM", "Mono"]
        mm_values = [other_mm, mono_mm]
        degrees_values = [other_degrees, mono_degrees]
    elif set(methods) == set(["Mono", "2CAMS"]):
        # Change order to ['Mono', '2CAMS'] (keep as is since 2CAMS is already second)
        mm_values = [mono_mm, other_mm]
        degrees_values = [mono_degrees, other_degrees]
    else:
        # For custom method names, keep original order
        mm_values = [mono_mm, other_mm]
        degrees_values = [mono_degrees, other_degrees]

    # Colors
    colors = ["#e74c3c", "#3498db"] if methods[0] == "WHAM" else ["#3498db", "#e74c3c"]

    # Plot 1: MAE in mm
    bars1 = ax1.bar(methods, mm_values, color=colors, width=0.6)
    ax1.set_ylabel("Global MAE (mm)")
    ax1.set_title("Comparison of Global MAE in mm")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    # Plot 2: MAE in degrees
    bars2 = ax2.bar(methods, degrees_values, color=colors, width=0.6)
    ax2.set_ylabel("Global MAE (degrees)")
    ax2.set_title("Comparison of Global MAE in degrees")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # Adjust layout and save
    plt.tight_layout()

    # Save plots
    methods_str = "_".join(method.lower() for method in methods)
    comparison_plot_path = os.path.join(
        output_path, f"{methods_str}_comparison{filename_suffix}.png"
    )
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")

    # Also save individual plots
    plt.figure(figsize=(6, 5))
    bars_mm = plt.bar(methods, mm_values, color=colors, width=0.6)
    plt.ylabel("Global MAE (mm)")
    plt.title("Comparison of Global MAE in mm")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_mm:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"{methods_str}_mm{filename_suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure(figsize=(6, 5))
    bars_deg = plt.bar(methods, degrees_values, color=colors, width=0.6)
    plt.ylabel("Global MAE (degrees)")
    plt.title("Comparison of Global MAE in degrees")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_deg:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"{methods_str}_degrees{filename_suffix}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Close all figures to free memory
    plt.close("all")


def create_comparison_barplots_three_methods(
    mono_mm,
    wham_mm,
    twocams_mm,
    mono_degrees,
    wham_degrees,
    twocams_degrees,
    output_path,
):
    """
    Create barplots comparing mono, WHAM, and 2CAMS results for both mm and degrees metrics.

    Args:
        mono_mm: Global MAE in mm for mono approach
        wham_mm: Global MAE in mm for WHAM approach
        twocams_mm: Global MAE in mm for 2CAMS approach
        mono_degrees: Global MAE in degrees for mono approach
        wham_degrees: Global MAE in degrees for WHAM approach
        twocams_degrees: Global MAE in degrees for 2CAMS approach
        output_path: Directory to save the plots
    """
    # Set up figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Data for plots - reordered to WHAM, Mono, 2CAMS
    methods = ["WHAM", "Mono", "2CAMS"]
    mm_values = [wham_mm, mono_mm, twocams_mm]
    degrees_values = [wham_degrees, mono_degrees, twocams_degrees]

    # Colors - maintain color consistency with the method
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # Plot 1: MAE in mm
    bars1 = ax1.bar(methods, mm_values, color=colors, width=0.6)
    ax1.set_ylabel("Global MAE (mm)")
    ax1.set_title("Comparison of Global MAE in mm")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    # Plot 2: MAE in degrees
    bars2 = ax2.bar(methods, degrees_values, color=colors, width=0.6)
    ax2.set_ylabel("Global MAE (degrees)")
    ax2.set_title("Comparison of Global MAE in degrees")
    ax2.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # Adjust layout and save
    plt.tight_layout()

    # Save plots
    comparison_plot_path = os.path.join(output_path, "all_methods_comparison.png")
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")

    # Also save individual plots
    plt.figure(figsize=(8, 5))
    bars_mm = plt.bar(methods, mm_values, color=colors, width=0.6)
    plt.ylabel("Global MAE (mm)")
    plt.title("Comparison of Global MAE in mm")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_mm:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "all_methods_mm.png"), dpi=300, bbox_inches="tight"
    )

    plt.figure(figsize=(8, 5))
    bars_deg = plt.bar(methods, degrees_values, color=colors, width=0.6)
    plt.ylabel("Global MAE (degrees)")
    plt.title("Comparison of Global MAE in degrees")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_deg:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "all_methods_degrees.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # Close all figures to free memory
    plt.close("all")


if __name__ == "__main__":
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_csv = os.path.join(repo_path, "output", "results.csv")
    subject = "subject10"
    session = "Session0"
    cam = "Cam1"
    video = "STS1"
    csv_path = output_csv
    ik_results_degrees, ik_results_mm, results_path, output_csv_path = run_ik_analysis(
        subject,
        session,
        cam,
        video,
        exisiting_results_path_csv=csv_path,
        output_path_csv=output_csv,
        run_2cams=True,
    )
    print(ik_results_degrees)
    print(ik_results_mm)
    print(output_csv_path)
