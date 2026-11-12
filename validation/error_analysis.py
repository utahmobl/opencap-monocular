# for a given list of folders and cameras, load all the translation_error.csv and translation_error_wham.csv files and analyze errors

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import shutil
import plot_config as pc

# Define DOF categories
TRANSLATIONS = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]
EXCLUDED_DOFS = []  # ['subtalar_angle_r', 'subtalar_angle_l']

# Define marker categories
PELVIS_MARKERS = ["pelvis"]
ANKLE_MARKERS = ["ankles"]
TOE_MARKERS = ["toes"]


def analyze_errors(base_folders, camera_numbers, output_dir=None, exclusion_file=None):
    """
    Analyze errors from files in the specified folders and cameras.
    Load both mono and WHAM data for comparison.

    Args:
        base_folders (list): List of base folder paths to search
        camera_numbers (list): List of camera numbers to include
        output_dir (str, optional): Directory to save output files. Defaults to current directory.
        exclusion_file (str, optional): Path to a file containing a list of trials to exclude.

    Returns:
        tuple: (ik_results_mono, ik_results_wham, ik_results_twocam, marker_results_mono, marker_results_wham)
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()
    else:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Output files will be saved to: {os.path.abspath(output_dir)}")

    # Create a log file for detailed tracking
    log_path = os.path.join(output_dir, "analysis_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("=== Error Analysis Log ===\n\n")
        log_file.write(f"Analysis started at: {pd.Timestamp.now()}\n\n")
        log_file.write("Files processed:\n\n")

    # Load excluded trials if an exclusion file is provided
    excluded_trials = set()
    if exclusion_file:
        excluded_trials = load_excluded_trials(exclusion_file, log_path)

    # Initialize DataFrames to store results
    ik_results_mono = pd.DataFrame()
    ik_results_wham = pd.DataFrame()
    ik_results_twocam = pd.DataFrame()  # Keep the 2CAMS IK results

    marker_results_mono = pd.DataFrame()
    marker_results_wham = pd.DataFrame()
    # Remove marker_results_twocam initialization

    # Lists to store processed file paths
    mono_ik_files = []
    wham_ik_files = []
    twocam_ik_files = []
    mono_marker_files = []
    wham_marker_files = []
    # Remove twocam_marker_files initialization

    # Count of found files for reporting
    count_mono_ik = 0
    count_wham_ik = 0
    count_twocam_ik = 0
    count_mono_marker = 0
    count_wham_marker = 0
    # Remove count_twocam_marker initialization

    # Metadata column names (case-insensitive)
    metadata_cols = ["movement", "session", "subject", "cam", "trial", "camera"]

    # Walk through each base folder
    for base_folder in base_folders:
        movement = os.path.basename(base_folder)
        print(f"Searching in {base_folder} (Movement: {movement})...")

        for root, dirs, files in os.walk(base_folder):
            # Check if we're in the IK directory
            if "OpenSim/IK/shiftedIK" in root:
                # Look for translation_error.csv and translation_error_wham.csv
                mono_csv = os.path.join(root, "translation_error.csv")
                wham_csv = os.path.join(root, "translation_error_wham.csv")
                twocam_csv = os.path.join(root, "translation_error_2cams.csv")

                # Extract camera number from path
                camera_match = re.search(r"Cam(\d+)", root)
                if camera_match:
                    camera_num = int(camera_match.group(1))

                    # Only process if camera is in our list
                    if camera_num in camera_numbers:
                        # Extract subject information from path
                        subject_match = re.search(r"subject(\d+)", root)
                        subject = (
                            f"subject{subject_match.group(1)}"
                            if subject_match
                            else "Unknown"
                        )

                        # Extract trial information from path
                        trial_parts = root.split(os.sep)
                        trial = trial_parts[-4] if len(trial_parts) > 4 else "Unknown"

                        # Standardize movement name for exclusion check
                        standardized_movement = "Unknown"
                        if "walking" in movement.lower():
                            standardized_movement = "walking"
                        elif "sts" in movement.lower():
                            standardized_movement = "STS"
                        elif "squat" in movement.lower():
                            standardized_movement = "squat"

                        # Check if the trial should be excluded
                        if (
                            standardized_movement,
                            subject,
                            camera_num,
                            trial,
                        ) in excluded_trials:
                            with open(log_path, "a") as log_file:
                                log_file.write(
                                    f"EXCLUDED IK Trial: {subject}, Cam{camera_num}, {trial}, Movement: {standardized_movement}\n\n"
                                )
                            continue

                        # Process mono data
                        if os.path.exists(mono_csv):
                            try:
                                mono_df = pd.read_csv(mono_csv)
                                count_mono_ik += 1
                                mono_ik_files.append(mono_csv)

                                # Log the file
                                with open(log_path, "a") as log_file:
                                    log_file.write(f"Mono IK: {mono_csv}\n")
                                    log_file.write(
                                        f"  Subject: {subject}, Camera: {camera_num}, Trial: {trial}, Movement: {movement}\n"
                                    )
                                    log_file.write(
                                        f"  Rows: {len(mono_df)}, Columns: {len(mono_df.columns)}\n\n"
                                    )

                                # Remove existing metadata columns (case-insensitive)
                                columns_to_drop = [
                                    col
                                    for col in mono_df.columns
                                    if col.lower() in metadata_cols
                                ]
                                mono_df = mono_df.drop(
                                    columns=columns_to_drop, errors="ignore"
                                )

                                # Add clean metadata columns
                                mono_df["Movement"] = movement.lower()
                                # Standardize movement names
                                if "walking" in movement.lower():
                                    mono_df["Movement"] = "walking"
                                elif "sts" in movement.lower():
                                    mono_df["Movement"] = "STS"
                                elif "squat" in movement.lower():
                                    mono_df["Movement"] = "squat"

                                mono_df["Subject"] = subject
                                mono_df["Camera"] = camera_num
                                mono_df["Trial"] = trial

                                # Append to main results
                                ik_results_mono = pd.concat(
                                    [ik_results_mono, mono_df], ignore_index=True
                                )
                            except Exception as e:
                                print(
                                    f"Error reading mono IK file {mono_csv}: {str(e)}"
                                )
                                with open(log_path, "a") as log_file:
                                    log_file.write(
                                        f"ERROR with {mono_csv}: {str(e)}\n\n"
                                    )

                        # Process 2-camera data
                        if os.path.exists(twocam_csv):
                            try:
                                twocam_df = pd.read_csv(twocam_csv)
                                count_twocam_ik += 1
                                twocam_ik_files.append(twocam_csv)

                                # Log the file
                                with open(log_path, "a") as log_file:
                                    log_file.write(f"2CAMS IK: {twocam_csv}\n")
                                    log_file.write(
                                        f"  Subject: {subject}, Camera: {camera_num}, Trial: {trial}, Movement: {movement}\n"
                                    )
                                    log_file.write(
                                        f"  Rows: {len(twocam_df)}, Columns: {len(twocam_df.columns)}\n\n"
                                    )

                                # Remove existing metadata columns (case-insensitive)
                                columns_to_drop = [
                                    col
                                    for col in twocam_df.columns
                                    if col.lower() in metadata_cols
                                ]
                                twocam_df = twocam_df.drop(
                                    columns=columns_to_drop, errors="ignore"
                                )

                                # Add clean metadata columns
                                twocam_df["Movement"] = movement.lower()
                                # Standardize movement names
                                if "walking" in movement.lower():
                                    twocam_df["Movement"] = "walking"
                                elif "sts" in movement.lower():
                                    twocam_df["Movement"] = "STS"
                                elif "squat" in movement.lower():
                                    twocam_df["Movement"] = "squat"

                                twocam_df["Subject"] = subject
                                twocam_df["Camera"] = camera_num
                                twocam_df["Trial"] = trial

                                # Append to main results
                                ik_results_twocam = pd.concat(
                                    [ik_results_twocam, twocam_df], ignore_index=True
                                )
                            except Exception as e:
                                print(
                                    f"Error reading 2CAMS IK file {twocam_csv}: {str(e)}"
                                )
                                with open(log_path, "a") as log_file:
                                    log_file.write(
                                        f"ERROR with {twocam_csv}: {str(e)}\n\n"
                                    )

                        # Process WHAM data
                        if os.path.exists(wham_csv):
                            try:
                                wham_df = pd.read_csv(wham_csv)
                                count_wham_ik += 1
                                wham_ik_files.append(wham_csv)

                                # Log the file
                                with open(log_path, "a") as log_file:
                                    log_file.write(f"WHAM IK: {wham_csv}\n")
                                    log_file.write(
                                        f"  Subject: {subject}, Camera: {camera_num}, Trial: {trial}, Movement: {movement}\n"
                                    )
                                    log_file.write(
                                        f"  Rows: {len(wham_df)}, Columns: {len(wham_df.columns)}\n\n"
                                    )

                                # Remove existing metadata columns (case-insensitive)
                                columns_to_drop = [
                                    col
                                    for col in wham_df.columns
                                    if col.lower() in metadata_cols
                                ]
                                wham_df = wham_df.drop(
                                    columns=columns_to_drop, errors="ignore"
                                )

                                # Add clean metadata columns
                                wham_df["Movement"] = movement.lower()
                                # Standardize movement names
                                if "walking" in movement.lower():
                                    wham_df["Movement"] = "walking"
                                elif "sts" in movement.lower():
                                    wham_df["Movement"] = "STS"
                                elif "squat" in movement.lower():
                                    wham_df["Movement"] = "squat"

                                wham_df["Subject"] = subject
                                wham_df["Camera"] = camera_num
                                wham_df["Trial"] = trial

                                # Append to main results
                                ik_results_wham = pd.concat(
                                    [ik_results_wham, wham_df], ignore_index=True
                                )
                            except Exception as e:
                                print(
                                    f"Error reading WHAM IK file {wham_csv}: {str(e)}"
                                )
                                with open(log_path, "a") as log_file:
                                    log_file.write(
                                        f"ERROR with {wham_csv}: {str(e)}\n\n"
                                    )

            # Check for marker data
            elif "MarkerData" in root:
                # Look for marker analysis results
                mono_marker_csv = os.path.join(root, "marker_analysis_results_mono.csv")
                wham_marker_csv = os.path.join(root, "marker_analysis_results_wham.csv")

                # Extract camera number from path
                camera_match = re.search(r"Cam(\d+)", root)
                if camera_match:
                    camera_num = int(camera_match.group(1))

                    # Only process if camera is in our list
                    if camera_num in camera_numbers:
                        # Extract subject information from path
                        subject_match = re.search(r"subject(\d+)", root)
                        subject = (
                            f"subject{subject_match.group(1)}"
                            if subject_match
                            else "Unknown"
                        )

                        # Extract trial information from path
                        trial_parts = root.split(os.sep)
                        trial = trial_parts[-4] if len(trial_parts) > 4 else "Unknown"

                        # Standardize movement name for exclusion check
                        standardized_movement = "Unknown"
                        if "walking" in movement.lower():
                            standardized_movement = "walking"
                        elif "sts" in movement.lower():
                            standardized_movement = "STS"
                        elif "squat" in movement.lower():
                            standardized_movement = "squat"

                        # Check if the trial should be excluded
                        if (
                            standardized_movement,
                            subject,
                            camera_num,
                            trial,
                        ) in excluded_trials:
                            with open(log_path, "a") as log_file:
                                log_file.write(
                                    f"EXCLUDED Marker Trial: {subject}, Cam{camera_num}, {trial}, Movement: {standardized_movement}\n\n"
                                )
                            continue

                        # Process mono marker data
                        if os.path.exists(mono_marker_csv):
                            try:
                                mono_marker_df = pd.read_csv(mono_marker_csv)
                                count_mono_marker += 1
                                mono_marker_files.append(mono_marker_csv)

                                # Log the file
                                with open(log_path, "a") as log_file:
                                    log_file.write(f"Mono Marker: {mono_marker_csv}\n")
                                    log_file.write(
                                        f"  Subject: {subject}, Camera: {camera_num}, Trial: {trial}, Movement: {movement}\n"
                                    )
                                    log_file.write(
                                        f"  Rows: {len(mono_marker_df)}, Columns: {len(mono_marker_df.columns)}\n\n"
                                    )

                                # Remove existing metadata columns (case-insensitive)
                                columns_to_drop = [
                                    col
                                    for col in mono_marker_df.columns
                                    if col.lower() in metadata_cols
                                ]
                                mono_marker_df = mono_marker_df.drop(
                                    columns=columns_to_drop, errors="ignore"
                                )

                                # Add clean metadata columns
                                mono_marker_df["Movement"] = movement.lower()
                                # Standardize movement names
                                if "walking" in movement.lower():
                                    mono_marker_df["Movement"] = "walking"
                                elif "sts" in movement.lower():
                                    mono_marker_df["Movement"] = "STS"
                                elif "squat" in movement.lower():
                                    mono_marker_df["Movement"] = "squat"

                                mono_marker_df["Subject"] = subject
                                mono_marker_df["Camera"] = camera_num
                                mono_marker_df["Trial"] = trial

                                # Append to main results
                                marker_results_mono = pd.concat(
                                    [marker_results_mono, mono_marker_df],
                                    ignore_index=True,
                                )
                            except Exception as e:
                                print(
                                    f"Error reading mono marker file {mono_marker_csv}: {str(e)}"
                                )
                                with open(log_path, "a") as log_file:
                                    log_file.write(
                                        f"ERROR with {mono_marker_csv}: {str(e)}\n\n"
                                    )

                        # Process WHAM marker data
                        if os.path.exists(wham_marker_csv):
                            try:
                                wham_marker_df = pd.read_csv(wham_marker_csv)
                                count_wham_marker += 1
                                wham_marker_files.append(wham_marker_csv)

                                # Log the file
                                with open(log_path, "a") as log_file:
                                    log_file.write(f"WHAM Marker: {wham_marker_csv}\n")
                                    log_file.write(
                                        f"  Subject: {subject}, Camera: {camera_num}, Trial: {trial}, Movement: {movement}\n"
                                    )
                                    log_file.write(
                                        f"  Rows: {len(wham_marker_df)}, Columns: {len(wham_marker_df.columns)}\n\n"
                                    )

                                # Remove existing metadata columns (case-insensitive)
                                columns_to_drop = [
                                    col
                                    for col in wham_marker_df.columns
                                    if col.lower() in metadata_cols
                                ]
                                wham_marker_df = wham_marker_df.drop(
                                    columns=columns_to_drop, errors="ignore"
                                )

                                # Add clean metadata columns
                                wham_marker_df["Movement"] = movement.lower()
                                # Standardize movement names
                                if "walking" in movement.lower():
                                    wham_marker_df["Movement"] = "walking"
                                elif "sts" in movement.lower():
                                    wham_marker_df["Movement"] = "STS"
                                elif "squat" in movement.lower():
                                    wham_marker_df["Movement"] = "squat"

                                wham_marker_df["Subject"] = subject
                                wham_marker_df["Camera"] = camera_num
                                wham_marker_df["Trial"] = trial

                                # Append to main results
                                marker_results_wham = pd.concat(
                                    [marker_results_wham, wham_marker_df],
                                    ignore_index=True,
                                )
                            except Exception as e:
                                print(
                                    f"Error reading WHAM marker file {wham_marker_csv}: {str(e)}"
                                )
                                with open(log_path, "a") as log_file:
                                    log_file.write(
                                        f"ERROR with {wham_marker_csv}: {str(e)}\n\n"
                                    )

    # Log summary information - update to include 2CAMS IK but not marker
    with open(log_path, "a") as log_file:
        log_file.write("\n=== Analysis Summary ===\n\n")
        log_file.write(f"Total mono IK files: {count_mono_ik}\n")
        log_file.write(f"Total WHAM IK files: {count_wham_ik}\n")
        log_file.write(f"Total 2CAMS IK files: {count_twocam_ik}\n")
        log_file.write(f"Total mono marker files: {count_mono_marker}\n")
        log_file.write(f"Total WHAM marker files: {count_wham_marker}\n\n")

        # Record unique combinations
        mono_combinations = ik_results_mono.drop_duplicates(
            subset=["Subject", "Camera", "Trial"]
        ).shape[0]
        wham_combinations = ik_results_wham.drop_duplicates(
            subset=["Subject", "Camera", "Trial"]
        ).shape[0]
        twocam_combinations = ik_results_twocam.drop_duplicates(
            subset=["Subject", "Camera", "Trial"]
        ).shape[0]

        log_file.write(
            f"Unique mono subject/camera/trial combinations: {mono_combinations}\n"
        )
        log_file.write(
            f"Unique WHAM subject/camera/trial combinations: {wham_combinations}\n"
        )
        log_file.write(
            f"Unique 2CAMS subject/camera/trial combinations: {twocam_combinations}\n\n"
        )

    # Update print statement to show 2CAMS IK but not marker
    print(
        f"\nFound {count_mono_ik} mono IK results, {count_wham_ik} WHAM IK results, {count_twocam_ik} 2CAMS IK results"
    )
    print(
        f"Found {count_mono_marker} mono marker results, {count_wham_marker} WHAM marker results"
    )
    print(f"Detailed log written to {os.path.abspath(log_path)}")

    # Check for and fix column data types
    if not ik_results_mono.empty:
        ik_results_mono = clean_dataframe(ik_results_mono, "mono IK")

    if not ik_results_wham.empty:
        ik_results_wham = clean_dataframe(ik_results_wham, "WHAM IK")

    if not ik_results_twocam.empty:
        ik_results_twocam = clean_dataframe(ik_results_twocam, "2CAMS IK")

    if not marker_results_mono.empty:
        marker_results_mono = clean_dataframe(marker_results_mono, "mono marker")

    if not marker_results_wham.empty:
        marker_results_wham = clean_dataframe(marker_results_wham, "WHAM marker")

    # Compare IK errors if we have both mono and WHAM data
    if (
        not ik_results_mono.empty
        and not ik_results_wham.empty
        and not ik_results_twocam.empty
    ):
        compare_ik_errors_three_methods(
            ik_results_mono,
            ik_results_wham,
            ik_results_twocam,
            output_dir,
            log_path,
            mono_ik_files,
            wham_ik_files,
            twocam_ik_files,
        )
    elif not ik_results_mono.empty and not ik_results_wham.empty:
        compare_ik_errors(
            ik_results_mono,
            ik_results_wham,
            output_dir,
            log_path,
            mono_ik_files,
            wham_ik_files,
        )
    elif not ik_results_mono.empty and not ik_results_twocam.empty:
        compare_ik_errors(
            ik_results_mono,
            ik_results_twocam,
            output_dir,
            log_path,
            mono_ik_files,
            twocam_ik_files,
            methods=["Mono", "2CAMS"],
        )

    # Keep only mono and WHAM for marker comparison - no 2CAMS
    if not marker_results_mono.empty and not marker_results_wham.empty:
        compare_marker_errors(
            marker_results_mono, marker_results_wham, output_dir, log_path
        )

    # Return only what we're actually analyzing
    return (
        ik_results_mono,
        ik_results_wham,
        ik_results_twocam,
        marker_results_mono,
        marker_results_wham,
    )


def load_excluded_trials(exclusion_file, log_path):
    """
    Loads a list of trials to be excluded from a text file.
    Each line in the file should be a path that contains movement, subject, camera, and trial info.
    """
    excluded_trials = set()
    if not os.path.exists(exclusion_file):
        print(f"Warning: Exclusion file not found at {exclusion_file}")
        return excluded_trials

    with open(exclusion_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                # E.g.: output/case_001_walking/subject8/Session1/Cam3/walkingTS1/...
                path = Path(line)
                parts = path.parts

                # Find movement, subject, camera, and trial from the path
                movement_str = next(
                    (p for p in parts if p.startswith("case_001_")), None
                )
                subject_str = next((p for p in parts if p.startswith("subject")), None)
                camera_str = next((p for p in parts if p.startswith("Cam")), None)

                if not all([movement_str, subject_str, camera_str]):
                    print(
                        f"Warning: Could not parse all parts from exclusion line: {line}"
                    )
                    continue

                # Standardize movement name
                if "walking" in movement_str:
                    movement = "walking"
                elif "sts" in movement_str:
                    movement = "STS"
                elif "squat" in movement_str:
                    movement = "squat"
                else:
                    movement = "Unknown"

                subject = subject_str
                camera_num = int(re.search(r"\d+", camera_str).group())

                # The trial name is the folder after the camera folder
                cam_index = parts.index(camera_str)
                trial = parts[cam_index + 1]

                excluded_trials.add((movement, subject, camera_num, trial))
                print(f"Excluded trial: {movement}, {subject}, {camera_num}, {trial}")

            except (ValueError, IndexError, AttributeError) as e:
                print(
                    f"Warning: Skipping malformed line in exclusion file: {line} ({e})"
                )

    with open(log_path, "a") as log_file:
        log_file.write(
            f"Loaded {len(excluded_trials)} trials to exclude from {exclusion_file}\n"
        )
        for excluded in sorted(list(excluded_trials)):
            log_file.write(f"  - {excluded}\n")
        log_file.write("\n")

    print(f"Loaded {len(excluded_trials)} trials to exclude from {exclusion_file}")
    return excluded_trials


def clean_dataframe(df, label):
    """
    Clean a dataframe by ensuring numeric columns are properly converted
    and non-numeric columns are identified.

    Args:
        df (pd.DataFrame): DataFrame to clean
        label (str): Label for logging purposes

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    #   print(f"\nCleaning {label} dataframe...")

    # Define known categorical columns
    categorical_cols = [
        "Subject",
        "Movement",
        "Trial",
        "Camera",
        "Time",
        "time",
        "global_mae_degrees",
        "global_mae_mm",
        "global_mae_weighted",
    ]

    # Identify problematic columns
    for col in df.columns:
        if col not in categorical_cols:
            try:
                # Try to convert to numeric, coerce errors to NaN
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Report if we have NaNs after conversion
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    print(
                        f"  Warning: Column '{col}' has {nan_count} NaN values after numeric conversion"
                    )
            except Exception as e:
                print(f"  Error converting column '{col}': {str(e)}")
                # Keep as is if can't convert
                print(f"  Keeping '{col}' as non-numeric column")

    # Print the final data types
    # print("  DataFrame data types after cleaning:")
    # for col, dtype in df.dtypes.items():
    #     print(f"    {col}: {dtype}")

    return df


def compare_ik_errors(
    mono_results,
    wham_results,
    output_dir=None,
    log_path=None,
    mono_files=None,
    wham_files=None,
):
    """
    Compare IK errors between mono and WHAM results.
    Create visualizations showing the differences for translations and rotations.

    Args:
        mono_results (pd.DataFrame): DataFrame with mono IK results
        wham_results (pd.DataFrame): DataFrame with WHAM IK results
        output_dir (str, optional): Directory to save output files
        log_path (str, optional): Path to log file
        mono_files (list, optional): List of mono file paths
        wham_files (list, optional): List of wham file paths
    """
    print("\nComparing IK errors between mono and WHAM...")

    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Get all DOFs (excluding metadata columns and global metrics)
    dof_comparison = pd.DataFrame(
        columns=["DOF", "Category", "Mono_Mean", "Mono_Std", "WHAM_Mean", "WHAM_Std"]
    )

    # Define columns to exclude
    exclude_cols = [
        "Subject",
        "Movement",
        "Trial",
        "Camera",
        "Time",
        "time",
        "global_mae_degrees",
        "global_mae_mm",
        "global_mae_weighted",
    ]

    # Add keywords to exclude
    exclude_keywords = ["min", "max", "std", "mean"]

    # Find common DOF columns, excluding columns with the specified keywords
    mono_dofs = [
        col
        for col in mono_results.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(mono_results[col])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    wham_dofs = [
        col
        for col in wham_results.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(wham_results[col])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    common_dofs = list(set(mono_dofs).intersection(set(wham_dofs)))

    rotation_dofs = []
    translation_dofs = []

    # Categorize DOFs
    for dof in common_dofs:
        if dof in EXCLUDED_DOFS:
            continue

        if any(keyword in dof for keyword in ["tx", "ty", "tz"]):
            translation_dofs.append(dof)
        else:
            rotation_dofs.append(dof)

    # Process DOFs by category
    for dof in common_dofs:
        if dof in EXCLUDED_DOFS:
            continue

        try:
            category = (
                "Translation"
                if any(keyword in dof for keyword in ["tx", "ty", "tz"])
                else "Rotation"
            )

            # Calculate statistics - rounded to 2 decimal places
            mono_mean = round(mono_results[dof].mean(), 2)
            mono_std = round(mono_results[dof].std(), 2)
            wham_mean = round(wham_results[dof].mean(), 2)
            wham_std = round(wham_results[dof].std(), 2)

            # Add to comparison DataFrame
            dof_comparison = pd.concat(
                [
                    dof_comparison,
                    pd.DataFrame(
                        [
                            {
                                "DOF": dof,
                                "Category": category,
                                "Mono_Mean": mono_mean,
                                "Mono_Std": mono_std,
                                "WHAM_Mean": wham_mean,
                                "WHAM_Std": wham_std,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            # Print the comparison
            print(
                f"  {dof} ({category}): Mono {mono_mean:.2f} ± {mono_std:.2f}, WHAM {wham_mean:.2f} ± {wham_std:.2f}"
            )
        except Exception as e:
            print(f"Error processing DOF {dof}: {str(e)}")

    # Create plots for each category
    for category in ["Translation", "Rotation"]:
        category_dofs = dof_comparison[dof_comparison["Category"] == category]
        if not category_dofs.empty:
            plot_dof_category_comparison(category_dofs, category, output_dir)

    # Save comparison to CSV
    csv_path = os.path.join(output_dir, "dof_comparison.csv")
    dof_comparison.to_csv(csv_path, index=False)
    print(f"DOF comparison saved to {os.path.abspath(csv_path)}")

    # Create global comparison plot
    create_global_comparison_plot(mono_results, wham_results, output_dir)

    # Create movement-based comparison plots
    create_movement_comparison_plots(mono_results, wham_results, output_dir)


def plot_global_metrics(
    mono_mae_degrees, wham_mae_degrees, mono_mae_mm, wham_mae_mm, output_dir=None
):
    """
    Create barplots comparing global metrics between mono and WHAM.

    Args:
        mono_mae_degrees: Global MAE in degrees for mono approach
        wham_mae_degrees: Global MAE in degrees for WHAM approach
        mono_mae_mm: Global MAE in mm for mono approach
        wham_mae_mm: Global MAE in mm for WHAM approach
        output_dir (str, optional): Directory to save output files
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Plot for degrees
    plt.figure(figsize=(8, 6))
    methods = ["Mono", "WHAM"]
    degrees_values = [mono_mae_degrees, wham_mae_degrees]

    bars = plt.bar(methods, degrees_values, color=["#3498db", "#e74c3c"], width=0.6)
    plt.ylabel("Global MAE (degrees)")
    plt.title("Comparison of Global MAE in degrees")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set integer y-ticks only
    ax = plt.gca()
    y_max = max(degrees_values) * 1.2  # Add 20% padding
    ax.set_yticks(range(int(y_max) + 1))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the plot with absolute path
    plot_filename = "global_degrees_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Global metrics comparison plot saved to {os.path.abspath(plot_path)}")

    # Plot for mm
    plt.figure(figsize=(8, 6))
    mm_values = [mono_mae_mm, wham_mae_mm]

    bars = plt.bar(methods, mm_values, color=["#3498db", "#e74c3c"], width=0.6)
    plt.ylabel("Global MAE (mm)")
    plt.title("Comparison of Global MAE in mm")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Set integer y-ticks only
    ax = plt.gca()
    y_max = max(mm_values) * 1.2  # Add 20% padding
    ax.set_yticks(range(int(y_max) + 1))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save the plot with absolute path
    plot_filename = "global_mm_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Global metrics comparison plot saved to {os.path.abspath(plot_path)}")


def plot_dof_category_comparison(dof_comparison, category, output_dir=None):
    """
    Create barplots comparing DOF errors for a specific category.
    Include horizontal lines showing averages across DOFs.

    Args:
        dof_comparison (pd.DataFrame): DataFrame with DOF comparison data for a category
        category (str): Category to plot ('Rotation' or 'Translation')
        output_dir (str, optional): Directory to save output files
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Apply plot_config font settings
    plt.rcParams.update({
        'font.family': pc.FONTS['family'],
        'font.size': pc.FONTS['axis_title_size']
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    dofs = dof_comparison["DOF"].values
    mono_means = dof_comparison["Mono_Mean"].values
    wham_means = dof_comparison["WHAM_Mean"].values

    # Calculate average values across all DOFs - rounded to 2 decimal places
    mono_avg = round(np.mean(mono_means), 2)
    wham_avg = round(np.mean(wham_means), 2)

    x = np.arange(len(dofs))
    width = 0.35

    # Use plot_config colors for method comparison
    mono_color = pc.COLORS['mono']
    wham_color = pc.COLORS['wham']

    # Create bars with unit-appropriate labels
    if category == "Translation":
        mono_bars = ax.bar(
            x - width / 2,
            mono_means,
            width,
            label=f"Mono (avg: {mono_avg:.2f} mm)",
            color=mono_color,
        )
        wham_bars = ax.bar(
            x + width / 2,
            wham_means,
            width,
            label=f"WHAM (avg: {wham_avg:.2f} mm)",
            color=wham_color,
        )
    else:
        mono_bars = ax.bar(
            x - width / 2,
            mono_means,
            width,
            label=f"Mono (avg: {mono_avg:.2f} degrees)",
            color=mono_color,
        )
        wham_bars = ax.bar(
            x + width / 2,
            wham_means,
            width,
            label=f"WHAM (avg: {wham_avg:.2f} degrees)",
            color=wham_color,
        )

    # Add horizontal lines for averages using plot_config line styling
    mono_line = ax.axhline(y=mono_avg, color=mono_color, linestyle=pc.LINES['dash_reference'], alpha=0.7, linewidth=pc.LINES['width'])
    wham_line = ax.axhline(y=wham_avg, color=wham_color, linestyle=pc.LINES['dash_reference'], alpha=0.7, linewidth=pc.LINES['width'])

    # Add details with proper unit labels and plot_config styling
    if category == "Translation":
        ax.set_ylabel("MAE (mm)", fontsize=pc.FONTS['axis_title_size'])
    else:
        ax.set_ylabel("MAE (degrees)", fontsize=pc.FONTS['axis_title_size'])

    # Set title and apply plot_config styling
    ax.set_title(f"{category} DOF Errors: Mono vs WHAM", fontsize=pc.FONTS['title_size'])
    ax.set_xticks(x)
    ax.set_xticklabels(dofs, rotation=45, ha="right", fontsize=pc.FONTS['tick_size'])
    
    # Apply plot_config legend styling
    legend = ax.legend(fontsize=pc.FONTS['legend_size'])
    legend.get_frame().set_facecolor(pc.COLORS['background'])
    legend.get_frame().set_edgecolor('none')
    
    # Apply axis styling from plot_config
    ax.tick_params(labelsize=pc.FONTS['tick_size'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor(pc.COLORS['background'])

    # Add value labels - rounded to 1 decimal place
    for i, v in enumerate(mono_means):
        ax.text(i - width / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
                fontsize=pc.FONTS['annotation_size'], color=pc.COLORS['text'])
    for i, v in enumerate(wham_means):
        ax.text(i + width / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
                fontsize=pc.FONTS['annotation_size'], color=pc.COLORS['text'])

    # Set integer y-ticks only
    y_max = max(max(mono_means), max(wham_means)) * 1.2  # Add 20% padding
    ax.set_yticks(range(int(y_max) + 1))

    # Apply overall figure styling
    fig.patch.set_facecolor(pc.COLORS['background'])
    plt.tight_layout()

    # Save the plot with absolute path
    plot_filename = f"{category.lower()}_dof_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=pc.COLORS['background'])
    plt.close()

    print(f"{category} DOF comparison plot saved to {os.path.abspath(plot_path)}")


def compare_marker_errors(
    mono_results,
    wham_results,
    output_dir=None,
    log_path=None,
    mono_files=None,
    wham_files=None,
):
    """
    Compare marker errors between mono and WHAM results.
    Create visualizations showing the differences by marker category.

    Args:
        mono_results (pd.DataFrame): DataFrame with mono marker results
        wham_results (pd.DataFrame): DataFrame with WHAM marker results
        output_dir (str, optional): Directory to save output files
        log_path (str, optional): Path to log file
        mono_files (list, optional): List of mono file paths
        wham_files (list, optional): List of wham file paths
    """
    print("\nComparing marker errors between mono and WHAM...")

    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Initialize DataFrame for comparison results
    marker_comparison = pd.DataFrame(
        columns=["Marker", "Category", "Mono_Mean", "Mono_Std", "WHAM_Mean", "WHAM_Std"]
    )

    # Find common marker columns
    exclude_cols = [
        "Subject",
        "Movement",
        "Trial",
        "Camera",
        "Time",
        "time",
        "global_mae_degrees",
        "global_mae_mm",
        "global_mae_weighted",
    ]

    # Add keywords to exclude
    exclude_keywords = ["min", "max", "std", "mean"]

    # Find marker columns excluding columns with specified keywords
    mono_markers = [
        col
        for col in mono_results.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(mono_results[col])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    wham_markers = [
        col
        for col in wham_results.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(wham_results[col])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    common_markers = list(set(mono_markers).intersection(set(wham_markers)))

    # Process each marker
    for marker in common_markers:
        try:
            # Determine marker category
            if marker in PELVIS_MARKERS:
                category = "Pelvis"
            elif marker in ANKLE_MARKERS:
                category = "Ankle"
            elif marker in TOE_MARKERS:
                category = "Toe"
            else:
                category = "Other"

            # Calculate statistics - rounded to 2 decimal places
            mono_mean = round(mono_results[marker].mean(), 2)
            mono_std = round(mono_results[marker].std(), 2)
            wham_mean = round(wham_results[marker].mean(), 2)
            wham_std = round(wham_results[marker].std(), 2)

            marker_comparison = pd.concat(
                [
                    marker_comparison,
                    pd.DataFrame(
                        [
                            {
                                "Marker": marker,
                                "Mono_Mean": mono_mean,
                                "Mono_Std": mono_std,
                                "WHAM_Mean": wham_mean,
                                "WHAM_Std": wham_std,
                                "Category": category,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        except Exception as e:
            print(f"Error processing marker {marker}: {str(e)}")

    # Print comparison summary
    print("\nMarker Comparison Summary:")
    for category in marker_comparison["Category"].unique():
        category_markers = marker_comparison[marker_comparison["Category"] == category]
        if not category_markers.empty:
            print(f"\n{category} Markers:")
            for _, row in category_markers.iterrows():
                print(
                    f"  {row['Marker']}: Mono {row['Mono_Mean']:.2f} ± {row['Mono_Std']:.2f}, WHAM {row['WHAM_Mean']:.2f} ± {row['WHAM_Std']:.2f}"
                )

    # Comment out the marker error plot calls
    # plot_global_metrics(mono_mae_mm, wham_mae_mm, None, None, output_dir)

    # Comment out the marker comparison plots
    # plot_combined_marker_comparison(marker_comparison, output_dir)

    # Save comparison to CSV
    csv_path = os.path.join(output_dir, "marker_comparison.csv")
    marker_comparison.to_csv(csv_path, index=False)
    print(f"Marker comparison saved to {os.path.abspath(csv_path)}")


def create_global_comparison_plot(ik_results_mono, ik_results_wham, output_dir=None):
    """
    Create a global comparison plot showing the overall MAE for both degrees and mm.

    Args:
        ik_results_mono (pd.DataFrame): DataFrame with mono IK results
        ik_results_wham (pd.DataFrame): DataFrame with WHAM IK results
        output_dir (str, optional): Directory to save output files
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Extract global metrics - rounded to 2 decimal places
    try:
        mono_mm = round(ik_results_mono["global_mae_mm"].mean(), 2)
        mono_degrees = round(ik_results_mono["global_mae_degrees"].mean(), 2)
        wham_mm = round(ik_results_wham["global_mae_mm"].mean(), 2)
        wham_degrees = round(ik_results_wham["global_mae_degrees"].mean(), 2)
        
        # Calculate standard deviations
        mono_mm_std = round(ik_results_mono["global_mae_mm"].std(), 2)
        mono_degrees_std = round(ik_results_mono["global_mae_degrees"].std(), 2)
        wham_mm_std = round(ik_results_wham["global_mae_mm"].std(), 2)
        wham_degrees_std = round(ik_results_wham["global_mae_degrees"].std(), 2)

        # Create Plotly subplot figure
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Comparison of Global MAE in mm", "Comparison of Global MAE in degrees"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Data for plots
        methods = ["Mono", "WHAM"]
        mm_values = [mono_mm, wham_mm]
        degrees_values = [mono_degrees, wham_degrees]
        mm_stds = [mono_mm_std, wham_mm_std]
        degrees_stds = [mono_degrees_std, wham_degrees_std]

        # Use plot_config colors for method comparison
        colors = [pc.COLORS['mono'], pc.COLORS['wham']]

        # Add mm plot with error bars
        fig.add_trace(
            go.Bar(
                x=methods,
                y=mm_values,
                error_y=dict(type='data', array=mm_stds, visible=True),
                marker_color=colors,
                name="MAE (mm)",
                text=[f"{val:.1f}" for val in mm_values],
                textposition='outside',
                textfont=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
                showlegend=False
            ),
            row=1, col=1
        )

        # Add degrees plot with error bars
        fig.add_trace(
            go.Bar(
                x=methods,
                y=degrees_values,
                error_y=dict(type='data', array=degrees_stds, visible=True),
                marker_color=colors,
                name="MAE (degrees)",
                text=[f"{val:.1f}" for val in degrees_values],
                textposition='outside',
                textfont=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
                showlegend=False
            ),
            row=1, col=2
        )

        # Add annotations with offset to avoid overlap with error bars
        for i, (method, val, std) in enumerate(zip(methods, mm_values, mm_stds)):
            fig.add_annotation(
                x=method,
                y=val + std + 0.5,  # Offset above error bar
                text=f"{val:.1f}",
                showarrow=False,
                font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
                xref="x1", yref="y1"
            )

        for i, (method, val, std) in enumerate(zip(methods, degrees_values, degrees_stds)):
            fig.add_annotation(
                x=method,
                y=val + std + 0.1,  # Offset above error bar
                text=f"{val:.1f}",
                showarrow=False,
                font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
                xref="x2", yref="y2"
            )

        # Remove text from bars since we're using annotations
        fig.update_traces(text="", row=1, col=1)
        fig.update_traces(text="", row=1, col=2)

        # Apply plot_config styling
        fig.update_layout(
            **pc.get_standard_layout(),
            width=pc.LAYOUT['width'] * 2,  # Wider for two subplots
            height=pc.LAYOUT['height']
        )

        # Update axes
        fig.update_xaxes(title_text="Method", row=1, col=1, **pc.get_standard_axes())
        fig.update_xaxes(title_text="Method", row=1, col=2, **pc.get_standard_axes())
        fig.update_yaxes(title_text="Global MAE (mm)", row=1, col=1, **pc.get_standard_axes())
        fig.update_yaxes(title_text="Global MAE (degrees)", row=1, col=2, **pc.get_standard_axes())

        # Save the plot
        plot_filename = "global_comparison.html"
        plot_path = os.path.join(output_dir, plot_filename)
        fig.write_html(plot_path)
        
        # Also save as SVG for publication
        plot_filename_svg = "global_comparison.svg"
        plot_path_svg = os.path.join(output_dir, plot_filename_svg)
        fig.write_image(plot_path_svg, format='svg', width=pc.LAYOUT['width'] * 2, height=pc.LAYOUT['height'])

        print(f"Global comparison plot saved to {os.path.abspath(plot_path)}")
    except Exception as e:
        print(f"Error creating global comparison plot: {str(e)}")


def check_for_data_issues(df, name, log_path):
    """Check for potential issues in the dataframe and log them."""
    with open(log_path, "a") as log_file:
        log_file.write(f"\n=== Data Quality Check for {name} ===\n\n")

        # Check for duplicate entries
        dupes = df.duplicated(subset=["Subject", "Camera", "Trial"]).sum()
        log_file.write(f"Duplicate entries: {dupes}\n")

        # Check for missing values in key columns
        for col in ["Subject", "Camera", "Trial"]:
            if col in df.columns:
                missing = df[col].isna().sum()
                log_file.write(f"Missing values in {col}: {missing}\n")

        # Check for data type issues
        for col in df.columns:
            log_file.write(f"Column {col}: Type {df[col].dtype}\n")

            # Sample of unique values
            unique_vals = df[col].unique()[:5]
            log_file.write(f"  Sample values: {unique_vals}\n")

        log_file.write("\n")


def plot_combined_marker_comparison(marker_comparison, output_dir=None):
    """
    Create a combined figure with all marker category comparisons in subplots.

    Args:
        marker_comparison (pd.DataFrame): DataFrame with marker comparison data
        output_dir (str, optional): Directory to save output files
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()

    # Create figure with 3 subplots (1 row, 3 columns)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    # Track global y max for consistent scaling
    y_max = 0

    # Categories to plot
    categories = ["Pelvis", "Ankle", "Toe"]

    # Process each category
    for i, category in enumerate(categories):
        # Get data for this category
        category_markers = marker_comparison[marker_comparison["Category"] == category]

        if category_markers.empty:
            # If no data for this category, display empty plot with message
            axes[i].text(
                0.5,
                0.5,
                f"No {category} marker data",
                ha="center",
                va="center",
                fontsize=12,
            )
            axes[i].set_title(f"{category} Marker Errors")
            continue

        markers = category_markers["Marker"].values
        mono_means = category_markers["Mono_Mean"].values
        wham_means = category_markers["WHAM_Mean"].values

        # Update global y max
        y_max = max(
            y_max,
            max(
                mono_means.max() if len(mono_means) > 0 else 0,
                wham_means.max() if len(wham_means) > 0 else 0,
            ),
        )

        # Calculate average values - rounded to 2 decimal places (for reference only)
        mono_avg = round(np.mean(mono_means), 2)
        wham_avg = round(np.mean(wham_means), 2)

        x = np.arange(len(markers))
        width = 0.35

        # Create bars in the appropriate subplot - no avg values in any legends
        mono_bars = axes[i].bar(
            x - width / 2, mono_means, width, label="Mono", color="#3498db"
        )
        wham_bars = axes[i].bar(
            x + width / 2, wham_means, width, label="WHAM", color="#e74c3c"
        )

        # Add subplot details
        axes[i].set_title(f"{category} Marker Errors")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(markers, rotation=45, ha="right")
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels - rounded to 2 decimal places
        for j, v in enumerate(mono_means):
            axes[i].text(
                j - width / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9
            )
        for j, v in enumerate(wham_means):
            axes[i].text(
                j + width / 2, v + 0.1, f"{v:.2f}", ha="center", va="bottom", fontsize=9
            )

        # Add legend to each subplot
        axes[i].legend()

    # Add common y-axis label
    fig.text(0.04, 0.5, "MAE (mm)", va="center", rotation="vertical", fontsize=12)

    # Set consistent y-axis limits with some padding
    for ax in axes:
        ax.set_ylim(0, y_max * 1.2)  # 20% padding above max value

    # Add figure title
    fig.suptitle("Marker Error Comparison: Mono vs WHAM", fontsize=16)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])  # Adjust for common y-label

    # Save the combined plot
    plot_filename = "combined_marker_comparison.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Combined marker comparison plot saved to {os.path.abspath(plot_path)}")


def clean_output_directory(output_dir):
    """
    Delete all non-Python files in the output directory.

    Args:
        output_dir (str): Path to the output directory
    """
    print(f"Cleaning output directory: {os.path.abspath(output_dir)}...")

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created new output directory: {os.path.abspath(output_dir)}")
        return

    # Delete files, preserving python files
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)

        # Skip python files
        if item.endswith(".py"):
            continue

        if os.path.isfile(item_path):
            os.remove(item_path)
            print(f"Deleted file: {item}")
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Deleted directory: {item}")

    print("Output directory cleaned successfully")


def create_movement_comparison_plots(mono_results, wham_results, output_dir=None):
    """
    Create comparison plots showing errors grouped by movement type (squat, STS, walking).
    Generate two plots: one for rotation errors and one for translation errors.

    Args:
        mono_results (pd.DataFrame): DataFrame with mono IK results
        wham_results (pd.DataFrame): DataFrame with WHAM IK results
        output_dir (str, optional): Directory to save output files
    """
    # Use current directory if no output dir specified
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Apply plot_config font settings
    plt.rcParams.update({
        'font.family': pc.FONTS['family'],
        'font.size': pc.FONTS['axis_title_size']
    })

    # Define columns to exclude
    exclude_cols = [
        "Subject",
        "Movement",
        "Trial",
        "Camera",
        "Time",
        "time",
        "global_mae_degrees",
        "global_mae_mm",
        "global_mae_weighted",
    ]

    # Add keywords to exclude
    exclude_keywords = ["min", "max", "std", "mean"]

    # Find common DOF columns, excluding columns with the specified keywords
    mono_dofs = [
        col
        for col in mono_results.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(mono_results[col])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    wham_dofs = [
        col
        for col in wham_results.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(wham_results[col])
        and not any(keyword in col.lower() for keyword in exclude_keywords)
    ]

    common_dofs = list(set(mono_dofs).intersection(set(wham_dofs)))

    # Categorize DOFs
    rotation_dofs = []
    translation_dofs = []

    for dof in common_dofs:
        if dof in EXCLUDED_DOFS:
            continue

        if any(keyword in dof for keyword in ["tx", "ty", "tz"]):
            translation_dofs.append(dof)
        else:
            rotation_dofs.append(dof)

    # Get unique movement types
    movements = sorted(
        list(
            set(mono_results["Movement"].unique())
            | set(wham_results["Movement"].unique())
        )
    )

    # Prepare data for rotation errors by movement
    rotation_data = {"Movement": [], "Method": [], "Error": []}

    # Prepare data for translation errors by movement
    translation_data = {"Movement": [], "Method": [], "Error": []}

    # Calculate average errors for each movement and method
    for movement in movements:
        # Filter data for this movement
        mono_movement = mono_results[mono_results["Movement"] == movement]
        wham_movement = wham_results[wham_results["Movement"] == movement]

        if not mono_movement.empty and not wham_movement.empty:
            # Calculate rotation errors
            mono_rot_mean = round(mono_movement[rotation_dofs].mean().mean(), 2)
            wham_rot_mean = round(wham_movement[rotation_dofs].mean().mean(), 2)

            # Add rotation data
            rotation_data["Movement"].append(movement)
            rotation_data["Method"].append("Mono")
            rotation_data["Error"].append(mono_rot_mean)

            rotation_data["Movement"].append(movement)
            rotation_data["Method"].append("WHAM")
            rotation_data["Error"].append(wham_rot_mean)

            # Calculate translation errors
            mono_trans_mean = round(mono_movement[translation_dofs].mean().mean(), 2)
            wham_trans_mean = round(wham_movement[translation_dofs].mean().mean(), 2)

            # Add translation data
            translation_data["Movement"].append(movement)
            translation_data["Method"].append("Mono")
            translation_data["Error"].append(mono_trans_mean)

            translation_data["Movement"].append(movement)
            translation_data["Method"].append("WHAM")
            translation_data["Error"].append(wham_trans_mean)

    # Create rotation error plot
    if rotation_data["Movement"]:
        rotation_df = pd.DataFrame(rotation_data)

        plt.figure(figsize=(10, 6))

        # Create grouped bar chart
        x = np.arange(len(movements))
        width = 0.35

        # Get data for each method
        mono_rot_data = [
            (
                rotation_df[
                    (rotation_df["Movement"] == m) & (rotation_df["Method"] == "Mono")
                ]["Error"].values[0]
                if not rotation_df[
                    (rotation_df["Movement"] == m) & (rotation_df["Method"] == "Mono")
                ].empty
                else 0
            )
            for m in movements
        ]

        wham_rot_data = [
            (
                rotation_df[
                    (rotation_df["Movement"] == m) & (rotation_df["Method"] == "WHAM")
                ]["Error"].values[0]
                if not rotation_df[
                    (rotation_df["Movement"] == m) & (rotation_df["Method"] == "WHAM")
                ].empty
                else 0
            )
            for m in movements
        ]

        # Create bars using plot_config colors
        mono_bars = plt.bar(
            x - width / 2, mono_rot_data, width, label="Mono", color=pc.COLORS['mono']
        )
        wham_bars = plt.bar(
            x + width / 2, wham_rot_data, width, label="WHAM", color=pc.COLORS['wham']
        )

        # Add details with plot_config styling
        plt.ylabel("Average Rotation Error (degrees)", fontsize=pc.FONTS['axis_title_size'])
        plt.title("Rotation Errors by Movement Type: Mono vs WHAM", fontsize=pc.FONTS['title_size'])
        plt.xticks(x, movements, fontsize=pc.FONTS['tick_size'])
        
        # Apply plot_config legend styling
        legend = plt.legend(fontsize=pc.FONTS['legend_size'])
        legend.get_frame().set_facecolor(pc.COLORS['background'])
        legend.get_frame().set_edgecolor('none')
        
        # Apply axis styling from plot_config
        plt.gca().tick_params(labelsize=pc.FONTS['tick_size'])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_facecolor(pc.COLORS['background'])

        # Add value labels with plot_config styling
        for i, v in enumerate(mono_rot_data):
            plt.text(i - width / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
                    fontsize=pc.FONTS['annotation_size'], color=pc.COLORS['text'])
        for i, v in enumerate(wham_rot_data):
            plt.text(i + width / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
                    fontsize=pc.FONTS['annotation_size'], color=pc.COLORS['text'])

        # Apply overall figure styling
        plt.gcf().patch.set_facecolor(pc.COLORS['background'])
        plt.tight_layout()

        # Save the plot
        plot_filename = "rotation_errors_by_movement.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=pc.COLORS['background'])
        plt.close()

        print(f"Rotation errors by movement plot saved to {os.path.abspath(plot_path)}")

    # Create translation error plot
    if translation_data["Movement"]:
        translation_df = pd.DataFrame(translation_data)

        plt.figure(figsize=(10, 6))

        # Create grouped bar chart
        x = np.arange(len(movements))
        width = 0.35

        # Get data for each method
        mono_trans_data = [
            (
                translation_df[
                    (translation_df["Movement"] == m)
                    & (translation_df["Method"] == "Mono")
                ]["Error"].values[0]
                if not translation_df[
                    (translation_df["Movement"] == m)
                    & (translation_df["Method"] == "Mono")
                ].empty
                else 0
            )
            for m in movements
        ]

        wham_trans_data = [
            (
                translation_df[
                    (translation_df["Movement"] == m)
                    & (translation_df["Method"] == "WHAM")
                ]["Error"].values[0]
                if not translation_df[
                    (translation_df["Movement"] == m)
                    & (translation_df["Method"] == "WHAM")
                ].empty
                else 0
            )
            for m in movements
        ]

        # Create bars using plot_config colors
        mono_bars = plt.bar(
            x - width / 2, mono_trans_data, width, label="Mono", color=pc.COLORS['mono']
        )
        wham_bars = plt.bar(
            x + width / 2, wham_trans_data, width, label="WHAM", color=pc.COLORS['wham']
        )

        # Add details with plot_config styling
        plt.ylabel("Average Translation Error (mm)", fontsize=pc.FONTS['axis_title_size'])
        plt.title("Translation Errors by Movement Type: Mono vs WHAM", fontsize=pc.FONTS['title_size'])
        plt.xticks(x, movements, fontsize=pc.FONTS['tick_size'])
        
        # Apply plot_config legend styling
        legend = plt.legend(fontsize=pc.FONTS['legend_size'])
        legend.get_frame().set_facecolor(pc.COLORS['background'])
        legend.get_frame().set_edgecolor('none')
        
        # Apply axis styling from plot_config
        plt.gca().tick_params(labelsize=pc.FONTS['tick_size'])
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_facecolor(pc.COLORS['background'])

        # Add value labels with plot_config styling
        for i, v in enumerate(mono_trans_data):
            plt.text(i - width / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
                    fontsize=pc.FONTS['annotation_size'], color=pc.COLORS['text'])
        for i, v in enumerate(wham_trans_data):
            plt.text(i + width / 2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
                    fontsize=pc.FONTS['annotation_size'], color=pc.COLORS['text'])

        # Apply overall figure styling
        plt.gcf().patch.set_facecolor(pc.COLORS['background'])
        plt.tight_layout()

        # Save the plot
        plot_filename = "translation_errors_by_movement.png"
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor=pc.COLORS['background'])
        plt.close()

        print(
            f"Translation errors by movement plot saved to {os.path.abspath(plot_path)}"
        )


def compare_ik_errors_three_methods(
    mono_results,
    wham_results,
    twocam_results,
    output_dir=None,
    log_path=None,
    mono_files=None,
    wham_files=None,
    twocam_files=None,
):
    # Check for required columns and add defaults if missing
    for df, name in [
        (mono_results, "Mono"),
        (wham_results, "WHAM"),
        (twocam_results, "2CAMS"),
    ]:
        if "global_mae_degrees" not in df.columns:
            print(
                f"Warning: global_mae_degrees column missing from {name} results. Using average of DOF values."
            )
            df["global_mae_degrees"] = df.filter(
                regex="^(?!.*(_min|_max|_mean|_std))"
            ).mean(axis=1)

        if "global_mae_mm" not in df.columns:
            print(
                f"Warning: global_mae_mm column missing from {name} results. Using average of translation values."
            )
            df["global_mae_mm"] = (
                df.filter(regex="^(pelvis_t[xyz])$").mean(axis=1) * 1000
            )  # Convert to mm

    # Now proceed with the rest of the function...
    print("\nComparing IK Errors for all three methods...")

    if log_path:
        with open(log_path, "a") as log_file:
            log_file.write("Comparing IK Errors for all three methods...\n")
            log_file.write(f"  Mono data: {len(mono_results)} rows\n")
            log_file.write(f"  WHAM data: {len(wham_results)} rows\n")
            log_file.write(f"  2CAMS data: {len(twocam_results)} rows\n\n")

    # Global metrics
    mono_mae_degrees = mono_results["global_mae_degrees"].mean()
    wham_mae_degrees = wham_results["global_mae_degrees"].mean()
    twocam_mae_degrees = twocam_results["global_mae_degrees"].mean()

    mono_mae_mm = mono_results["global_mae_mm"].mean()
    wham_mae_mm = wham_results["global_mae_mm"].mean()
    twocam_mae_mm = twocam_results["global_mae_mm"].mean()

    # Global standard deviations
    mono_mae_degrees_std = mono_results["global_mae_degrees"].std()
    wham_mae_degrees_std = wham_results["global_mae_degrees"].std()
    twocam_mae_degrees_std = twocam_results["global_mae_degrees"].std()

    mono_mae_mm_std = mono_results["global_mae_mm"].std()
    wham_mae_mm_std = wham_results["global_mae_mm"].std()
    twocam_mae_mm_std = twocam_results["global_mae_mm"].std()

    print(
        f"Global MAE (degrees): Mono: {mono_mae_degrees:.2f}, WHAM: {wham_mae_degrees:.2f}, 2CAMS: {twocam_mae_degrees:.2f}"
    )
    print(
        f"Global MAE (mm): Mono: {mono_mae_mm:.2f}, WHAM: {wham_mae_mm:.2f}, 2CAMS: {twocam_mae_mm:.2f}"
    )

    if log_path:
        with open(log_path, "a") as log_file:
            log_file.write(
                f"Global MAE (degrees): Mono: {mono_mae_degrees:.2f}, WHAM: {wham_mae_degrees:.2f}, 2CAMS: {twocam_mae_degrees:.2f}\n"
            )
            log_file.write(
                f"Global MAE (mm): Mono: {mono_mae_mm:.2f}, WHAM: {wham_mae_mm:.2f}, 2CAMS: {twocam_mae_mm:.2f}\n\n"
            )

    # Plot global metrics
    plot_global_metrics_three_methods(
        mono_mae_degrees,
        wham_mae_degrees,
        twocam_mae_degrees,
        mono_mae_mm,
        wham_mae_mm,
        twocam_mae_mm,
        mono_mae_degrees_std,
        wham_mae_degrees_std,
        twocam_mae_degrees_std,
        mono_mae_mm_std,
        wham_mae_mm_std,
        twocam_mae_mm_std,
        output_dir,
    )

    # Generate detailed DOF comparison
    dof_categories = {
        "pelvis_position": ["pelvis_tx", "pelvis_ty", "pelvis_tz"],
        "pelvis_orientation": ["pelvis_tilt", "pelvis_list", "pelvis_rotation"],
        "hip": [
            "hip_flexion_l",
            "hip_flexion_r",
            "hip_adduction_l",
            "hip_adduction_r",
            "hip_rotation_l",
            "hip_rotation_r",
        ],
        "knee": ["knee_angle_l", "knee_angle_r"],
        "ankle": ["ankle_angle_l", "ankle_angle_r", "subtalar_angle_l", "subtalar_angle_r"],
        "lumbar": ["lumbar_extension", "lumbar_bending", "lumbar_rotation"],
    }

    # Create a DataFrame to store results
    dof_comparison = pd.DataFrame()

    for category, dofs in dof_categories.items():
        category_mono = []
        category_wham = []
        category_twocam = []

        for dof in dofs:
            # Check if DOF exists in all datasets
            if (
                dof in mono_results.columns
                and dof in wham_results.columns
                and dof in twocam_results.columns
            ):
                mono_mean = mono_results[dof].mean()
                wham_mean = wham_results[dof].mean()
                twocam_mean = twocam_results[dof].mean()

                # Add to the comparison DataFrame
                dof_comparison = pd.concat(
                    [
                        dof_comparison,
                        pd.DataFrame(
                            {
                                "Category": [category],
                                "DOF": [dof],
                                "Mono": [mono_mean],
                                "WHAM": [wham_mean],
                                "2CAMS": [twocam_mean],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

                category_mono.append(mono_mean)
                category_wham.append(wham_mean)
                category_twocam.append(twocam_mean)

        # Calculate category means
        if category_mono and category_wham and category_twocam:
            category_mono_mean = sum(category_mono) / len(category_mono)
            category_wham_mean = sum(category_wham) / len(category_wham)
            category_twocam_mean = sum(category_twocam) / len(category_twocam)

            # Add category mean to the comparison DataFrame
            dof_comparison = pd.concat(
                [
                    dof_comparison,
                    pd.DataFrame(
                        {
                            "Category": [category],
                            "DOF": ["MEAN"],
                            "Mono": [category_mono_mean],
                            "WHAM": [category_wham_mean],
                            "2CAMS": [category_twocam_mean],
                        }
                    ),
                ],
                ignore_index=True,
            )

            # Plot category comparison
            plot_dof_category_comparison_three_methods(
                dof_comparison, category, output_dir
            )

    # Create a combined plot with all rotation DOFs on x-axis
    plot_all_rotation_dofs_three_methods(
        dof_comparison, mono_results, wham_results, twocam_results, output_dir
    )

    # Create a movement-based comparison
    # create_movement_comparison_plots_three_methods(
    #     mono_results, wham_results, twocam_results, output_dir
    # )

    # Create comprehensive global comparison plot
    create_global_comparison_plot_three_methods(
        mono_results, wham_results, twocam_results, output_dir
    )

    return dof_comparison


def plot_global_metrics_three_methods(
    mono_mae_degrees,
    wham_mae_degrees,
    twocam_mae_degrees,
    mono_mae_mm,
    wham_mae_mm,
    twocam_mae_mm,
    mono_mae_degrees_std,
    wham_mae_degrees_std,
    twocam_mae_degrees_std,
    mono_mae_mm_std,
    wham_mae_mm_std,
    twocam_mae_mm_std,
    output_dir=None,
):
    """Plot global metrics comparing all three methods."""
    # Apply plot_config font settings
    plt.rcParams.update({
        'font.family': pc.FONTS['family'],
        'font.size': 30  # Keep larger font for this specific plot
    })

    plt.figure(figsize=(12, 9))

    # Set up data for plots - maintain consistent order: WHAM, Mono, 2CAMS
    methods = ["WHAM", "OpenCap Monocular", "OpenCap Two-camera"]
    degrees_values = [wham_mae_degrees, mono_mae_degrees, twocam_mae_degrees]
    mm_values = [wham_mae_mm, mono_mae_mm, twocam_mae_mm]
    degrees_stds = [wham_mae_degrees_std, mono_mae_degrees_std, twocam_mae_degrees_std]
    mm_stds = [wham_mae_mm_std, mono_mae_mm_std, twocam_mae_mm_std]

    # Use plot_config colors for all three methods
    colors = [pc.COLORS['wham'], pc.COLORS['mono'], pc.COLORS['twocam']]  # WHAM, OpenCap Monocular, OpenCap Two-camera

    # Create subplot for degrees with error bars
    ax1 = plt.subplot(1, 2, 1)
    bars = plt.bar(methods, degrees_values, color=colors,
                  yerr=degrees_stds, capsize=5, error_kw={'color': 'black', 'linewidth': 1})

    # Set integer y-ticks only for degrees plot (account for error bars)
    y_max1 = max([degrees_values[i] + degrees_stds[i] for i in range(len(degrees_values))]) * 1.2  # Add 20% padding
    ax1.set_yticks(range(int(y_max1) + 1))

    # Add value labels on top of bars
    for bar, value in zip(bars, degrees_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            color=pc.COLORS['text']
        )

    # Apply plot_config styling to remove grid and spines
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_facecolor(pc.COLORS['background'])

    # Create subplot for mm with error bars
    ax2 = plt.subplot(1, 2, 2)
    bars = plt.bar(methods, mm_values, color=colors,
                  yerr=mm_stds, capsize=5, error_kw={'color': 'black', 'linewidth': 1})

    # Set integer y-ticks only for mm plot (account for error bars)
    y_max2 = max([mm_values[i] + mm_stds[i] for i in range(len(mm_values))]) * 1.2  # Add 20% padding
    ax2.set_yticks(range(int(y_max2) + 1))

    # Add value labels on top of bars
    for bar, value in zip(bars, mm_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            color=pc.COLORS['text']
        )

    # Apply plot_config styling to remove grid and spines
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_facecolor(pc.COLORS['background'])

    # Apply overall figure styling
    plt.gcf().patch.set_facecolor(pc.COLORS['background'])
    plt.tight_layout()

    if output_dir:
        plt.savefig(
            os.path.join(output_dir, "global_metrics_comparison_all_methods.png"),
            dpi=300,
            bbox_inches="tight",
            facecolor=pc.COLORS['background']
        )

    plt.close()


def plot_dof_category_comparison_three_methods(
    dof_comparison, category, output_dir=None
):
    """Plot DOF category comparison for all three methods."""
    # Filter for the specific category and remove MEAN row
    category_data = dof_comparison[
        (dof_comparison["Category"] == category) & (dof_comparison["DOF"] != "MEAN")
    ]

    if len(category_data) == 0:
        return

    # Apply plot_config font settings
    plt.rcParams.update({
        'font.family': pc.FONTS['family'],
        'font.size': 30  # Keep larger font for this specific plot
    })

    # Extract DOFs and values
    dofs = category_data["DOF"].tolist()
    mono_values = category_data["Mono"].tolist()
    wham_values = category_data["WHAM"].tolist()
    twocam_values = category_data["2CAMS"].tolist()

    # Create figure
    plt.figure(figsize=(12, 9))

    # Set width of bars
    bar_width = 0.25
    positions = np.arange(len(dofs))

    # Determine if this is a translation or rotation category
    is_translation = category == "pelvis_position" or any("tx" in dof or "ty" in dof or "tz" in dof for dof in dofs)
    ylabel = "MAE (mm)" if is_translation else "MAE (degrees)"
    
    # Create bars using plot_config colors
    wham_bars = plt.bar(positions - bar_width, wham_values, bar_width, color=pc.COLORS['wham'], label="WHAM")
    mono_bars = plt.bar(positions, mono_values, bar_width, color=pc.COLORS['mono'], label="OpenCap Monocular")
    twocam_bars = plt.bar(
        positions + bar_width, twocam_values, bar_width, color=pc.COLORS['twocam'], label="OpenCap Two-camera"
    )
    
    # Apply plot_config legend styling
    legend = plt.legend(fontsize=pc.FONTS['legend_size'])
    legend.get_frame().set_facecolor(pc.COLORS['background'])
    legend.get_frame().set_edgecolor('none')

    # Add value labels on top of bars
    for i, v in enumerate(wham_values):
        plt.text(
            positions[i] - bar_width,
            v + 0.1,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=pc.FONTS['annotation_size'],
            color=pc.COLORS['text']
        )
    for i, v in enumerate(mono_values):
        plt.text(
            positions[i], v + 0.1, f"{v:.1f}", ha="center", va="bottom", 
            fontsize=pc.FONTS['annotation_size'],
            color=pc.COLORS['text']
        )
    for i, v in enumerate(twocam_values):
        plt.text(
            positions[i] + bar_width,
            v + 0.1,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=pc.FONTS['annotation_size'],
            color=pc.COLORS['text']
        )

    # Get current axis and apply plot_config styling
    ax = plt.gca()

    # Add y-axis label
    ax.set_ylabel(ylabel, fontsize=pc.FONTS['axis_title_size'])
    
    # Add title
    ax.set_title(f"{category.replace('_', ' ').title()} DOF Errors: All Methods", fontsize=pc.FONTS['title_size'])

    # Apply plot_config styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(pc.COLORS['background'])
    ax.tick_params(labelsize=pc.FONTS['tick_size'])

    plt.xticks(positions, dofs, rotation=45, ha="right", fontsize=pc.FONTS['tick_size'], color=pc.COLORS['text'])

    # Set integer y-ticks only
    y_max = (
        max(max(wham_values), max(mono_values), max(twocam_values)) * 1.2
    )  # Add 20% padding
    ax.set_yticks(range(int(y_max) + 1))

    # Apply overall figure styling
    plt.gcf().patch.set_facecolor(pc.COLORS['background'])
    plt.tight_layout()

    if output_dir:
        # Save as SVG
        svg_path = os.path.join(output_dir, f"{category}_comparison_all_methods.svg")
        plt.savefig(
            svg_path,
            format='svg',
            bbox_inches="tight",
            facecolor=pc.COLORS['background']
        )
        print(f"{category} DOF comparison plot (SVG) saved to {os.path.abspath(svg_path)}")
        
        # Also save as PNG for compatibility
        png_path = os.path.join(output_dir, f"{category}_comparison_all_methods.png")
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=pc.COLORS['background']
        )
        print(f"{category} DOF comparison plot (PNG) saved to {os.path.abspath(png_path)}")

    plt.close()


def plot_all_rotation_dofs_three_methods(
    dof_comparison, mono_results, wham_results, twocam_results, output_dir=None
):
    """
    Create a single plot with all rotation DOFs on the x-axis showing MAE in degrees with error bars.
    
    Args:
        dof_comparison (pd.DataFrame): DataFrame with DOF comparison data
        mono_results (pd.DataFrame): DataFrame with mono IK results
        wham_results (pd.DataFrame): DataFrame with WHAM IK results
        twocam_results (pd.DataFrame): DataFrame with 2CAMS IK results
        output_dir (str, optional): Directory to save output files
    """
    # Filter for rotation DOFs (exclude translation DOFs and MEAN rows)
    rotation_categories = ["pelvis_orientation", "hip", "knee", "ankle", "lumbar"]
    rotation_data = dof_comparison[
        (dof_comparison["Category"].isin(rotation_categories)) & 
        (dof_comparison["DOF"] != "MEAN")
    ]
    
    if len(rotation_data) == 0:
        print("No rotation DOF data found for combined plot")
        return
    
    # Sort by category and DOF for consistent ordering
    category_order = ["pelvis_orientation", "hip", "knee", "ankle", "lumbar"]
    rotation_data["Category_Order"] = rotation_data["Category"].apply(
        lambda x: category_order.index(x) if x in category_order else 999
    )
    rotation_data = rotation_data.sort_values(["Category_Order", "DOF"]).reset_index(drop=True)
    
    # Apply plot_config font settings
    plt.rcParams.update({
        'font.family': pc.FONTS['family'],
        'font.size': pc.FONTS['axis_title_size']
    })
    
    # Extract DOFs and mean values
    dofs = rotation_data["DOF"].tolist()
    mono_values = rotation_data["Mono"].tolist()
    wham_values = rotation_data["WHAM"].tolist()
    twocam_values = rotation_data["2CAMS"].tolist()
    
    # Calculate standard deviations from original data
    mono_stds = []
    wham_stds = []
    twocam_stds = []
    
    for dof in dofs:
        if dof in mono_results.columns:
            mono_stds.append(mono_results[dof].std())
        else:
            mono_stds.append(0)
        
        if dof in wham_results.columns:
            wham_stds.append(wham_results[dof].std())
        else:
            wham_stds.append(0)
        
        if dof in twocam_results.columns:
            twocam_stds.append(twocam_results[dof].std())
        else:
            twocam_stds.append(0)
    
    # Create figure with wider width to accommodate all DOFs
    num_dofs = len(dofs)
    fig_width = max(16, num_dofs * 0.6)  # Adjust width based on number of DOFs
    plt.figure(figsize=(fig_width, 8))
    
    # Set width of bars
    bar_width = 0.25
    positions = np.arange(len(dofs))
    
    # Create bars with error bars using plot_config colors
    wham_bars = plt.bar(
        positions - bar_width, wham_values, bar_width,
        yerr=wham_stds,
        color=pc.COLORS['wham'],
        label="WHAM",
        capsize=3,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': 'black'}
    )
    mono_bars = plt.bar(
        positions, mono_values, bar_width,
        yerr=mono_stds,
        color=pc.COLORS['mono'],
        label="OpenCap Monocular",
        capsize=3,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': 'black'}
    )
    twocam_bars = plt.bar(
        positions + bar_width, twocam_values, bar_width,
        yerr=twocam_stds,
        color=pc.COLORS['twocam'],
        label="OpenCap Two-camera",
        capsize=3,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'color': 'black'}
    )
    
    # Apply plot_config legend styling
    legend = plt.legend(fontsize=pc.FONTS['legend_size'])
    legend.get_frame().set_facecolor(pc.COLORS['background'])
    legend.get_frame().set_edgecolor('none')
    
    # Add value labels on top of bars (only if there's space, otherwise skip for readability)
    # Position labels above error bars
    if num_dofs <= 20:  # Only add labels if not too many DOFs
        for i, (v, std) in enumerate(zip(wham_values, wham_stds)):
            plt.text(
                positions[i] - bar_width,
                v + std + 0.2,  # Position above error bar
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=pc.FONTS['annotation_size'],
                color=pc.COLORS['text']
            )
        for i, (v, std) in enumerate(zip(mono_values, mono_stds)):
            plt.text(
                positions[i], v + std + 0.2, f"{v:.1f}", ha="center", va="bottom", 
                fontsize=pc.FONTS['annotation_size'],
                color=pc.COLORS['text']
            )
        for i, (v, std) in enumerate(zip(twocam_values, twocam_stds)):
            plt.text(
                positions[i] + bar_width,
                v + std + 0.2,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=pc.FONTS['annotation_size'],
                color=pc.COLORS['text']
            )
    
    # Get current axis and apply plot_config styling
    ax = plt.gca()
    
    # Add y-axis label
    ax.set_ylabel("MAE (degrees)", fontsize=pc.FONTS['axis_title_size'])
    
    # Add title
    ax.set_title("Rotation DOF Errors: All Methods", fontsize=pc.FONTS['title_size'])
    
    # Apply plot_config styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor(pc.COLORS['background'])
    ax.tick_params(labelsize=pc.FONTS['tick_size'])
    
    # Set x-axis ticks with rotation for readability
    plt.xticks(positions, dofs, rotation=45, ha="right", 
              fontsize=pc.FONTS['tick_size'], color=pc.COLORS['text'])
    
    # Set integer y-ticks only (account for error bars)
    y_max = max(
        max([v + s for v, s in zip(wham_values, wham_stds)]),
        max([v + s for v, s in zip(mono_values, mono_stds)]),
        max([v + s for v, s in zip(twocam_values, twocam_stds)])
    ) * 1.2  # Add 20% padding
    ax.set_yticks(range(int(y_max) + 1))
    
    # Apply overall figure styling
    plt.gcf().patch.set_facecolor(pc.COLORS['background'])
    plt.tight_layout()
    
    if output_dir:
        # Save as SVG
        svg_path = os.path.join(output_dir, "all_rotation_dofs_comparison.svg")
        plt.savefig(
            svg_path,
            format='svg',
            bbox_inches="tight",
            facecolor=pc.COLORS['background']
        )
        print(f"All rotation DOFs comparison plot (SVG) saved to {os.path.abspath(svg_path)}")
        
        # Also save as PNG for compatibility
        png_path = os.path.join(output_dir, "all_rotation_dofs_comparison.png")
        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=pc.COLORS['background']
        )
        print(f"All rotation DOFs comparison plot (PNG) saved to {os.path.abspath(png_path)}")
    
    plt.close()


def create_global_comparison_plot_three_methods(
    ik_results_mono, ik_results_wham, ik_results_twocam, output_dir=None
):
    """Create separate global comparison plots for degrees and mm using Plotly."""
    from plotly.subplots import make_subplots

    # Calculate global metrics per movement
    # Use lowercase to match standardized movement names in data
    movements = ["walking", "squat", "STS"]

    # Prepare data structure for plotting
    plot_data = {
        "movement": [],
        "mono_degrees": [],
        "wham_degrees": [],
        "twocam_degrees": [],
        "mono_mm": [],
        "wham_mm": [],
        "twocam_mm": [],
        "mono_degrees_std": [],
        "wham_degrees_std": [],
        "twocam_degrees_std": [],
        "mono_mm_std": [],
        "wham_mm_std": [],
        "twocam_mm_std": [],
        "num_trials": [],
    }

    for movement in movements:
        # Filter results for this movement - use lowercase for filtering
        mono_movement = ik_results_mono[
            ik_results_mono["Movement"].str.lower() == movement.lower()
        ]
        wham_movement = ik_results_wham[
            ik_results_wham["Movement"].str.lower() == movement.lower()
        ]
        twocam_movement = ik_results_twocam[
            ik_results_twocam["Movement"].str.lower() == movement.lower()
        ]

        # Only include movement if all methods have data
        if (
            len(mono_movement) > 0
            and len(wham_movement) > 0
            and len(twocam_movement) > 0
        ):
            plot_data["movement"].append(movement)

            # Calculate metrics
            plot_data["mono_degrees"].append(mono_movement["global_mae_degrees"].mean())
            plot_data["wham_degrees"].append(wham_movement["global_mae_degrees"].mean())
            plot_data["twocam_degrees"].append(
                twocam_movement["global_mae_degrees"].mean()
            )

            plot_data["mono_mm"].append(mono_movement["global_mae_mm"].mean())
            plot_data["wham_mm"].append(wham_movement["global_mae_mm"].mean())
            plot_data["twocam_mm"].append(twocam_movement["global_mae_mm"].mean())

            # Calculate standard deviations
            plot_data["mono_degrees_std"].append(mono_movement["global_mae_degrees"].std())
            plot_data["wham_degrees_std"].append(wham_movement["global_mae_degrees"].std())
            plot_data["twocam_degrees_std"].append(twocam_movement["global_mae_degrees"].std())

            plot_data["mono_mm_std"].append(mono_movement["global_mae_mm"].std())
            plot_data["wham_mm_std"].append(wham_movement["global_mae_mm"].std())
            plot_data["twocam_mm_std"].append(twocam_movement["global_mae_mm"].std())
            
            # Track number of trials (using mono_movement as reference, all should have same count)
            plot_data["num_trials"].append(len(mono_movement))

    # Calculate overall averages
    plot_data["movement"].append("Average")
    plot_data["mono_degrees"].append(ik_results_mono["global_mae_degrees"].mean())
    plot_data["wham_degrees"].append(ik_results_wham["global_mae_degrees"].mean())
    plot_data["twocam_degrees"].append(ik_results_twocam["global_mae_degrees"].mean())
    plot_data["mono_mm"].append(ik_results_mono["global_mae_mm"].mean())
    plot_data["wham_mm"].append(ik_results_wham["global_mae_mm"].mean())
    plot_data["twocam_mm"].append(ik_results_twocam["global_mae_mm"].mean())

    # Calculate overall standard deviations
    plot_data["mono_degrees_std"].append(ik_results_mono["global_mae_degrees"].std())
    plot_data["wham_degrees_std"].append(ik_results_wham["global_mae_degrees"].std())
    plot_data["twocam_degrees_std"].append(ik_results_twocam["global_mae_degrees"].std())
    plot_data["mono_mm_std"].append(ik_results_mono["global_mae_mm"].std())
    plot_data["wham_mm_std"].append(ik_results_wham["global_mae_mm"].std())
    plot_data["twocam_mm_std"].append(ik_results_twocam["global_mae_mm"].std())
    
    # Add total number of trials for "Average" row
    plot_data["num_trials"].append(len(ik_results_mono))

    # Colors for the bars using plot_config
    colors = [
        pc.COLORS['wham'],
        pc.COLORS['mono'],
        pc.COLORS['twocam'],
    ]  # WHAM, Mono, 2CAMS

    # Create degrees plot with Plotly
    fig_degrees = go.Figure()

    # Add bars for degrees with error bars
    fig_degrees.add_trace(go.Bar(
        name="WHAM",
        x=plot_data["movement"],
        y=plot_data["wham_degrees"],
        error_y=dict(type='data', array=plot_data["wham_degrees_std"], visible=True, thickness=2, width=5),
        marker_color=colors[0],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.7,
        width=0.35
    ))

    fig_degrees.add_trace(go.Bar(
        name="OpenCap Monocular",
        x=plot_data["movement"],
        y=plot_data["mono_degrees"],
        error_y=dict(type='data', array=plot_data["mono_degrees_std"], visible=True, thickness=2, width=5),
        marker_color=colors[1],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.7,
        width=0.35
    ))

    fig_degrees.add_trace(go.Bar(
        name="OpenCap Two-camera",
        x=plot_data["movement"],
        y=plot_data["twocam_degrees"],
        error_y=dict(type='data', array=plot_data["twocam_degrees_std"], visible=True, thickness=2, width=5),
        marker_color=colors[2],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.7,
        width=0.35
    ))

    # Customize layout
    fig_degrees.update_layout(
        xaxis_title={
            'text': 'Movement',
            # 'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        yaxis_title={
            'text': 'Global MAE (degrees)',
            # 'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        # font={'size': 14, 'family': 'Arial, sans-serif'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=50, t=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        barmode='group',
        bargap=0.2,  # More space between different movement groups
        bargroupgap=0.1  # Space between bars within each group
    )

    # Update axes - remove gridlines and borders
    fig_degrees.update_xaxes(
        showgrid=False,  # Remove vertical grid lines
        zeroline=False,  # Remove vertical zero line
        linecolor='black',
        linewidth=1,
        mirror=False,    # Don't mirror to remove right border
        ticks='outside',
        # tickfont={'size': 14}
    )
    fig_degrees.update_yaxes(
        showgrid=False,  # Remove horizontal grid lines
        zeroline=True,   # Show axis line
        zerolinecolor='black',
        zerolinewidth=1,
        linecolor='black',
        linewidth=1,
        mirror=False,    # Don't mirror to remove top border
        ticks='outside',
        # tickfont={'size': 14}
    )

    # Add annotations right above the bars
    for i, movement in enumerate(plot_data["movement"]):
        # WHAM annotation (leftmost bar)
        fig_degrees.add_annotation(
            x=movement,
            y=plot_data["wham_degrees"][i] + plot_data["wham_degrees_std"][i] + 0.1,
            text=f"{plot_data['wham_degrees'][i]:.1f}",
            showarrow=False,
            font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
            xshift=-20  # Offset to the left for leftmost bar
        )
        # OpenCap Monocular annotation (middle bar)
        fig_degrees.add_annotation(
            x=movement,
            y=plot_data["mono_degrees"][i] + plot_data["mono_degrees_std"][i] + 0.1,
            text=f"{plot_data['mono_degrees'][i]:.1f}",
            showarrow=False,
            font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
            xshift=0  # Center for middle bar
        )
        # OpenCap Two-camera annotation (rightmost bar)
        fig_degrees.add_annotation(
            x=movement,
            y=plot_data["twocam_degrees"][i] + plot_data["twocam_degrees_std"][i] + 0.1,
            text=f"{plot_data['twocam_degrees'][i]:.1f}",
            showarrow=False,
            font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
            xshift=20  # Offset to the right for rightmost bar
        )

    # Remove text from bars since we're using annotations
    fig_degrees.update_traces(text="")

    if output_dir:
        # Save as HTML
        fig_degrees.write_html(os.path.join(output_dir, "global_comparison_degrees.html"))
        # Save as SVG
        fig_degrees.write_image(
            os.path.join(output_dir, "global_comparison_degrees.svg"),
            format='svg', width=pc.LAYOUT['width'], height=pc.LAYOUT['height']
        )

    # Create cm plot (converting from mm) with Plotly
    fig_cm = go.Figure()

    # Convert mm values to cm by dividing by 10
    wham_cm = [x / 10 for x in plot_data["wham_mm"]]
    mono_cm = [x / 10 for x in plot_data["mono_mm"]]
    twocam_cm = [x / 10 for x in plot_data["twocam_mm"]]

    # Convert std values to cm by dividing by 10
    wham_cm_std = [x / 10 for x in plot_data["wham_mm_std"]]
    mono_cm_std = [x / 10 for x in plot_data["mono_mm_std"]]
    twocam_cm_std = [x / 10 for x in plot_data["twocam_mm_std"]]

    # Add bars for cm with error bars
    fig_cm.add_trace(go.Bar(
        name="WHAM",
        x=plot_data["movement"],
        y=wham_cm,
        error_y=dict(type='data', array=wham_cm_std, visible=True, thickness=2, width=5),
        marker_color=colors[0],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.7,
        width=0.2
    ))

    fig_cm.add_trace(go.Bar(
        name="OpenCap Monocular",
        x=plot_data["movement"],
        y=mono_cm,
        error_y=dict(type='data', array=mono_cm_std, visible=True, thickness=2, width=5),
        marker_color=colors[1],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.7,
        width=0.2
    ))

    fig_cm.add_trace(go.Bar(
        name="OpenCap Two-camera",
        x=plot_data["movement"],
        y=twocam_cm,
        error_y=dict(type='data', array=twocam_cm_std, visible=True, thickness=2, width=5),
        marker_color=colors[2],
        marker_line_color='black',
        marker_line_width=1.5,
        opacity=0.7,
        width=0.2
    ))

    # Customize layout
    fig_cm.update_layout(
        xaxis_title={
            'text': 'Movement',
            # 'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        yaxis_title={
            'text': 'Global MAE (cm)',
            # 'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        # font={'size': 14, 'family': 'Arial, sans-serif'},
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        margin=dict(l=80, r=50, t=80, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0)',
            bordercolor='rgba(0,0,0,0)',
            borderwidth=0
        ),
        barmode='group',
        bargap=0.2,  # More space between different movement groups
        bargroupgap=0.1  # Space between bars within each group
    )

    # Update axes - remove gridlines and borders
    fig_cm.update_xaxes(
        showgrid=False,  # Remove vertical grid lines
        zeroline=False,  # Remove vertical zero line
        linecolor='black',
        linewidth=1,
        mirror=False,    # Don't mirror to remove right border
        ticks='outside',
        tickfont={'size': 14}
    )
    fig_cm.update_yaxes(
        showgrid=False,  # Remove horizontal grid lines
        zeroline=True,   # Show axis line
        zerolinecolor='black',
        zerolinewidth=1,
        linecolor='black',
        linewidth=1,
        mirror=False,    # Don't mirror to remove top border
        ticks='outside',
        tickfont={'size': 14}
    )

    # Add annotations right above the bars
    for i, movement in enumerate(plot_data["movement"]):
        # WHAM annotation (leftmost bar)
        fig_cm.add_annotation(
            x=movement,
            y=wham_cm[i] + 0.08,
            text=f"{wham_cm[i]:.1f}",
            showarrow=False,
            font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
            xshift=- 6   # Offset to the left for leftmost bar
        )
        # OpenCap Monocular annotation (middle bar)
        fig_cm.add_annotation(
            x=movement,
            y=mono_cm[i] + 0.08,
            text=f"{mono_cm[i]:.1f}",
            showarrow=False,
            font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
            xshift=2  # Center for middle bar
        )
        # OpenCap Two-camera annotation (rightmost bar)
        fig_cm.add_annotation(
            x=movement,
            y=twocam_cm[i] + 0.08,
            text=f"{twocam_cm[i]:.1f}",
            showarrow=False,
            font=dict(size=pc.FONTS['annotation_size'], color=pc.COLORS['text']),
            xshift= 6  # Offset to the right for rightmost bar
        )

    # Remove text from bars since we're using annotations
    fig_cm.update_traces(text="")

    if output_dir:
        # Save as HTML
        fig_cm.write_html(os.path.join(output_dir, "global_comparison_cm.html"))
        # Save as SVG
        fig_cm.write_image(
            os.path.join(output_dir, "global_comparison_cm.svg"),
            format='svg', width=pc.LAYOUT['width'], height=pc.LAYOUT['height']
        )

    # Export global metrics to CSV
    if output_dir:
        import pandas as pd
        
        # Create DataFrame for degrees metrics
        degrees_data = {
            'Movement': plot_data["movement"],
            'Num_Trials': plot_data["num_trials"],
            'WHAM_Mean_Degrees': plot_data["wham_degrees"],
            'WHAM_Std_Degrees': plot_data["wham_degrees_std"],
            'OpenCap_Monocular_Mean_Degrees': plot_data["mono_degrees"],
            'OpenCap_Monocular_Std_Degrees': plot_data["mono_degrees_std"],
            'OpenCap_Two_camera_Mean_Degrees': plot_data["twocam_degrees"],
            'OpenCap_Two_camera_Std_Degrees': plot_data["twocam_degrees_std"]
        }
        
        # Create DataFrame for cm metrics
        cm_data = {
            'Movement': plot_data["movement"],
            'Num_Trials': plot_data["num_trials"],
            'WHAM_Mean_cm': wham_cm,
            'WHAM_Std_cm': wham_cm_std,
            'OpenCap_Monocular_Mean_cm': mono_cm,
            'OpenCap_Monocular_Std_cm': mono_cm_std,
            'OpenCap_Two_camera_Mean_cm': twocam_cm,
            'OpenCap_Two_camera_Std_cm': twocam_cm_std
        }
        
        # Convert to DataFrames
        degrees_df = pd.DataFrame(degrees_data)
        cm_df = pd.DataFrame(cm_data)
        
        # Save to CSV files
        degrees_csv_path = os.path.join(output_dir, "global_metrics_degrees.csv")
        cm_csv_path = os.path.join(output_dir, "global_metrics_cm.csv")
        
        degrees_df.to_csv(degrees_csv_path, index=False)
        cm_df.to_csv(cm_csv_path, index=False)
        
        print(f"Global metrics degrees CSV saved to {os.path.abspath(degrees_csv_path)}")
        print(f"Global metrics cm CSV saved to {os.path.abspath(cm_csv_path)}")


def create_movement_comparison_plots_three_methods(
    mono_results, wham_results, twocam_results, output_dir=None
):
    """Create detailed movement comparison plots for all three methods."""
    # Use lowercase to match standardized movement names in data
    movements = ["walking", "squat", "STS"]

    for movement in movements:
        # Filter results for this movement - use case-insensitive comparison
        mono_movement = mono_results[
            mono_results["Movement"].str.lower() == movement.lower()
        ]
        wham_movement = wham_results[
            wham_results["Movement"].str.lower() == movement.lower()
        ]
        twocam_movement = twocam_results[
            twocam_results["Movement"].str.lower() == movement.lower()
        ]

        # Only include movement if all methods have data
        if (
            len(mono_movement) == 0
            or len(wham_movement) == 0
            or len(twocam_movement) == 0
        ):
            continue

        # DOF categories to analyze
        dof_categories = {
            "pelvis_position": ["pelvis_tx", "pelvis_ty", "pelvis_tz"],
            "pelvis_orientation": ["pelvis_tilt", "pelvis_list", "pelvis_rotation"],
            "hip": [
                "hip_flexion_l",
                "hip_flexion_r",
                "hip_adduction_l",
                "hip_adduction_r",
                "hip_rotation_l",
                "hip_rotation_r",
            ],
            "knee": ["knee_angle_l", "knee_angle_r"],
            "ankle": ["ankle_angle_l", "ankle_angle_r", "subtalar_angle_l", "subtalar_angle_r"],
            "lumbar": ["lumbar_extension", "lumbar_bending", "lumbar_rotation"],
        }

        # Create a plot for each DOF category
        for category, dofs in dof_categories.items():
            # Check that DOFs exist in all datasets
            valid_dofs = [
                dof
                for dof in dofs
                if dof in mono_movement.columns
                and dof in wham_movement.columns
                and dof in twocam_movement.columns
            ]

            if not valid_dofs:
                continue

            # Calculate means for valid DOFs
            mono_means = [mono_movement[dof].mean() for dof in valid_dofs]
            wham_means = [wham_movement[dof].mean() for dof in valid_dofs]
            twocam_means = [twocam_movement[dof].mean() for dof in valid_dofs]

            # Create figure
            plt.figure(figsize=(12, 6))

            # Bar width
            bar_width = 0.25

            # X positions
            x = np.arange(len(valid_dofs))

            # Create bars using plot_config colors
            wham_bars = plt.bar(
                x - bar_width, wham_means, bar_width, label="WHAM", color=pc.COLORS['wham']
            )
            mono_bars = plt.bar(x, mono_means, bar_width, label="OpenCap Monocular", color=pc.COLORS['mono'])
            twocam_bars = plt.bar(
                x + bar_width, twocam_means, bar_width, label="OpenCap Two-camera", color=pc.COLORS['twocam']
            )

            # Add value labels on top of bars
            for i, v in enumerate(wham_means):
                plt.text(
                    x[i] - bar_width,
                    v + 0.1,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            for i, v in enumerate(mono_means):
                plt.text(
                    x[i], v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=8
                )
            for i, v in enumerate(twocam_means):
                plt.text(
                    x[i] + bar_width,
                    v + 0.1,
                    f"{v:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Add labels, title, and legend with plot_config styling
            plt.ylabel("Error", fontsize=pc.FONTS['axis_title_size'])
            plt.title(f"{movement.capitalize()} - {category.capitalize()} Errors", fontsize=pc.FONTS['title_size'])
            plt.xticks(x, valid_dofs, rotation=45, ha="right", fontsize=pc.FONTS['tick_size'])
            
            # Apply plot_config legend styling
            legend = plt.legend(fontsize=pc.FONTS['legend_size'])
            legend.get_frame().set_facecolor(pc.COLORS['background'])
            legend.get_frame().set_edgecolor('none')
            
            # Apply axis styling from plot_config
            plt.gca().tick_params(labelsize=pc.FONTS['tick_size'])
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().set_facecolor(pc.COLORS['background'])

            # Apply overall figure styling
            plt.gcf().patch.set_facecolor(pc.COLORS['background'])
            plt.tight_layout()

            if output_dir:
                plt.savefig(
                    os.path.join(output_dir, f"{movement}_{category}_all_methods.png"),
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=pc.COLORS['background']
                )

            plt.close()


def main():
    # Use the base folders that were specified
    base_folders = [
        "output/nas/case_001_walking",
        "output/nas/case_001_STS",
        "output/nas/case_001_squats"
    ]

    camera_numbers = [3]

    # Define output directory
    output_dir = os.path.join(os.getcwd(), "analysis_results")

    # Define exclusion file
    exclusion_file = "validation/time_sync_issue.txt"

    # Clean the output directory first
    clean_output_directory(output_dir)

    # Create directory if it doesn't exist (should already be done by clean_output_directory)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(output_dir)}")

    # Generate list of all files in input folders for reference
    file_inventory_path = os.path.join(output_dir, "all_files_inventory.txt")
    with open(file_inventory_path, "w") as inventory_file:
        inventory_file.write("=== Complete Inventory of Files in Input Folders ===\n\n")
        for base_folder in base_folders:
            inventory_file.write(f"Files in {base_folder}:\n")
            for root, _, files in os.walk(base_folder):
                for file in files:
                    if file.endswith(".csv"):
                        inventory_file.write(f"  {os.path.join(root, file)}\n")
            inventory_file.write("\n")

    print(f"Complete file inventory saved to {os.path.abspath(file_inventory_path)}")

    # Analyze errors with output directory
    try:
        (
            ik_results_mono,
            ik_results_wham,
            ik_results_twocam,
            marker_results_mono,
            marker_results_wham,
        ) = analyze_errors(
            base_folders, camera_numbers, output_dir, exclusion_file=exclusion_file
        )

        # Save results to CSV with absolute paths
        if not ik_results_mono.empty:
            csv_path = os.path.join(output_dir, "ik_results_mono.csv")
            ik_results_mono.to_csv(csv_path, index=False)
            print(f"Mono IK results saved to {os.path.abspath(csv_path)}")

        if not ik_results_wham.empty:
            csv_path = os.path.join(output_dir, "ik_results_wham.csv")
            ik_results_wham.to_csv(csv_path, index=False)
            print(f"WHAM IK results saved to {os.path.abspath(csv_path)}")

        if not ik_results_twocam.empty:
            csv_path = os.path.join(output_dir, "ik_results_twocam.csv")
            ik_results_twocam.to_csv(csv_path, index=False)
            print(f"2CAMS IK results saved to {os.path.abspath(csv_path)}")

        if not marker_results_mono.empty:
            csv_path = os.path.join(output_dir, "marker_results_mono.csv")
            marker_results_mono.to_csv(csv_path, index=False)
            print(f"Mono marker results saved to {os.path.abspath(csv_path)}")

        if not marker_results_wham.empty:
            csv_path = os.path.join(output_dir, "marker_results_wham.csv")
            marker_results_wham.to_csv(csv_path, index=False)
            print(f"WHAM marker results saved to {os.path.abspath(csv_path)}")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
