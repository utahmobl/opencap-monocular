import os
import sys
import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

# Add the root directory to Python path
repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_path)

from utils.utils_trc import TRCFile


def calculate_marker_errors(trc_mono_or_wham, trc_mocap):
    """Calculate errors between corresponding markers in mono and mocap TRCs"""
    markers = {
        "r_knee": "r_knee",
        "l_knee": "L_knee",
        "r_shoulder": "R_Shoulder",
        "l_shoulder": "L_Shoulder",
        "r_ASIS": "r.ASIS",
        "l_ASIS": "L.ASIS",
        "r_PSIS": "r.PSIS",
        "l_PSIS": "L.PSIS",
        "r_ankle": "r_ankle",
        "l_ankle": "L_ankle",
        "r_toe": "r_toe",
        "l_toe": "L_toe",
    }

    marker_errors = {}
    for mono_marker, mocap_marker in markers.items():
        if not (
            trc_mono_or_wham.marker_exists(mono_marker)
            and trc_mocap.marker_exists(mocap_marker)
        ):
            continue

        mono_or_wham_data = trc_mono_or_wham.marker(mono_marker)
        mocap_data = trc_mocap.marker(mocap_marker)

        # if the shape of the mocap data is not the same as the mono data, then trim the mocap data to the same length as the mono data
        if mocap_data.shape[0] != mono_or_wham_data.shape[0]:
            mocap_data = mocap_data[: mono_or_wham_data.shape[0]]

        # Calculate Euclidean distance between markers at each frame
        errors = np.linalg.norm(mono_or_wham_data - mocap_data, axis=1)
        marker_errors[mono_marker] = round(np.mean(errors), 2)

    # compute the mean of toe and ankle errors
    marker_errors["toes"] = np.mean([marker_errors["r_toe"], marker_errors["l_toe"]])
    marker_errors["ankles"] = np.mean(
        [marker_errors["r_ankle"], marker_errors["l_ankle"]]
    )

    # compute the mean of PSIS and ASIS errors
    marker_errors["pelvis"] = np.mean(
        [
            marker_errors["r_PSIS"],
            marker_errors["l_PSIS"],
            marker_errors["r_ASIS"],
            marker_errors["l_ASIS"],
        ]
    )

    # remove the toe and ankle errors from the marker_errors dictionary
    marker_errors.pop("r_toe", None)
    marker_errors.pop("l_toe", None)
    marker_errors.pop("r_ankle", None)
    marker_errors.pop("l_ankle", None)
    marker_errors.pop("r_PSIS", None)
    marker_errors.pop("l_PSIS", None)
    marker_errors.pop("r_ASIS", None)
    marker_errors.pop("l_ASIS", None)

    return marker_errors


def run_marker_analysis(
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
    """Analyze marker position differences between mocap and mono TRC files. Also mocap/wham and mocap/2cams if enabled."""
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    trc_wham = None
    trc_2cams = None

    # Setup paths
    if not trimmed:
        mono_path_folder = os.path.join(
            root_path,
            f"output/{subject}/{session}/{cam}/{movement}/{movement}/MarkerData",
        )
        if output_case_path is not None:
            mono_path_folder = os.path.join(output_case_path, "MarkerData")
    else:
        mono_path_folder = os.path.join(
            root_path,
            f"output/{subject}/{session}/{cam}/{movement}/{movement}_trimmed/MarkerData",
        )
        if output_case_path is not None:
            mono_path_folder = os.path.join(output_case_path, "MarkerData")

    if run_wham:
        if not trimmed:
            wham_path_fol = os.path.join(
                root_path,
                f"output/{subject}/{session}/{cam}/{movement}/{movement}/MarkerData",
            )
        else:
            wham_path_fol = os.path.join(
                root_path,
                f"output/{subject}/{session}/{cam}/{movement}/{movement}_trimmed/MarkerData",
            )
        # get the folder in the wham_path_folder that contains the word 'wham_result'
        wham_path_folder = next(
            (d for d in os.listdir(wham_path_fol) if "wham_result" in d), None
        )
        wham_path_folder = os.path.join(wham_path_fol, wham_path_folder)
        # get the trc file in the wham_path_folder that contains the word 'sync_wham.trc'
        wham_trc_file = next(
            (f for f in os.listdir(wham_path_folder) if "sync_wham.trc" in f), None
        )
        if wham_trc_file is None:
            logger.error("No wham TRC file found")
        else:
            wham_trc_file = os.path.join(wham_path_folder, wham_trc_file)
            trc_wham = TRCFile(wham_trc_file)

    if run_2cams:
        # Path to 2-camera TRC file
        twocam_folder = os.path.join(
            root_path,
            f"LabValidation_withVideos1/{subject}/MarkerData/Video/HRNet/2-cameras/",
        )
        twocam_trc_files = []
        for root_dir, dirs, files in os.walk(twocam_folder):
            for file in files:
                if file.endswith(".trc") and movement in file:
                    twocam_trc_files.append(os.path.join(root_dir, file))

        if len(twocam_trc_files) == 0:
            logger.error(f"No 2-camera TRC file found for {subject}/{movement}")
        else:
            # If multiple files found, use the first one
            trc_2cams = TRCFile(twocam_trc_files[0])
            logger.info(f"Using 2-camera TRC file: {twocam_trc_files[0]}")

    # Find the mono TRC file
    mono_subdirs = [
        d for d in os.listdir(mono_path_folder) if d != "errors" and d != "shiftedIK"
    ]
    if not mono_subdirs:
        logger.error(f"No marker data found in {mono_path_folder}")
        return None, None, None

    folder = next(
        (d for d in mono_subdirs if "." not in d and "wham_result" not in d), None
    )
    if folder is None:
        logger.error("No valid marker data folder found")
        return None, None, None

    mono_path = os.path.join(mono_path_folder, folder)
    mono_files = [f for f in os.listdir(mono_path) if f.endswith("_sync.trc")]
    if not mono_files:
        logger.error("No synced TRC file found")
        return None, None, None

    mono_trc_path = os.path.join(mono_path, mono_files[0])
    mocap_trc_path = os.path.join(
        root_path,
        f"LabValidation_withVideos1/{subject}/MarkerData/Mocap/{movement}.trc",
    )

    # Load TRC files
    try:
        trc_mono = TRCFile(mono_trc_path)
        trc_mocap = TRCFile(mocap_trc_path)
    except Exception as e:
        logger.error(f"Error loading TRC files: {e}")
        return None, None, None

    # Initialize or load results DataFrame
    if exisiting_results_path_csv is not None and os.path.exists(
        exisiting_results_path_csv
    ):
        output_df = pd.read_csv(exisiting_results_path_csv)
        new_row = pd.DataFrame(columns=output_df.columns)
    else:
        output_df = pd.DataFrame()
        new_row = None

    # Calculate marker errors
    marker_errors = calculate_marker_errors(trc_mono, trc_mocap)
    if not marker_errors:
        logger.error("No matching markers found between mono and mocap")
        return None, None, None

    marker_mae_wham = None
    if trc_wham is not None:
        marker_errors_wham = calculate_marker_errors(trc_wham, trc_mocap)
        if not marker_errors_wham:
            logger.error("No matching markers found between mocap and wham")
        else:
            marker_mae_wham = np.mean(list(marker_errors_wham.values()))
            marker_mae_wham = np.round(marker_mae_wham, 2)

            results_path_wham = os.path.join(
                mono_path_folder, "marker_analysis_results_wham.txt"
            )
            with open(results_path_wham, "w") as f:
                f.write(f"Global Marker MAE (mm): {marker_mae_wham}\n")
                for marker, error in marker_errors_wham.items():
                    f.write(f"{marker}: {error:.2f} mm\n")
            csv_path_wham = os.path.join(
                mono_path_folder, "marker_analysis_results_wham.csv"
            )
            output_df_wham = pd.DataFrame(marker_errors_wham, index=[0])
            output_df_wham.to_csv(csv_path_wham, index=False)

    marker_mae_2cams = None
    if trc_2cams is not None:
        marker_errors_2cams = calculate_marker_errors(trc_2cams, trc_mocap)
        if not marker_errors_2cams:
            logger.error("No matching markers found between mocap and 2cams")
        else:
            marker_mae_2cams = np.mean(list(marker_errors_2cams.values()))
            marker_mae_2cams = np.round(marker_mae_2cams, 2)

            results_path_2cams = os.path.join(
                mono_path_folder, "marker_analysis_results_2cams.txt"
            )
            with open(results_path_2cams, "w") as f:
                f.write(f"Global Marker MAE (mm): {marker_mae_2cams}\n")
                for marker, error in marker_errors_2cams.items():
                    f.write(f"{marker}: {error:.2f} mm\n")
            csv_path_2cams = os.path.join(
                mono_path_folder, "marker_analysis_results_2cams.csv"
            )
            output_df_2cams = pd.DataFrame(marker_errors_2cams, index=[0])
            output_df_2cams.to_csv(csv_path_2cams, index=False)

    # Calculate global marker MAE
    marker_mae = np.mean(list(marker_errors.values()))
    marker_mae = np.round(marker_mae, 2)

    # Prepare results
    results_path = os.path.join(mono_path_folder, "marker_analysis_results_mono.txt")
    with open(results_path, "w") as f:
        f.write(f"Global Marker MAE (mm): {marker_mae}\n")
        for marker, error in marker_errors.items():
            f.write(f"{marker}: {error:.2f} mm\n")
    csv_path_mono = os.path.join(mono_path_folder, "marker_analysis_results_mono.csv")
    output_df_mono = pd.DataFrame(marker_errors, index=[0])
    output_df_mono.to_csv(csv_path_mono, index=False)
    output_df_mono = None

    # Update DataFrame
    if new_row is not None:
        # Check if a row with the same identifiers already exists
        existing_row = output_df[
            (output_df["subject"] == subject)
            & (output_df["session"] == session)
            & (output_df["cam"] == cam)
            & (output_df["movement"] == movement)
        ]

        if len(existing_row) > 0:
            # Update the existing row's marker_mae_mm value
            idx = output_df[
                (output_df["subject"] == subject)
                & (output_df["session"] == session)
                & (output_df["cam"] == cam)
                & (output_df["movement"] == movement)
            ].index[0]

            output_df.loc[idx, "marker_mae_mm"] = marker_mae
            if marker_mae_wham is not None:
                output_df.loc[idx, "marker_mae_mm_wham"] = marker_mae_wham
            if marker_mae_2cams is not None:
                output_df.loc[idx, "marker_mae_mm_2cams"] = marker_mae_2cams
        else:
            # Add a new row if no matching row exists
            new_row["subject"] = [subject]
            new_row["session"] = [session]
            new_row["cam"] = [cam]
            new_row["movement"] = [movement]
            new_row["marker_mae_mm"] = [marker_mae]
            if marker_mae_wham is not None:
                new_row["marker_mae_mm_wham"] = [marker_mae_wham]
            if marker_mae_2cams is not None:
                new_row["marker_mae_mm_2cams"] = [marker_mae_2cams]
            output_df = pd.concat([output_df, new_row], ignore_index=True)
    else:
        data_dict = {
            "subject": [subject],
            "session": [session],
            "cam": [cam],
            "movement": [movement],
            "marker_mae_mm": [marker_mae],
        }
        if marker_mae_wham is not None:
            data_dict["marker_mae_mm_wham"] = [marker_mae_wham]
        if marker_mae_2cams is not None:
            data_dict["marker_mae_mm_2cams"] = [marker_mae_2cams]
        output_df = pd.DataFrame(data_dict)

    # Save results
    output_csv_path = (
        output_path_csv
        if output_path_csv
        else os.path.join(mono_path_folder, "marker_analysis_results.csv")
    )
    output_df.to_csv(output_csv_path, index=False)

    logger.info(f"Marker analysis completed. Global MAE: {marker_mae} mm")
    if marker_mae_wham is not None:
        logger.info(f"WHAM Global Marker MAE: {marker_mae_wham} mm")
    if marker_mae_2cams is not None:
        logger.info(f"2CAMS Global Marker MAE: {marker_mae_2cams} mm")

    # Create comparison bar charts
    if run_wham or run_2cams:
        create_marker_comparison_plots(
            mono_path_folder,
            subject,
            movement,
            marker_errors=marker_errors,
            marker_errors_wham=marker_errors_wham if trc_wham is not None else None,
            marker_errors_2cams=marker_errors_2cams if trc_2cams is not None else None,
        )

    # Return results depending on what was calculated
    results_dict = {
        "mono": marker_mae,
        "results_path": results_path,
        "output_csv_path": output_csv_path,
    }

    if marker_mae_wham is not None:
        results_dict["wham"] = marker_mae_wham

    if marker_mae_2cams is not None:
        results_dict["2cams"] = marker_mae_2cams

    return results_dict


def create_marker_comparison_plots(
    output_folder,
    subject,
    movement,
    marker_errors,
    marker_errors_wham=None,
    marker_errors_2cams=None,
):
    """
    Create bar charts comparing marker errors across different methods.

    Args:
        output_folder: Path to save the output plots
        subject: Subject identifier
        movement: Movement type
        marker_errors: Dictionary of marker errors for mono method
        marker_errors_wham: Optional dictionary of marker errors for WHAM method
        marker_errors_2cams: Optional dictionary of marker errors for 2CAMS method
    """
    # Define the categories and get their values
    categories = ["pelvis", "ankles", "toes"]
    mono_values = [
        marker_errors.get("pelvis", 0),
        marker_errors.get("ankles", 0),
        marker_errors.get("toes", 0),
    ]

    # Determine which methods are available
    methods = ["Mono"]
    all_values = [mono_values]
    colors = ["#3498db"]  # Blue for Mono

    if marker_errors_wham is not None:
        methods.append("WHAM")
        wham_values = [
            marker_errors_wham.get("pelvis", 0),
            marker_errors_wham.get("ankles", 0),
            marker_errors_wham.get("toes", 0),
        ]
        all_values.append(wham_values)
        colors.append("#e74c3c")  # Red for WHAM

    if marker_errors_2cams is not None:
        methods.append("2CAMS")
        twocams_values = [
            marker_errors_2cams.get("pelvis", 0),
            marker_errors_2cams.get("ankles", 0),
            marker_errors_2cams.get("toes", 0),
        ]
        all_values.append(twocams_values)
        colors.append("#2ecc71")  # Green for 2CAMS

    # Reorder methods to match the desired order: WHAM, Mono, 2CAMS
    if set(methods) == set(["Mono", "WHAM", "2CAMS"]):
        methods = ["WHAM", "Mono", "2CAMS"]
        all_values = [
            all_values[methods.index("WHAM")],
            all_values[methods.index("Mono")],
            all_values[methods.index("2CAMS")],
        ]
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
    elif set(methods) == set(["Mono", "WHAM"]):
        methods = ["WHAM", "Mono"]
        all_values = [
            all_values[methods.index("WHAM")],
            all_values[methods.index("Mono")],
        ]
        colors = ["#e74c3c", "#3498db"]

    # Set up the bar chart
    x = np.arange(len(categories))
    width = 0.8 / len(methods)  # Adjust width based on number of methods

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create the bars
    rects = []
    for i, (method_values, color) in enumerate(zip(all_values, colors)):
        offset = width * (i - len(methods) / 2 + 0.5)
        rect = ax.bar(x + offset, method_values, width, label=methods[i], color=color)
        rects.append(rect)

    # Add labels and title
    ax.set_ylabel("Mean Error (mm)", fontsize=12)
    ax.set_title(f"Marker Errors Comparison - {subject}/{movement}", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)

    # Add grid for easier reading
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    for rect_group in rects:
        autolabel(rect_group)

    # Save the figure
    plt.tight_layout()
    methods_str = "_".join(method.lower() for method in methods)
    chart_path = os.path.join(output_folder, f"marker_errors_{methods_str}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Comparison chart saved to: {chart_path}")

    # Now create a chart for global MAE values
    if len(methods) > 1:
        # Get global MAE values
        global_values = []
        for method_idx, method in enumerate(methods):
            if method == "Mono":
                global_values.append(np.mean(list(marker_errors.values())))
            elif method == "WHAM" and marker_errors_wham is not None:
                global_values.append(np.mean(list(marker_errors_wham.values())))
            elif method == "2CAMS" and marker_errors_2cams is not None:
                global_values.append(np.mean(list(marker_errors_2cams.values())))

        # Create the global MAE chart
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(methods, global_values, color=colors, width=0.6)

        # Add labels and styling
        ax.set_ylabel("Global MAE (mm)", fontsize=12)
        ax.set_title(
            f"Global Marker MAE Comparison - {subject}/{movement}", fontsize=14
        )
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        # Save the global chart
        plt.tight_layout()
        global_chart_path = os.path.join(
            output_folder, f"global_marker_mae_{methods_str}.png"
        )
        plt.savefig(global_chart_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Global MAE comparison chart saved to: {global_chart_path}")


if __name__ == "__main__":
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_csv = os.path.join(repo_path, "output", "marker_results.csv")
    subject = "subject4"
    session = "Session0"
    cam = "Cam1"
    video = "STSweakLegs1"
    trimmed = False

    results = run_marker_analysis(
        subject,
        session,
        cam,
        video,
        exisiting_results_path_csv=output_csv,
        output_path_csv=output_csv,
        trimmed=trimmed,
        run_wham=True,
        run_2cams=True,
    )

    print(f"Results: {results}")
    print(f"Results saved to: {results['output_csv_path']}")
