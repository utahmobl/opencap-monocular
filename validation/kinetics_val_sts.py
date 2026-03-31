import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the root directory to the path so we can import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_mot import load_mot, mot_to_dataframe

# Create output directory
output_dir = "sts_kinetics_validation"
os.makedirs(output_dir, exist_ok=True)

# List of subjects to process
subjects = []
for i in range(1, 12):
    subjects.append(f"subject{i}")

base_path = "/home/selim/opencap-mono/output/v7"
case = "case_001_STS"
session = "Session0"
cam = "Cam3"

vid_normal = "STS1"
vid_weak = "STSweakLegs1"

# Repetitions to analyze (simulation numbers, 1-based)
reps = [1, 2, 3]  # These correspond to simulation numbers

# Create dictionaries to hold results for all subjects
all_subjects_normal = {}
all_subjects_weak = {}
all_subjects_diff = {}
precision = 2

# List to store all data for combined CSV (for statistical analysis)
all_data_for_stats = []


# Function to create summary plots
def create_summary_plot(results_df, condition, subject=None, output_dir="."):
    plt.figure(figsize=(15, 10))

    # Bar chart of knee extension moments across simulations
    plt.subplot(3, 1, 1)
    plt.bar(
        [str(s) for s in results_df["simulation"]],
        results_df["knee_ext_l"],
        color="b",
        alpha=0.7,
        label="Left",
    )
    plt.bar(
        [str(s) for s in results_df["simulation"]],
        results_df["knee_ext_r"],
        color="r",
        alpha=0.7,
        label="Right",
        bottom=results_df["knee_ext_l"],
    )
    plt.axhline(
        y=results_df["knee_ext_combined"].mean(),
        color="k",
        linestyle="--",
        label=f'Avg: {results_df["knee_ext_combined"].mean():.{precision}f} Nm',
    )
    subject_label = f"{subject} - " if subject else ""
    plt.title(
        f"Knee Extension Moments - {subject_label}{condition.capitalize()} Simulations"
    )
    plt.ylabel("Moment (Nm)")
    plt.legend()
    plt.grid(True)

    # Bar chart of hip extension moments across simulations
    plt.subplot(3, 1, 2)
    plt.bar(
        [str(s) for s in results_df["simulation"]],
        results_df["hip_ext_l"],
        color="b",
        alpha=0.7,
        label="Left",
    )
    plt.bar(
        [str(s) for s in results_df["simulation"]],
        results_df["hip_ext_r"],
        color="r",
        alpha=0.7,
        label="Right",
        bottom=results_df["hip_ext_l"],
    )
    plt.axhline(
        y=results_df["hip_ext_combined"].mean(),
        color="k",
        linestyle="--",
        label=f'Avg: {results_df["hip_ext_combined"].mean():.{precision}f} Nm',
    )
    plt.title(
        f"Hip Extension Moments - {subject_label}{condition.capitalize()} Simulations"
    )
    plt.ylabel("Moment (Nm)")
    plt.legend()
    plt.grid(True)

    # Bar chart of ankle plantarflexion moments across simulations
    plt.subplot(3, 1, 3)
    plt.bar(
        [str(s) for s in results_df["simulation"]],
        results_df["ankle_pf_l"],
        color="b",
        alpha=0.7,
        label="Left",
    )
    plt.bar(
        [str(s) for s in results_df["simulation"]],
        results_df["ankle_pf_r"],
        color="r",
        alpha=0.7,
        label="Right",
        bottom=results_df["ankle_pf_l"],
    )
    plt.axhline(
        y=results_df["ankle_pf_combined"].mean(),
        color="k",
        linestyle="--",
        label=f'Avg: {results_df["ankle_pf_combined"].mean():.{precision}f} Nm',
    )
    plt.title(
        f"Ankle Plantarflexion Moments - {subject_label}{condition.capitalize()} Simulations"
    )
    plt.xlabel("Simulation")
    plt.ylabel("Moment (Nm)")
    plt.legend()
    plt.grid(True)

    # Adjust layout and save the summary plot
    plt.tight_layout()
    filename_prefix = f"{subject}_" if subject else ""
    summary_plot_filename = os.path.join(
        output_dir, f"{filename_prefix}{condition}_all_simulations_comparison.png"
    )
    plt.savefig(summary_plot_filename)
    plt.close()
    print(f"\nSummary plot saved as {os.path.abspath(summary_plot_filename)}")


# Function to process kinetics data from a mot file
def process_kinetics_data(
    mot_file, start_time, end_time, subject, rep, condition, method, output_dir
):
    """Process kinetics data from a mot file and return results dictionary."""
    if not os.path.exists(mot_file):
        return None

    try:
        mot_data, mot_headers = load_mot(mot_file, outputFormat="dataframe")
    except Exception as e:
        print(f"Error loading mot file {mot_file}: {e}")
        return None

    # Verify the required columns exist
    required_columns = [
        "knee_angle_l_moment",
        "knee_angle_r_moment",
        "hip_flexion_l_moment",
        "hip_flexion_r_moment",
        "ankle_angle_r_moment",
        "ankle_angle_l_moment",
    ]

    missing_columns = [col for col in required_columns if col not in mot_data.columns]
    if missing_columns:
        print(f"Warning: Missing columns in {mot_file}: {missing_columns}")
        return None

    # Get data for the specific rising phase
    phase_data = mot_data[
        (mot_data.time >= start_time) & (mot_data.time <= end_time)
    ]

    if phase_data.empty:
        print(
            f"Warning: No data found for rising phase ({start_time}s to {end_time}s) in {mot_file}"
        )
        return None

    # Calculate average moments (negate for extension/plantarflexion)
    avg_knee_ext_moment_l = -phase_data["knee_angle_l_moment"].mean()
    avg_knee_ext_moment_r = -phase_data["knee_angle_r_moment"].mean()
    avg_hip_ext_moment_l = -phase_data["hip_flexion_l_moment"].mean()
    avg_hip_ext_moment_r = -phase_data["hip_flexion_r_moment"].mean()
    avg_ankle_pf_moment_l = -phase_data["ankle_angle_l_moment"].mean()
    avg_ankle_pf_moment_r = -phase_data["ankle_angle_r_moment"].mean()

    # Create results dictionary
    results = {
        "subject": subject,
        "condition": condition,
        "method": method,
        "simulation": rep,
        "knee_ext_l": avg_knee_ext_moment_l,
        "knee_ext_r": avg_knee_ext_moment_r,
        "hip_ext_l": avg_hip_ext_moment_l,
        "hip_ext_r": avg_hip_ext_moment_r,
        "ankle_pf_l": avg_ankle_pf_moment_l,
        "ankle_pf_r": avg_ankle_pf_moment_r,
        "knee_ext_combined": avg_knee_ext_moment_l + avg_knee_ext_moment_r,
        "hip_ext_combined": avg_hip_ext_moment_l + avg_hip_ext_moment_r,
        "ankle_pf_combined": avg_ankle_pf_moment_l + avg_ankle_pf_moment_r,
    }

    return results


# Lists to store aggregated results across all subjects
aggregated_normal = []
aggregated_weak = []
aggregated_diff = []

# Process each subject
for subject in subjects:
    print(f"\n{'='*50}")
    print(f"PROCESSING SUBJECT: {subject}")
    print(f"{'='*50}")

    # Initialize dictionaries for this subject
    all_results_normal = {}
    all_results_weak = {}
    all_phase_data_normal = {}
    all_phase_data_weak = {}

    # Base simulation name for this subject
    simulation_base = f"{subject}_STS_"

    # Process both videos for this subject
    for video_type in ["normal", "weak"]:
        # Select which video to use
        vid = vid_normal if video_type == "normal" else vid_weak
        print(f"\n{'-'*40}")
        print(f"PROCESSING {video_type.upper()} VIDEO: {vid} FOR {subject}")
        print(f"{'-'*40}")

        # Path to segmentation results
        segmentation_result_file = os.path.expanduser(
            f"{base_path}/{case}/{subject}/{session}/{cam}/{vid}/OpenSim/Simulations/STS_times.json"
        )

        # Check if segmentation file exists
        if not os.path.exists(segmentation_result_file):
            print(f"Warning: Cannot access file at {segmentation_result_file}")
            continue

        # Load segmentation results
        try:
            with open(segmentation_result_file, "r") as f:
                rising_times = json.load(f)

            # The file contains a direct array of time intervals
            if not rising_times or not isinstance(rising_times, list):
                print(f"Warning: No rising times found for {subject}")
                continue
        except Exception as e:
            print(f"Error loading segmentation data for {subject}: {e}")
            continue

        for rep in reps:
            # Skip if rep is out of range (adjust for 0-based indexing in list)
            rep_idx = rep - 1  # Convert to 0-based index
            if rep_idx >= len(rising_times):
                print(
                    f"Warning: Rep {rep} out of range for {subject}. Only have {len(rising_times)} rising times."
                )
                continue

            # Get the specific rising time interval for this repetition
            start_time, end_time = rising_times[rep_idx]
            print(f"\nAnalyzing rising phase {rep} (time: {start_time}s to {end_time}s)")

            # Process OpenCap Mono data
            # Simulation folders are 0-based, so use rep_idx for folder name but rep for results
            simulation = f"{simulation_base}{rep_idx}"
            mono_mot_file = os.path.expanduser(
                f"{base_path}/{case}/{subject}/{session}/{cam}/{vid}/OpenSim/Simulations/{simulation}/OpenSimData/Dynamics/{simulation}/kinetics_{simulation}_{case}.mot"
            )

            mono_results = process_kinetics_data(
                mono_mot_file,
                start_time,
                end_time,
                subject,
                rep,
                video_type,
                "opencap_mono",
                output_dir,
            )

            if mono_results:
                # Store results
                if video_type == "normal":
                    all_results_normal[rep] = mono_results
                    aggregated_normal.append(mono_results)
                else:
                    all_results_weak[rep] = mono_results
                    aggregated_weak.append(mono_results)

                # Add to combined stats list
                all_data_for_stats.append(mono_results)

                # Print results
                print(f"\nOpenCap Mono - Average moments during rising phase {rep}:")
                print(
                    f"Left knee extension moment: {mono_results['knee_ext_l']:.{precision}f} Nm"
                )
                print(
                    f"Right knee extension moment: {mono_results['knee_ext_r']:.{precision}f} Nm"
                )
                print(
                    f"Left hip extension moment: {mono_results['hip_ext_l']:.{precision}f} Nm"
                )
                print(
                    f"Right hip extension moment: {mono_results['hip_ext_r']:.{precision}f} Nm"
                )
                print(
                    f"Left ankle plantarflexion moment: {mono_results['ankle_pf_l']:.{precision}f} Nm"
                )
                print(
                    f"Right ankle plantarflexion moment: {mono_results['ankle_pf_r']:.{precision}f} Nm"
                )

            # Process Mocap data (if available)
            # Ground truth mocap kinetics files are in LabValidation_withVideos1
            mocap_base_path = "/home/selim/opencap-mono/LabValidation_withVideos1"
            mocap_sto_file = os.path.expanduser(
                f"{mocap_base_path}/{subject}/OpenSimData/Mocap/ID/{vid}.sto"
            )

            mocap_results = process_kinetics_data(
                mocap_sto_file,
                start_time,
                end_time,
                subject,
                rep,
                video_type,
                "mocap",
                output_dir,
            )

            if mocap_results:
                # Add to combined stats list
                all_data_for_stats.append(mocap_results)

                print(f"\nMocap - Average moments during rising phase {rep}:")
                print(
                    f"Left knee extension moment: {mocap_results['knee_ext_l']:.{precision}f} Nm"
                )
                print(
                    f"Right knee extension moment: {mocap_results['knee_ext_r']:.{precision}f} Nm"
                )

            # Create individual repetition plot (only for OpenCap Mono if available)
            if mono_results:
                # Load the data again for plotting
                try:
                    mot_data, _ = load_mot(mono_mot_file, outputFormat="dataframe")
                    phase_data = mot_data[
                        (mot_data.time >= start_time) & (mot_data.time <= end_time)
                    ]

                    if not phase_data.empty:
                        plt.figure(figsize=(15, 10))

                        # Knee extension moments
                        plt.subplot(3, 1, 1)
                        plt.plot(
                            phase_data["time"],
                            -phase_data["knee_angle_l_moment"],
                            "b-",
                            label="Left Knee",
                        )
                        plt.plot(
                            phase_data["time"],
                            -phase_data["knee_angle_r_moment"],
                            "r-",
                            label="Right Knee",
                        )
                        plt.axhline(
                            y=mono_results["knee_ext_l"],
                            color="b",
                            linestyle="--",
                            label=f"Left Avg: {mono_results['knee_ext_l']:.{precision}f} Nm",
                        )
                        plt.axhline(
                            y=mono_results["knee_ext_r"],
                            color="r",
                            linestyle="--",
                            label=f"Right Avg: {mono_results['knee_ext_r']:.{precision}f} Nm",
                        )
                        plt.title(
                            f"Knee Extension Moments - {subject} - {video_type.capitalize()} Simulation {simulation}"
                        )
                        plt.ylabel("Moment (Nm)")
                        plt.legend()
                        plt.grid(True)

                        # Hip extension moments
                        plt.subplot(3, 1, 2)
                        plt.plot(
                            phase_data["time"],
                            -phase_data["hip_flexion_l_moment"],
                            "b-",
                            label="Left Hip",
                        )
                        plt.plot(
                            phase_data["time"],
                            -phase_data["hip_flexion_r_moment"],
                            "r-",
                            label="Right Hip",
                        )
                        plt.axhline(
                            y=mono_results["hip_ext_l"],
                            color="b",
                            linestyle="--",
                            label=f"Left Avg: {mono_results['hip_ext_l']:.{precision}f} Nm",
                        )
                        plt.axhline(
                            y=mono_results["hip_ext_r"],
                            color="r",
                            linestyle="--",
                            label=f"Right Avg: {mono_results['hip_ext_r']:.{precision}f} Nm",
                        )
                        plt.title(
                            f"Hip Extension Moments - {subject} - {video_type.capitalize()} Simulation {simulation}"
                        )
                        plt.ylabel("Moment (Nm)")
                        plt.legend()
                        plt.grid(True)

                        # Ankle plantarflexion moments
                        plt.subplot(3, 1, 3)
                        plt.plot(
                            phase_data["time"],
                            -phase_data["ankle_angle_l_moment"],
                            "b-",
                            label="Left Ankle",
                        )
                        plt.plot(
                            phase_data["time"],
                            -phase_data["ankle_angle_r_moment"],
                            "r-",
                            label="Right Ankle",
                        )
                        plt.axhline(
                            y=mono_results["ankle_pf_l"],
                            color="b",
                            linestyle="--",
                            label=f"Left Avg: {mono_results['ankle_pf_l']:.{precision}f} Nm",
                        )
                        plt.axhline(
                            y=mono_results["ankle_pf_r"],
                            color="r",
                            linestyle="--",
                            label=f"Right Avg: {mono_results['ankle_pf_r']:.{precision}f} Nm",
                        )
                        plt.title(
                            f"Ankle Plantarflexion Moments - {subject} - {video_type.capitalize()} Simulation {simulation}"
                        )
                        plt.xlabel("Time (s)")
                        plt.ylabel("Moment (Nm)")
                        plt.legend()
                        plt.grid(True)

                        # Adjust layout and save the plot
                        plt.tight_layout()
                        plot_filename = os.path.join(
                            output_dir,
                            f"{subject}_{video_type}_simulation_{rep}_moments.png",
                        )
                        plt.savefig(plot_filename)
                        plt.close()
                        print(f"Plot saved as {os.path.abspath(plot_filename)}")
                except Exception as e:
                    print(f"Error creating plot: {e}")

        # Save results for this subject and video type to CSV
        if video_type == "normal" and all_results_normal:
            results_df = pd.DataFrame.from_dict(all_results_normal, orient="index")

            # Save the normal results to a CSV file
            csv_filename = os.path.join(
                output_dir, f"{subject}_normal_simulation_moments_results.csv"
            )
            results_df.to_csv(csv_filename, index=False)
            print(
                f"Normal results for {subject} saved to {os.path.abspath(csv_filename)}"
            )

            # Create summary plot for normal
            create_summary_plot(results_df, "normal", subject, output_dir)

            # Store in the all subjects dictionary
            all_subjects_normal[subject] = results_df

        elif video_type == "weak" and all_results_weak:
            results_df = pd.DataFrame.from_dict(all_results_weak, orient="index")

            # Save the weak results to a CSV file
            csv_filename = os.path.join(
                output_dir, f"{subject}_weak_simulation_moments_results.csv"
            )
            results_df.to_csv(csv_filename, index=False)
            print(
                f"Weak results for {subject} saved to {os.path.abspath(csv_filename)}"
            )

            # Create summary plot for weak
            create_summary_plot(results_df, "weak", subject, output_dir)

            # Store in the all subjects dictionary
            all_subjects_weak[subject] = results_df

    # Calculate the difference between normal and weak conditions for this subject
    if all_results_normal and all_results_weak:
        print("\n" + "-" * 40)
        print(f"CALCULATING DIFFERENCES (NORMAL - WEAK) FOR {subject}")
        print("-" * 40)

        # Find common repetitions for comparison
        common_reps = set(all_results_normal.keys()).intersection(
            set(all_results_weak.keys())
        )

        if not common_reps:
            print(
                f"No common repetitions found between normal and weak conditions for {subject}."
            )
        else:
            differences = {}

            for rep in common_reps:
                normal_results = all_results_normal[rep]
                weak_results = all_results_weak[rep]

                diff = {"subject": subject, "simulation": rep}

                for key in [
                    "knee_ext_l",
                    "knee_ext_r",
                    "hip_ext_l",
                    "hip_ext_r",
                    "ankle_pf_l",
                    "ankle_pf_r",
                ]:
                    diff[key] = normal_results[key] - weak_results[key]

                differences[rep] = diff
                aggregated_diff.append(diff)  # Add to aggregated differences

                # Print differences for this repetition
                print(f"\nDifference (Normal - Weak) for {subject} repetition {rep}:")
                for key in [
                    "knee_ext_l",
                    "knee_ext_r",
                    "hip_ext_l",
                    "hip_ext_r",
                    "ankle_pf_l",
                    "ankle_pf_r",
                ]:
                    print(f"{key}: {diff[key]:.{precision}f} Nm")

            # Convert differences to a dataframe
            if differences:
                diff_df = pd.DataFrame.from_dict(differences, orient="index")

                # Add combined values for differences
                diff_df["knee_ext_combined"] = (
                    diff_df["knee_ext_l"] + diff_df["knee_ext_r"]
                )
                diff_df["hip_ext_combined"] = (
                    diff_df["hip_ext_l"] + diff_df["hip_ext_r"]
                )
                diff_df["ankle_pf_combined"] = (
                    diff_df["ankle_pf_l"] + diff_df["ankle_pf_r"]
                )

                # Save the difference results to a CSV file
                csv_filename = os.path.join(
                    output_dir, f"{subject}_difference_simulation_moments_results.csv"
                )
                diff_df.to_csv(csv_filename, index=False)
                print(
                    f"Difference results for {subject} saved to {os.path.abspath(csv_filename)}"
                )

                # Store in the all subjects dictionary
                all_subjects_diff[subject] = diff_df

# Process aggregated results across all subjects
print("\n" + "=" * 50)
print("PROCESSING AGGREGATED RESULTS ACROSS ALL SUBJECTS")
print("=" * 50)

# Convert aggregated data to DataFrames
if aggregated_normal:
    normal_df = pd.DataFrame(aggregated_normal)
    normal_df["knee_ext_combined"] = normal_df["knee_ext_l"] + normal_df["knee_ext_r"]
    normal_df["hip_ext_combined"] = normal_df["hip_ext_l"] + normal_df["hip_ext_r"]
    normal_df["ankle_pf_combined"] = normal_df["ankle_pf_l"] + normal_df["ankle_pf_r"]

    # Save aggregated normal results
    normal_df.to_csv(
        os.path.join(output_dir, "all_subjects_normal_results.csv"), index=False
    )
    print("All subjects normal results saved to all_subjects_normal_results.csv")

    # Print average normal results across all subjects
    print("\nAverage normal moments across all subjects:")
    for col in [
        "knee_ext_l",
        "knee_ext_r",
        "hip_ext_l",
        "hip_ext_r",
        "ankle_pf_l",
        "ankle_pf_r",
        "knee_ext_combined",
        "hip_ext_combined",
        "ankle_pf_combined",
    ]:
        print(f"{col}: {normal_df[col].mean():.{precision}f} Nm")

if aggregated_weak:
    weak_df = pd.DataFrame(aggregated_weak)
    weak_df["knee_ext_combined"] = weak_df["knee_ext_l"] + weak_df["knee_ext_r"]
    weak_df["hip_ext_combined"] = weak_df["hip_ext_l"] + weak_df["hip_ext_r"]
    weak_df["ankle_pf_combined"] = weak_df["ankle_pf_l"] + weak_df["ankle_pf_r"]

    # Save aggregated weak results
    weak_df.to_csv(
        os.path.join(output_dir, "all_subjects_weak_results.csv"), index=False
    )
    print("All subjects weak results saved to all_subjects_weak_results.csv")

    # Print average weak results across all subjects
    print("\nAverage weak moments across all subjects:")
    for col in [
        "knee_ext_l",
        "knee_ext_r",
        "hip_ext_l",
        "hip_ext_r",
        "ankle_pf_l",
        "ankle_pf_r",
        "knee_ext_combined",
        "hip_ext_combined",
        "ankle_pf_combined",
    ]:
        print(f"{col}: {weak_df[col].mean():.{precision}f} Nm")

if aggregated_diff:
    diff_df = pd.DataFrame(aggregated_diff)
    diff_df["knee_ext_combined"] = diff_df["knee_ext_l"] + diff_df["knee_ext_r"]
    diff_df["hip_ext_combined"] = diff_df["hip_ext_l"] + diff_df["hip_ext_r"]
    diff_df["ankle_pf_combined"] = diff_df["ankle_pf_l"] + diff_df["ankle_pf_r"]

    # Save aggregated difference results
    diff_df.to_csv(
        os.path.join(output_dir, "all_subjects_difference_results.csv"), index=False
    )
    print(
        "All subjects difference results saved to all_subjects_difference_results.csv"
    )

    # Print average difference results across all subjects
    print("\nAverage differences (normal - weak) across all subjects:")
    for col in [
        "knee_ext_l",
        "knee_ext_r",
        "hip_ext_l",
        "hip_ext_r",
        "ankle_pf_l",
        "ankle_pf_r",
        "knee_ext_combined",
        "hip_ext_combined",
        "ankle_pf_combined",
    ]:
        print(f"{col}: {diff_df[col].mean():.{precision}f} Nm")

    # Create aggregate bar plots by subject
    plt.figure(figsize=(15, 10))

    # Get average differences by subject
    subject_avgs = diff_df.groupby("subject").mean()

    # Knee extension differences by subject
    plt.subplot(3, 1, 1)
    plt.bar(
        subject_avgs.index, subject_avgs["knee_ext_combined"], color="purple", alpha=0.7
    )
    plt.axhline(
        y=diff_df["knee_ext_combined"].mean(),
        color="k",
        linestyle="--",
        label=f'Overall Avg: {diff_df["knee_ext_combined"].mean():.{precision}f} Nm',
    )
    plt.title("Knee Extension Moment Differences by Subject (Normal - Weak)")
    plt.ylabel("Moment Difference (Nm)")
    plt.legend()
    plt.grid(True)

    # Hip extension differences by subject
    plt.subplot(3, 1, 2)
    plt.bar(
        subject_avgs.index, subject_avgs["hip_ext_combined"], color="green", alpha=0.7
    )
    plt.axhline(
        y=diff_df["hip_ext_combined"].mean(),
        color="k",
        linestyle="--",
        label=f'Overall Avg: {diff_df["hip_ext_combined"].mean():.{precision}f} Nm',
    )
    plt.title("Hip Extension Moment Differences by Subject (Normal - Weak)")
    plt.ylabel("Moment Difference (Nm)")
    plt.legend()
    plt.grid(True)

    # Ankle plantarflexion differences by subject
    plt.subplot(3, 1, 3)
    plt.bar(
        subject_avgs.index, subject_avgs["ankle_pf_combined"], color="orange", alpha=0.7
    )
    plt.axhline(
        y=diff_df["ankle_pf_combined"].mean(),
        color="k",
        linestyle="--",
        label=f'Overall Avg: {diff_df["ankle_pf_combined"].mean():.{precision}f} Nm',
    )
    plt.title("Ankle Plantarflexion Moment Differences by Subject (Normal - Weak)")
    plt.xlabel("Subject")
    plt.ylabel("Moment Difference (Nm)")
    plt.legend()
    plt.grid(True)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_subjects_differences_comparison.png"))
    plt.close()
    print(
        "All subjects differences comparison plot saved as all_subjects_differences_comparison.png"
    )

# Create combined CSV for statistical analysis
if all_data_for_stats:
    combined_df = pd.DataFrame(all_data_for_stats)
    combined_csv_path = os.path.join(
        output_dir, "combined_kinetics_data_for_statistics.csv"
    )
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\n{'='*50}")
    print(f"Combined data for statistical analysis saved to:")
    print(f"{os.path.abspath(combined_csv_path)}")
    print(f"{'='*50}")
    print(f"\nColumns in combined CSV:")
    print(f"{', '.join(combined_df.columns.tolist())}")
    print(f"\nTotal rows: {len(combined_df)}")
    print(f"Subjects: {combined_df['subject'].unique()}")
    print(f"Conditions: {combined_df['condition'].unique()}")
    print(f"Methods: {combined_df['method'].unique()}")
