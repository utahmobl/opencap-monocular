import os
import sys
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.utils_mot import load_mot


def run_grf_analysis(gt_path, pred_path, meta_path, output_path=None):

    output_df = pd.DataFrame()

    # Load metadata
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    # extract weight from metadata
    subj_weight = meta["mass_kg"]
    bodyweight_N = subj_weight * 9.81  # Convert kg to Newtons

    # Load ground truth data from .mot file
    gt_data = load_mot(gt_path)
    gt = pd.DataFrame(data=gt_data[0], columns=gt_data[1])

    # Load predicted data - can be .csv or .mot
    if pred_path.endswith(".mot"):
        pred_data = load_mot(pred_path)
        pred = pd.DataFrame(data=pred_data[0], columns=pred_data[1])
        # For .mot vs .mot, map simulation columns to ground truth columns
        column_mapping = {
            "R_ground_force_vx": "ground_force_right_vx",
            "R_ground_force_vy": "ground_force_right_vy",
            "R_ground_force_vz": "ground_force_right_vz",
            "L_ground_force_vx": "ground_force_left_vx",
            "L_ground_force_vy": "ground_force_left_vy",
            "L_ground_force_vz": "ground_force_left_vz",
        }

    else:  # It's a CSV
        pred = pd.read_csv(pred_path)
        pred.rename(
            columns={
                "time (sec)": "time",
                "GRF_x_right (N)": "GRF_x_right",
                "GRF_y_right (N)": "GRF_y_right",
                "GRF_z_right (N)": "GRF_z_right",
                "GRF_x_left (N)": "GRF_x_left",
                "GRF_y_left (N)": "GRF_y_left",
                "GRF_z_left (N)": "GRF_z_left",
            },
            inplace=True,
        )
        # Map column names from ground truth to predicted CSV
        column_mapping = {
            "R_ground_force_vx": "GRF_x_right",
            "R_ground_force_vy": "GRF_y_right",
            "R_ground_force_vz": "GRF_z_right",
            "L_ground_force_vx": "GRF_x_left",
            "L_ground_force_vy": "GRF_y_left",
            "L_ground_force_vz": "GRF_z_left",
        }

    gt_time = gt["time"]
    pred_time = pred["time"]

    min_time = max(gt_time.iloc[0], pred_time.iloc[0])
    max_time = min(gt_time.iloc[-1], pred_time.iloc[-1])

    # Filter time ranges
    gt = gt[(gt["time"] >= min_time) & (gt["time"] <= max_time)]
    pred = pred[(pred["time"] >= min_time) & (pred["time"] <= max_time)]

    # Calculate sampling periods
    gt_period = gt_time.diff().mean()
    pred_period = pred_time.diff().mean()

    pred_freq = 1 / pred_period
    gt_freq = 1 / gt_period

    # Downsample ground truth to match prediction frequency if needed
    if gt_period < pred_period:
        # Set time as index for resampling
        gt = gt.set_index("time")

        # Create a new time index matching prediction frequency
        new_index = pred[pred["time"].between(min_time, max_time)]["time"]

        # Resample and interpolate
        gt = gt.reindex(gt.index.union(new_index)).interpolate(method="linear")
        gt = gt.reindex(new_index)

        # Reset index to get time back as column
        gt = gt.reset_index()

    assert gt.shape[0] == pred.shape[0], "Lengths do not match"

    # Single string for the time
    min_time = str(min_time)
    max_time = str(max_time)
    time = f"{min_time} to {max_time}"
    print(f"Time: {time}")

    # Reset the index
    gt.reset_index(drop=True, inplace=True)
    pred.reset_index(drop=True, inplace=True)

    # Create interactive plot comparing GT and Predicted GRF
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=("Right X", "Right Y", "Right Z", "Left X", "Left Y", "Left Z"),
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    # Plot data for each force component
    force_components = [
        # (ground_truth_column, row, col, title)
        ("R_ground_force_vx", 1, 1, "Right X"),
        ("R_ground_force_vy", 1, 2, "Right Y"),
        ("R_ground_force_vz", 1, 3, "Right Z"),
        ("L_ground_force_vx", 2, 1, "Left X"),
        ("L_ground_force_vy", 2, 2, "Left Y"),
        ("L_ground_force_vz", 2, 3, "Left Z"),
    ]

    for gt_col, row, col, title in force_components:
        # Get the corresponding prediction column name from the mapping
        pred_col = column_mapping.get(gt_col)

        if gt_col in gt.columns and pred_col and pred_col in pred.columns:
            # Ground truth
            fig.add_trace(
                go.Scatter(
                    x=gt["time"],
                    y=gt[gt_col],
                    mode="lines",
                    name=f"GT {title}",
                    line=dict(color="blue", width=2),
                    showlegend=(row == 1 and col == 1),  # Show legend only once
                ),
                row=row,
                col=col,
            )

            # Predicted
            fig.add_trace(
                go.Scatter(
                    x=pred["time"],
                    y=pred[pred_col],
                    mode="lines",
                    name=f"Pred {title}",
                    line=dict(color="red", width=2, dash="dash"),
                    showlegend=(row == 1 and col == 1),  # Show legend only once
                ),
                row=row,
                col=col,
            )

    # Update layout
    fig.update_layout(
        title=f"Ground Truth vs Predicted GRF Comparison<br>Subject Weight: {subj_weight} kg",
        height=600,
        showlegend=True,
        legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
        font=dict(size=12),
    )

    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=2)
    fig.update_yaxes(title_text="Force (N)")

    # Save the plot
    plot_path = os.path.join(output_path, "grf_comparison.html")
    fig.write_html(plot_path)
    print(f"Interactive plot saved to: {plot_path}")

    output_df["time"] = [time]
    output_df["subject_weight_kg"] = [subj_weight]

    assert gt.shape[0] == pred.shape[0], "Lengths do not match"

    measurements = {
        "right_force": ["R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz"],
        "left_force": ["L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz"],
    }

    # Collect all individual MAE values for proper global calculation
    all_mae_force = []
    all_mae_force_bw = []

    # Collect MAE values by axis for axis-specific averages
    x_mae_force = []
    y_mae_force = []
    z_mae_force = []
    x_mae_force_bw = []
    y_mae_force_bw = []
    z_mae_force_bw = []

    for key, cols in measurements.items():
        mae_list = []
        mae_bw_list = []
        for col in cols:
            if col in gt.columns and column_mapping[col] in pred.columns:
                mae = (gt[col] - pred[column_mapping[col]]).abs().mean()
                mae_bw = (
                    mae / bodyweight_N
                ) * 100  # Convert to percentage of bodyweight

                # Convert to appropriate units and round
                if "force" in key:
                    mae = np.round(mae, 2)  # Force in N
                    mae_bw = np.round(mae_bw, 2)  # Percentage of bodyweight
                    all_mae_force.append(mae)
                    all_mae_force_bw.append(mae_bw)

                    # Categorize by axis
                    if "vx" in col:
                        x_mae_force.append(mae)
                        x_mae_force_bw.append(mae_bw)
                    elif "vy" in col:
                        y_mae_force.append(mae)
                        y_mae_force_bw.append(mae_bw)
                    elif "vz" in col:
                        z_mae_force.append(mae)
                        z_mae_force_bw.append(mae_bw)

                mae_list.append(mae)
                mae_bw_list.append(mae_bw)
                print(f"{col}: {mae} N ({mae_bw}% BW)")

                output_df[col] = [mae]
                output_df[f"{col}_pct_bw"] = [mae_bw]

    # Calculate global MAEs as the mean of all individual force component MAEs
    mae_global_force = np.mean(all_mae_force) if all_mae_force else 0
    mae_global_force_bw = np.mean(all_mae_force_bw) if all_mae_force_bw else 0

    mae_global_force = np.round(mae_global_force, 2)
    mae_global_force_bw = np.round(mae_global_force_bw, 2)

    # Calculate axis-specific MAEs
    mae_x_force = np.round(np.mean(x_mae_force), 2) if x_mae_force else 0
    mae_y_force = np.round(np.mean(y_mae_force), 2) if y_mae_force else 0
    mae_z_force = np.round(np.mean(z_mae_force), 2) if z_mae_force else 0

    mae_x_force_bw = np.round(np.mean(x_mae_force_bw), 2) if x_mae_force_bw else 0
    mae_y_force_bw = np.round(np.mean(y_mae_force_bw), 2) if y_mae_force_bw else 0
    mae_z_force_bw = np.round(np.mean(z_mae_force_bw), 2) if z_mae_force_bw else 0

    # Add statistics for each measurement
    for key, cols in measurements.items():
        for col in cols:
            if col in gt.columns and column_mapping[col] in pred.columns:
                diff = (gt[col] - pred[column_mapping[col]]).abs()
                diff_bw = (
                    diff / bodyweight_N
                ) * 100  # Convert to percentage of bodyweight

                min_val = np.round(diff.min(), 2)
                max_val = np.round(diff.max(), 2)
                mean_val = np.round(diff.mean(), 2)
                std_val = np.round(diff.std(), 2)

                min_val_bw = np.round(diff_bw.min(), 2)
                max_val_bw = np.round(diff_bw.max(), 2)
                mean_val_bw = np.round(diff_bw.mean(), 2)
                std_val_bw = np.round(diff_bw.std(), 2)

                output_df[f"{col}_min"] = [min_val]
                output_df[f"{col}_max"] = [max_val]
                output_df[f"{col}_mean"] = [mean_val]
                output_df[f"{col}_std"] = [std_val]
                output_df[f"{col}_min_pct_bw"] = [min_val_bw]
                output_df[f"{col}_max_pct_bw"] = [max_val_bw]
                output_df[f"{col}_mean_pct_bw"] = [mean_val_bw]
                output_df[f"{col}_std_pct_bw"] = [std_val_bw]

    # Add global MAEs to the dataframe
    output_df["global_mae_force_N"] = [mae_global_force]
    output_df["global_mae_force_pct_bw"] = [mae_global_force_bw]

    # Add axis-specific MAEs to the dataframe
    output_df["mae_x_force_N"] = [mae_x_force]
    output_df["mae_y_force_N"] = [mae_y_force]
    output_df["mae_z_force_N"] = [mae_z_force]
    output_df["mae_x_force_pct_bw"] = [mae_x_force_bw]
    output_df["mae_y_force_pct_bw"] = [mae_y_force_bw]
    output_df["mae_z_force_pct_bw"] = [mae_z_force_bw]

    print(f"Subject weight: {subj_weight} kg ({bodyweight_N:.1f} N)")
    print(f"Global MAE force: {mae_global_force} N ({mae_global_force_bw}% BW)")
    print(f"  X-axis MAE: {mae_x_force} N ({mae_x_force_bw}% BW)")
    print(f"  Y-axis MAE: {mae_y_force} N ({mae_y_force_bw}% BW)")
    print(f"  Z-axis MAE: {mae_z_force} N ({mae_z_force_bw}% BW)")

    results_path = os.path.join(output_path, "grf_error.txt")

    # Write results to a file
    with open(results_path, "w") as f:
        f.write(f"Subject weight: {subj_weight} kg ({bodyweight_N:.1f} N)\n")
        f.write(f"Global MAE force: {mae_global_force} N ({mae_global_force_bw}% BW)\n")
        f.write(f"  X-axis MAE: {mae_x_force} N ({mae_x_force_bw}% BW)\n")
        f.write(f"  Y-axis MAE: {mae_y_force} N ({mae_y_force_bw}% BW)\n")
        f.write(f"  Z-axis MAE: {mae_z_force} N ({mae_z_force_bw}% BW)\n")
        f.write("\nIndividual component errors:\n")
        for key, cols in measurements.items():
            for col in cols:
                if col in gt.columns and column_mapping[col] in pred.columns:
                    mae = np.round(
                        (gt[col] - pred[column_mapping[col]]).abs().mean(), 2
                    )
                    mae_bw = np.round((mae / bodyweight_N) * 100, 2)
                    f.write(f"{col}: {mae} N ({mae_bw}% BW)\n")

    # Write CSV file from the dataframe
    output_csv_path = os.path.join(output_path, "grfs_analysis.csv")

    output_df.to_csv(output_csv_path, index=False)

    return (
        mae_global_force,
        mae_global_force_bw,
        results_path,
        output_csv_path,
        plot_path,
    )


def categorize_trial(trial_name):
    """Categorize trial into activity type"""
    if trial_name.startswith("squats1"):
        return "squat"
    elif trial_name.startswith("squatsAsym"):
        return "squatAsym"
    elif trial_name.startswith("STS1"):
        return "STS"
    elif trial_name.startswith("STSweakLegs"):
        return "STSweakLegs"
    elif trial_name.startswith("walking") and not trial_name.startswith("walkingTS"):
        return "walking"
    elif trial_name.startswith("walkingTS"):
        return "walkingTS"
    else:
        return "unknown"


if __name__ == "__main__":
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Choose analysis type
    print("Choose analysis type:")
    print("1. Standard GRF analysis (all activities)")
    print("2. Simulation GRF analysis (STS only)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        # Simulation analysis parameters
        simulation_analysis = True
        # Prediction files for simulation are in the case folder
        predictions_path = "output/june_24/case_001_STS"
        dataset = "LabValidation_withVideos1"
        sync = True

        # keep this to 2,12. Do not change
        subjects = [f"subject{i}" for i in range(2, 12)]

        # Main trials that have sub-events in simulations
        main_trials = ["STS1", "STSweakLegs1"]

        # Each main trial has 3 sub-events
        sub_events = ["1", "2", "3"]

        print(f"Running simulation GRF analysis for STS activities with sub-events...")

    else:
        # Standard analysis parameters
        simulation_analysis = False
        predictions_path = "output/June_24_Janelle"
        dataset = "LabValidation_withVideos1"
        sync = True

        subjects = [f"subject{i}" for i in range(2, 12)]

        # List of all trials for standard analysis
        trials = [
            "squats1",
            "squatsAsym1",
            "STS1",
            "STSweakLegs1",
            "walking1",
            "walking2",
            "walking3",
            "walking4",
            "walkingTS1",
            "walkingTS2",
            "walkingTS3",
        ]

    grf_results_force_bw = []
    all_results_df = pd.DataFrame()
    count = 0

    if simulation_analysis:
        # Process simulation analysis with sub-events
        for subject in subjects:
            for main_trial in main_trials:
                for sub_event in sub_events:
                    try:
                        # Ground truth is always the main trial (e.g., STS1_forces.mot)
                        gt_path = (
                            f"{dataset}/{subject}/ForceData/{main_trial}_forces.mot"
                        )

                        # Construct the simulation sub-event name
                        sim_subject_trial = f"{subject}_STS_{sub_event}"

                        # Path to the simulation results for this sub-event
                        sim_path_base = f"output/june_24/case_001_STS/{subject}/Session0/Cam3/{main_trial}/OpenSim/Simulations/{sim_subject_trial}/OpenSimData/Dynamics/{sim_subject_trial}"
                        pred_path = f"{sim_path_base}/GRF_resultant_{sim_subject_trial}_case_001_STS.mot"

                        meta_path = f"{dataset}/{subject}/sessionMetadata.yaml"

                        # Create a unique trial identifier for the sub-event
                        trial_id = f"{main_trial}_sub{sub_event}"
                        output_path = (
                            f"output/june_24/simulation_analysis/{subject}/{trial_id}"
                        )

                        # Create output directory if it doesn't exist
                        os.makedirs(output_path, exist_ok=True)

                        (
                            grf_force_N,
                            grf_force_bw,
                            results_path,
                            output_csv_path,
                            plot_path,
                        ) = run_grf_analysis(
                            gt_path, pred_path, meta_path, output_path=output_path
                        )
                        grf_results_force_bw.append(grf_force_bw)

                        # Load the individual results CSV and add subject/trial info
                        trial_results = pd.read_csv(output_csv_path)
                        trial_results["subject"] = subject
                        trial_results["trial"] = (
                            trial_id  # Use the sub-event identifier
                        )
                        trial_results["main_trial"] = (
                            main_trial  # Keep track of the main trial
                        )
                        trial_results["sub_event"] = (
                            sub_event  # Keep track of the sub-event
                        )
                        trial_results["activity"] = categorize_trial(
                            main_trial
                        )  # Categorize based on main trial

                        # Add to master dataframe
                        all_results_df = pd.concat(
                            [all_results_df, trial_results], ignore_index=True
                        )

                        count += 1

                        print(f"Results saved to: {output_csv_path}")
                        print(f"-----------------------------------")
                        print("\n")
                    except Exception as e:
                        print(
                            f"Error processing {subject}/{main_trial} sub-event {sub_event}: {e}"
                        )
                        print(f"-----------------------------------")
                        print("\n")
    else:
        # Standard analysis processing
        for subject in subjects:
            for trial in trials:
                try:
                    # Standard analysis paths
                    gt_path = f"{dataset}/{subject}/ForceData/{trial}_forces.mot"
                    if sync:
                        pred_path = f"{predictions_path}/{subject}/{trial}/sync_{trial}_pred_grfs.csv"
                    else:
                        pred_path = f"{predictions_path}/{subject}/{trial}/{trial}_pred_grfs.csv"
                    meta_path = f"{dataset}/{subject}/sessionMetadata.yaml"
                    output_path = f"{predictions_path}/{subject}/{trial}"

                    # Create output directory if it doesn't exist
                    os.makedirs(output_path, exist_ok=True)

                    (
                        grf_force_N,
                        grf_force_bw,
                        results_path,
                        output_csv_path,
                        plot_path,
                    ) = run_grf_analysis(
                        gt_path, pred_path, meta_path, output_path=output_path
                    )
                    grf_results_force_bw.append(grf_force_bw)

                    # Load the individual results CSV and add subject/trial info
                    trial_results = pd.read_csv(output_csv_path)
                    trial_results["subject"] = subject
                    trial_results["trial"] = trial
                    trial_results["activity"] = categorize_trial(trial)

                    # Add to master dataframe
                    all_results_df = pd.concat(
                        [all_results_df, trial_results], ignore_index=True
                    )

                    count += 1

                    print(f"Results saved to: {output_csv_path}")
                    print(f"-----------------------------------")
                    print("\n")
                except Exception as e:
                    print(f"Error processing {subject}/{trial}: {e}")
                    print(f"-----------------------------------")
                    print("\n")

    # Save consolidated results to CSV
    if simulation_analysis:
        consolidated_csv_path = os.path.join(
            "output/june_24/simulation_analysis",
            "consolidated_simulation_grf_analysis.csv",
        )
        os.makedirs("output/june_24/simulation_analysis", exist_ok=True)
    else:
        consolidated_csv_path = os.path.join(
            predictions_path, "consolidated_grf_analysis.csv"
        )

    # Only proceed if we have results
    if not all_results_df.empty:
        # Reorder columns to have subject, trial, and activity first
        if simulation_analysis:
            # For simulation analysis, include main_trial and sub_event columns
            cols = ["subject", "trial", "main_trial", "sub_event", "activity"] + [
                col
                for col in all_results_df.columns
                if col
                not in ["subject", "trial", "main_trial", "sub_event", "activity"]
            ]
        else:
            cols = ["subject", "trial", "activity"] + [
                col
                for col in all_results_df.columns
                if col not in ["subject", "trial", "activity"]
            ]
        all_results_df = all_results_df[cols]

        all_results_df.to_csv(consolidated_csv_path, index=False)
        print(f"Consolidated results saved to: {consolidated_csv_path}")
    else:
        print(f"No results to save - all trials failed to process")

    grf_results_force_bw_avg = np.round(np.mean(grf_results_force_bw), 2)
    analysis_type = "Simulation" if simulation_analysis else "Standard"
    print(
        f"{analysis_type} Force Error in BW (average): {grf_results_force_bw_avg}% BW over {count} trials"
    )

    # Print summary statistics
    if not all_results_df.empty:
        print(f"\n{analysis_type} Analysis Summary Statistics:")
        print(f"Total trials processed: {len(all_results_df)}")
        print(f"Subjects: {sorted(all_results_df['subject'].unique())}")
        print(f"Trials: {sorted(all_results_df['trial'].unique())}")
        print(f"Activities: {sorted(all_results_df['activity'].unique())}")

        print(f"\nOverall Averages:")
        print(
            f"Average global MAE force: {all_results_df['global_mae_force_N'].mean():.2f} N"
        )
        print(
            f"Average global MAE force (% BW): {all_results_df['global_mae_force_pct_bw'].mean():.2f}% BW"
        )
        print(
            f"  Average X-axis MAE: {all_results_df['mae_x_force_N'].mean():.2f} N ({all_results_df['mae_x_force_pct_bw'].mean():.2f}% BW)"
        )
        print(
            f"  Average Y-axis MAE: {all_results_df['mae_y_force_N'].mean():.2f} N ({all_results_df['mae_y_force_pct_bw'].mean():.2f}% BW)"
        )
        print(
            f"  Average Z-axis MAE: {all_results_df['mae_z_force_N'].mean():.2f} N ({all_results_df['mae_z_force_pct_bw'].mean():.2f}% BW)"
        )

        print(f"\nActivity-Specific Averages:")
        activity_stats = (
            all_results_df.groupby("activity")
            .agg(
                {
                    "global_mae_force_N": "mean",
                    "global_mae_force_pct_bw": "mean",
                    "mae_x_force_N": "mean",
                    "mae_y_force_N": "mean",
                    "mae_z_force_N": "mean",
                    "mae_x_force_pct_bw": "mean",
                    "mae_y_force_pct_bw": "mean",
                    "mae_z_force_pct_bw": "mean",
                }
            )
            .round(2)
        )

        for activity in sorted(all_results_df["activity"].unique()):
            if activity in activity_stats.index:
                stats = activity_stats.loc[activity]
                n_trials = len(all_results_df[all_results_df["activity"] == activity])
                print(f"\n{activity} ({n_trials} trials):")
                print(
                    f"  Global MAE: {stats['global_mae_force_N']:.2f} N ({stats['global_mae_force_pct_bw']:.2f}% BW)"
                )
                print(
                    f"    X-axis: {stats['mae_x_force_N']:.2f} N ({stats['mae_x_force_pct_bw']:.2f}% BW)"
                )
                print(
                    f"    Y-axis: {stats['mae_y_force_N']:.2f} N ({stats['mae_y_force_pct_bw']:.2f}% BW)"
                )
                print(
                    f"    Z-axis: {stats['mae_z_force_N']:.2f} N ({stats['mae_z_force_pct_bw']:.2f}% BW)"
                )

        # Create activity-specific bar plots
        print(f"\nCreating activity-specific visualization...")

        # Prepare data for plotting
        activities = sorted(all_results_df["activity"].unique())

        # Global MAE bar plot
        fig_global = go.Figure()

        global_mae_n = [
            activity_stats.loc[act, "global_mae_force_N"] for act in activities
        ]
        global_mae_bw = [
            activity_stats.loc[act, "global_mae_force_pct_bw"] for act in activities
        ]
        trial_counts = [
            len(all_results_df[all_results_df["activity"] == act]) for act in activities
        ]

        fig_global.add_trace(
            go.Bar(
                x=activities,
                y=global_mae_n,
                name="Global MAE",
                text=[
                    f"{mae:.1f} N<br>({bw:.1f}% BW)<br>n={n}"
                    for mae, bw, n in zip(global_mae_n, global_mae_bw, trial_counts)
                ],
                textposition="auto",
                marker_color="steelblue",
            )
        )

        fig_global.update_layout(
            title="Ground Reaction Force Errors by Activity Type<br><sub>Global MAE (Mean Absolute Error)</sub>",
            xaxis_title="Activity Type",
            yaxis_title="MAE (N)",
            height=500,
            showlegend=False,
            font=dict(size=12),
        )

        # Save global MAE plot
        if simulation_analysis:
            plot_base_path = "output/june_24/simulation_analysis"
        else:
            plot_base_path = predictions_path

        global_plot_path = os.path.join(plot_base_path, "activity_global_mae.html")
        fig_global.write_html(global_plot_path)

        # Axis-specific stacked bar plot
        fig_axes = go.Figure()

        x_mae_n = [activity_stats.loc[act, "mae_x_force_N"] for act in activities]
        y_mae_n = [activity_stats.loc[act, "mae_y_force_N"] for act in activities]
        z_mae_n = [activity_stats.loc[act, "mae_z_force_N"] for act in activities]

        fig_axes.add_trace(
            go.Bar(
                x=activities,
                y=x_mae_n,
                name="X-axis (Medio-Lateral)",
                marker_color="lightcoral",
            )
        )

        fig_axes.add_trace(
            go.Bar(
                x=activities,
                y=y_mae_n,
                name="Y-axis (Anterior-Posterior)",
                marker_color="gold",
            )
        )

        fig_axes.add_trace(
            go.Bar(
                x=activities,
                y=z_mae_n,
                name="Z-axis (Vertical)",
                marker_color="lightseagreen",
            )
        )

        fig_axes.update_layout(
            title="Ground Reaction Force Errors by Activity Type and Axis<br><sub>MAE (Mean Absolute Error) by Force Direction</sub>",
            xaxis_title="Activity Type",
            yaxis_title="MAE (N)",
            height=500,
            barmode="group",
            font=dict(size=12),
            legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
        )

        # Save axis-specific plot
        axes_plot_path = os.path.join(plot_base_path, "activity_axes_mae.html")
        fig_axes.write_html(axes_plot_path)

        # Percentage body weight plot
        fig_bw = go.Figure()

        x_mae_bw = [activity_stats.loc[act, "mae_x_force_pct_bw"] for act in activities]
        y_mae_bw = [activity_stats.loc[act, "mae_y_force_pct_bw"] for act in activities]
        z_mae_bw = [activity_stats.loc[act, "mae_z_force_pct_bw"] for act in activities]

        fig_bw.add_trace(
            go.Bar(
                x=activities,
                y=x_mae_bw,
                name="X-axis (Medio-Lateral)",
                marker_color="lightcoral",
            )
        )

        fig_bw.add_trace(
            go.Bar(
                x=activities,
                y=y_mae_bw,
                name="Y-axis (Anterior-Posterior)",
                marker_color="gold",
            )
        )

        fig_bw.add_trace(
            go.Bar(
                x=activities,
                y=z_mae_bw,
                name="Z-axis (Vertical)",
                marker_color="lightseagreen",
            )
        )

        fig_bw.update_layout(
            title="Ground Reaction Force Errors by Activity Type and Axis<br><sub>MAE as Percentage of Body Weight</sub>",
            xaxis_title="Activity Type",
            yaxis_title="MAE (% Body Weight)",
            height=500,
            barmode="group",
            font=dict(size=12),
            legend=dict(x=0, y=1, bgcolor="rgba(255,255,255,0.8)"),
        )

        # Save percentage body weight plot
        bw_plot_path = os.path.join(plot_base_path, "activity_axes_mae_bw.html")
        fig_bw.write_html(bw_plot_path)

        print(f"{analysis_type} analysis visualizations saved:")
        print(f"  Global MAE plot: {global_plot_path}")
        print(f"  Axis-specific MAE plot: {axes_plot_path}")
        print(f"  Percentage body weight plot: {bw_plot_path}")
