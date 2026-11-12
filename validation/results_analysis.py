import pandas as pd
import plotly.express as px
import numpy as np
import plot_config as pc

ik_results_mono = pd.read_csv("analysis_results/ik_results_mono.csv")
ik_results_twocam = pd.read_csv("analysis_results/ik_results_twocam.csv")
ik_results_wham = pd.read_csv("analysis_results/ik_results_wham.csv")

# filter by movement
walking_results = ik_results_mono[ik_results_mono["Movement"] == "walking"]
sts_results = ik_results_mono[ik_results_mono["Movement"] == "STS"]
squats_results = ik_results_mono[ik_results_mono["Movement"] == "squat"]

movements = ["walking", "STS", "squat"]

for movement in movements:
    print("\n")
    results = ik_results_mono[ik_results_mono["Movement"] == movement]
    print(f'{movement} Mean Global MAE Degrees: {results["global_mae_degrees"].mean()}')
    print("\n")

    print("MAE above 5.0:")
    results_mae_above_5 = results[results["global_mae_degrees"] > 5.0]
    print(
        results_mae_above_5[
            ["Subject", "Camera", "Movement", "Trial", "global_mae_degrees"]
        ]
    )
    print("===============================================")

    # now let's find the values in each column that are above 6.0. not just for global_mae_degrees, but for all columns
    print(f"\nRows for {movement} with any column value > 6.0:")
    numeric_cols = results.select_dtypes(include="number").columns
    filtered_cols = [
        col
        for col in numeric_cols
        if "_tx" not in col
        and "_ty" not in col
        and "_tz" not in col
        and "mm" not in col
        and "Camera" not in col
        and "global" not in col
    ]
    rows_with_high_values = results[(results[filtered_cols] > 6.0).any(axis=1)]
    for _, row in rows_with_high_values.iterrows():
        high_values = row[filtered_cols][row[filtered_cols] > 6.0]
        identifiers = row[["Subject", "Camera", "Movement", "Trial"]]
        print(pd.concat([identifiers, high_values]))
        print("---")
    print("===============================================")

# Option to plot average over activities (default: True)
plot_average_over_activities = True

# Create combined plot with DOF as x-axis labels
# Get the filtered columns (DOFs) from the first movement
results = ik_results_mono[ik_results_mono["Movement"] == movements[0]]
numeric_cols = results.select_dtypes(include="number").columns
filtered_cols = [
    col
    for col in numeric_cols
    if "_tx" not in col
    and "_ty" not in col
    and "_tz" not in col
    and "mm" not in col
    and "Camera" not in col
    and "global" not in col
]

# Create data for the combined plot
plot_data = []
stats_data = []
if plot_average_over_activities:
    # Plot average over all activities
    for dof in filtered_cols:
        # Calculate average and std across all movements for each method
        mono_errors = []
        twocam_errors = []
        wham_errors = []
        
        for movement in movements:
            results_mono = ik_results_mono[ik_results_mono["Movement"] == movement]
            results_twocam = ik_results_twocam[ik_results_twocam["Movement"] == movement]
            results_wham = ik_results_wham[ik_results_wham["Movement"] == movement]
            
            mono_errors.append(results_mono[dof].mean())
            twocam_errors.append(results_twocam[dof].mean())
            wham_errors.append(results_wham[dof].mean())
        
        # Calculate mean and std
        twocam_mean = np.mean(twocam_errors)
        twocam_std = np.std(twocam_errors)
        wham_mean = np.mean(wham_errors)
        wham_std = np.std(wham_errors)
        mono_mean = np.mean(mono_errors)
        mono_std = np.std(mono_errors)
        
        # Add data for each method in desired order: Two-camera, WHAM, Mono

        plot_data.append({
            'DOF': dof,
            'Method': 'WHAM',
            'Error': wham_mean,
            'Error_std': wham_std
        })
        plot_data.append({
            'DOF': dof,
            'Method': 'OpenCap Monocular',
            'Error': mono_mean,
            'Error_std': mono_std
        })
        plot_data.append({
            'DOF': dof,
            'Method': 'OpenCap Two-camera',
            'Error': twocam_mean,
            'Error_std': twocam_std
        })
        
        # Store stats for export
        stats_data.append({
            'DOF': dof,
            'OpenCap_Two_camera_mean': twocam_mean,
            'OpenCap_Two_camera_std': twocam_std,
            'WHAM_mean': wham_mean,
            'WHAM_std': wham_std,
            'OpenCap_Monocular_mean': mono_mean,
            'OpenCap_Monocular_std': mono_std
        })
else:
    # Plot individual activities
    for dof in filtered_cols:
        for movement in movements:
            results_mono = ik_results_mono[ik_results_mono["Movement"] == movement]
            results_twocam = ik_results_twocam[ik_results_twocam["Movement"] == movement]
            results_wham = ik_results_wham[ik_results_wham["Movement"] == movement]
            
            # Add data for each method in desired order: Two-camera, WHAM, Mono


            plot_data.append({
                'DOF': dof,
                'Movement': movement,
                'Method': 'WHAM',
                'Error': results_wham[dof].mean()
            })
            plot_data.append({
                'DOF': dof,
                'Movement': movement,
                'Method': 'Mono',
                'Error': results_mono[dof].mean()
            })
            plot_data.append({
                'DOF': dof,
                'Movement': movement,
                'Method': 'Two-camera',
                'Error': results_twocam[dof].mean()
            })


# Create DataFrame for plotting
plot_df = pd.DataFrame(plot_data)

# Create the combined plot
if plot_average_over_activities:
    fig = px.bar(
        plot_df,
        x='DOF',
        y='Error',
        color='Method',
        barmode='group',
        error_y='Error_std',
        # title="Mean Rotation Errors by DOF (Average over Activities)",
        labels={"DOF": "Degree of Freedom", "Error": "Error (degrees)"},
        color_discrete_map={
            'OpenCap Two-camera': '#FFA500',  # Orange for 2CAMS
            'WHAM': '#e74c3c',                # Red for WHAM
            'OpenCap Monocular': '#3498db'    # Blue for Mono
        }
    )
else:
    fig = px.bar(
        plot_df,

        x='DOF',
        y='Error',
        color='Method',
        pattern_shape='Movement',
        barmode='group',
        # title="Mean Rotation Errors by DOF and Movement",
        labels={"DOF": "Degree of Freedom", "Error": "Error (degrees)"},
        color_discrete_map={
            'OpenCap Two-camera': '#FFA500',  # Orange for 2CAMS
            'WHAM': '#e74c3c',                # Red for WHAM
            'OpenCap Monocular': '#3498db'    # Blue for Mono
        }
    )

# Add horizontal lines for average across all DOFs
if plot_average_over_activities:
    # Calculate overall average for each method across all DOFs
    overall_avg_twocam = plot_df[plot_df['Method'] == 'OpenCap Two-camera']['Error'].mean()
    overall_avg_wham = plot_df[plot_df['Method'] == 'WHAM']['Error'].mean()
    overall_avg_mono = plot_df[plot_df['Method'] == 'OpenCap Monocular']['Error'].mean()
    
    # Add horizontal lines using colors from error_analysis.py
    # WHAM (red), Mono (blue), 2CAMS (orange)
    # fig.add_hline(
    #     y=overall_avg_twocam,
    #     line_dash="solid",
    #     annotation_text=f"OpenCap Two-camera Avg: {overall_avg_twocam:.2f}°",
    #     annotation_position="bottom right",
    #     line_color="#FFA500",  # Orange for 2CAMS
    #     opacity=0.7
    # )
    # fig.add_hline(
    #     y=overall_avg_wham,
    #     line_dash="solid",
    #     annotation_text=f"WHAM Avg: {overall_avg_wham:.2f}°",
    #     annotation_position="bottom right",
    #     line_color="#e74c3c",  # Red for WHAM
    #     opacity=0.7
    # )
    # fig.add_hline(
    #     y=overall_avg_mono,
    #     line_dash="solid",
    #     annotation_text=f"OpenCap Monocular Avg: {overall_avg_mono:.2f}°",
    #     annotation_position="bottom right",
    #     line_color="#3498db",  # Blue for Mono
    #     opacity=0.7
    # )

# Apply the updated styling configuration
pc.apply_standard_layout(
    fig,
    # title="Mean Rotation Errors by DOF (Average over Activities)",
    yaxis_title="MAE (degrees)"
)

# Remove legend title
fig.update_layout(legend_title_text="")

# Customize x-axis tick labels
def format_tick_label(label):
    # Replace underscores with spaces
    formatted = label.replace("_", " ")
    
    # Replace standalone "r" with "right" and "l" with "left"
    import re
    # Pattern to match "r" or "l" that are not followed by a letter
    formatted = re.sub(r'\br\b(?!\w)', 'right', formatted)
    formatted = re.sub(r'\bl\b(?!\w)', 'left', formatted)
    
    return formatted

# Get current tick labels and format them
fig.update_xaxes(
    tickangle=45,
    ticktext=[format_tick_label(label) for label in filtered_cols],
    tickvals=list(range(len(filtered_cols))),
    title_text=""  # Remove x-axis title
)
# Update error bar styling and bar spacing
fig.update_traces(
    error_y=dict(
        color='grey',
        thickness=0.5,
        width=2
    )
)

# Increase horizontal spacing between DOF groups
fig.update_layout(
    bargap=0.3,  # Space between groups of bars
    bargroupgap=0.1  # Space between bars within each group
)

fig.write_html("analysis_results/combined_mean_errors.html")
fig.write_image("analysis_results/combined_mean_errors.svg", format="svg")
fig.write_image("analysis_results/combined_mean_errors.png", format="png", width=800, height=600, scale=3)
fig.show()

# Export statistics to CSV
stats_df = pd.DataFrame(stats_data)
stats_df.to_csv("analysis_results/combined_mean_errors_stats.csv", index=False)
print("Statistics exported to analysis_results/combined_mean_errors_stats.csv")

# print the mean of the global_mae_mm column
