import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
from scipy import stats
import plot_config as pc

# Load data
csv_path = "validation/MAE_Table.csv"
df = pd.read_csv(csv_path)

# Define the methods and their corresponding column patterns
methods = {
    'WHAM': 'Wham MAE GRF',
    'OpenCap Monocular + ML': 'Mono MAE GRF',
    'OpenCap 2 cameras + Simulations': 'Orig OpenCap MAE GRF', 
    'OpenCap 2 cameras + ML': 'OpenCap + ML (no feet fix) MAE GRF',
}

# GRF components with anatomical labels
components = ['y', 'x', 'z']
component_labels = {
    'y': 'Vertical GRF MAE',  # Vertical (Y is typically vertical in biomechanics)
    'x': 'Mediolateral GRF MAE',  # Medial-Lateral
    'z': 'Anteroposterior GRF MAE'   # Anterior-Posterior
}

# Colors for each method - using standardized colors
colors = [pc.COLORS['wham'], pc.COLORS['opencap'], pc.COLORS['mocap'], pc.COLORS['knee'], pc.COLORS['hip']]  

# Create subplots for each GRF component
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[component_labels[comp] for comp in components]
)

# Calculate statistics and create bar plots for each component
for i, component in enumerate(components):
    method_names = list(methods.keys())
    means = []
    stds = []
    all_errors = []
    
    # Calculate means and stds for each method
    for method_name, col_pattern in methods.items():
        col_name = f"{col_pattern} {component}"
        values = df[col_name].values
        means.append(values.mean())
        stds.append(values.std())
        all_errors.append(values)
    
    # Add bar traces for each method
    for j, (method, mean, std, errors) in enumerate(zip(method_names, means, stds, all_errors)):
        # Create hover text with individual data points
        hover_text = f"Method: {method}<br>Mean: {mean:.3f} %BW<br>Std: {std:.3f} %BW<br>N trials: {len(errors)}"
        
        fig.add_trace(
            go.Bar(
                x=[method],
                y=[mean],
                error_y=dict(type='data', array=[std], visible=True),
                name=method if i == 0 else "",  # Only show legend for first subplot
                showlegend=True if i == 0 else False,
                marker_color=colors[j],
                opacity=0.7,
                hovertemplate=hover_text + "<extra></extra>"
            ),
            row=1, col=i+1
        )
        
        # Add annotation above the error bars
        annotation_y = mean + 0.2 # Position well above the error bar
        # Offset annotation slightly to the right to avoid overlap with error bars
        annotation_x_offset = 0.2
        annotation_config = pc.ANNOTATION_CONFIG.copy()
        annotation_config.update({
            'x': j + annotation_x_offset,  # Use bar index with offset instead of method name
            'y': annotation_y,
            'text': f'{mean:.2f}',
            'xref': f"x{i+1}",
            'yref': f"y{i+1}"
        })
        fig.add_annotation(**annotation_config)

# Update layout using standardized configuration
layout_config = {
    'font': {'size': pc.FONTS['axis_title_size'], 'family': pc.FONTS['family']},
    'plot_bgcolor': pc.COLORS['background'],
    'paper_bgcolor': pc.COLORS['background'],
    'height': 700,  # Increased height to accommodate legend
    'width': 1400,
    'showlegend': True,
    'legend': dict(
        orientation="v",
        yanchor="top",
        y=0.88,  # Position legend at top right inside plot area
        xanchor="right",
        x=0.98,
        font={'size': pc.FONTS['legend_size'], 'family': pc.FONTS['family']},
        bgcolor="rgba(255,255,255,0.8)",  # Semi-transparent white background
        # bordercolor="black",
        # borderwidth=0
    ),
    'margin': dict(t=80)  # Reduced top margin since legend is now inside
}
fig.update_layout(**layout_config)

# Update y-axis labels and styling for each subplot using standardized configuration
axis_config = pc.get_standard_axes()
for i, component in enumerate(components):
    # Only show y-axis label on the first (leftmost) plot
    if i == 0:
        fig.update_yaxes(
            title_text='Error (%BW) compared to force plates',
            title_font={'size': pc.FONTS['axis_title_size'], 'family': pc.FONTS['family']},
            range=[0, 45], 
            row=1, col=i+1,
            **axis_config
        )
    elif i == 1:
        fig.update_yaxes(
            title_text='',
            range=[0, 11], 
            row=1, col=i+1,
            **axis_config
        )
    elif i == 2:
        fig.update_yaxes(
            title_text='',
            range=[0, 3], 
            row=1, col=i+1,
            **axis_config
        )
    fig.update_xaxes(row=1, col=i+1, **axis_config)

# Statistical significance annotations removed as requested

fig.show()

# Create overall GRF magnitude comparison
overall_data = []
for method_name, col_pattern in methods.items():
    # Calculate Euclidean norm across x, y, z components for each trial
    x_vals = df[f"{col_pattern} x"].values
    y_vals = df[f"{col_pattern} y"].values  
    z_vals = df[f"{col_pattern} z"].values
    
    overall_errors = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)
    overall_data.extend([{
        'Method': method_name,
        'Overall_Error': error,
        'Subject': df['Subject'].iloc[i],
        'Trial': df['Trial'].iloc[i]
    } for i, error in enumerate(overall_errors)])

overall_df = pd.DataFrame(overall_data)

# Create box plot with individual points
fig2 = go.Figure()

for i, method in enumerate(methods.keys()):
    method_data = overall_df[overall_df['Method'] == method]
    
    # Add box plot
    fig2.add_trace(go.Box(
        y=method_data['Overall_Error'],
        x=[method] * len(method_data),
        name=method,
        marker_color=colors[i],
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        hovertemplate=f"Method: {method}<br>Overall Error: %{{y:.3f}} %BW<br>Subject: %{{customdata[0]}}<br>Trial: %{{customdata[1]}}<extra></extra>",
        customdata=method_data[['Subject', 'Trial']].values
    ))
    
    # Add mean value annotation
    mean_val = method_data['Overall_Error'].mean()
    std_val = method_data['Overall_Error'].std()


    fig2.add_annotation(
        x=i,
        y=mean_val,
        text=f"μ={mean_val:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=colors[i],
        bgcolor="white",
        bordercolor=colors[i],
        borderwidth=2
    )

# Apply standardized layout to second figure
layout_config2 = {
    'font': {'size': pc.FONTS['axis_title_size'], 'family': pc.FONTS['family']},
    'plot_bgcolor': pc.COLORS['background'],
    'paper_bgcolor': pc.COLORS['background'],
    'yaxis_title': {
        'text': 'Overall GRF Error Magnitude (%BW)',
        'font': {'size': pc.FONTS['axis_title_size'], 'family': pc.FONTS['family']}
    },
    'height': 600,
    'width': 1000,
    'showlegend': False
}
fig2.update_layout(**layout_config2)

# Apply standardized axis configuration
fig2.update_yaxes(range=[0, 45], **axis_config)
fig2.update_xaxes(**axis_config)

fig2.show()

# Print summary statistics
print("=== GRF Error Summary Statistics ===")
for component in components:
    print(f"\n{component_labels[component]}:")
    for method_name, col_pattern in methods.items():
        col_name = f"{col_pattern} {component}"
        values = df[col_name]
        print(f"  {method_name}: {values.mean():.3f} ± {values.std():.3f} %BW")

print(f"\nOverall GRF Error (Magnitude):")
for method in methods.keys():
    method_data = overall_df[overall_df['Method'] == method]['Overall_Error']
    print(f"  {method}: {method_data.mean():.3f} ± {method_data.std():.3f} %BW")

# Paired t-tests between WHAM and Mono for each GRF component
print("\n" + "="*80)
print("PAIRED T-TEST: WHAM vs OpenCap Monocular + ML")
print("="*80)
alpha = 0.05

# Option: aggregate to subject level (default) or use all trials
# Set to False if trials are truly independent and you want n=30
aggregate_to_subject = True  # Default: aggregate to avoid pseudoreplication

if aggregate_to_subject:
    print("\nAggregating to subject level (averaging trials within each subject)...")
    print("  This avoids pseudoreplication and uses n = number of subjects")
    print("  Set aggregate_to_subject = False to use all trials (n = number of trials)")
    
    # Aggregate to subject level by averaging trials within each subject
    agg_df = df.groupby('Subject').agg({
        f"{methods['WHAM']} y": 'mean',
        f"{methods['WHAM']} x": 'mean',
        f"{methods['WHAM']} z": 'mean',
        f"{methods['OpenCap Monocular + ML']} y": 'mean',
        f"{methods['OpenCap Monocular + ML']} x": 'mean',
        f"{methods['OpenCap Monocular + ML']} z": 'mean',
        'Subject': 'first'
    }).reset_index(drop=True)
    
    print(f"  After aggregation: {len(agg_df)} subjects (from {len(df)} trials)")
    analysis_df = agg_df
    unit_label = "subjects"
else:
    print("\nUsing all trials (assuming independence)...")
    print("  WARNING: This assumes trials within subjects are independent")
    analysis_df = df
    unit_label = "trials"

for component in components:
    # Get column names for WHAM and Mono
    wham_col = f"{methods['WHAM']} {component}"
    mono_col = f"{methods['OpenCap Monocular + ML']} {component}"
    
    # Extract paired data (remove any NaN pairs)
    wham_vals = analysis_df[wham_col].values
    mono_vals = analysis_df[mono_col].values
    
    # Create pairs and remove any with NaN values
    pairs = [(w, m) for w, m in zip(wham_vals, mono_vals) if not (np.isnan(w) or np.isnan(m))]
    
    if len(pairs) < 2:
        print(f"\n{component_labels[component]}:")
        print(f"  Insufficient data for paired t-test (n = {len(pairs)})")
        continue
    
    wham_paired = np.array([w for w, m in pairs])
    mono_paired = np.array([m for w, m in pairs])
    differences = wham_paired - mono_paired
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(wham_paired, mono_paired)
    
    # Calculate statistics
    n = len(pairs)  # Number of pairs
    dof = n - 1     # Degrees of freedom for paired t-test
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)  # Sample standard deviation
    se_diff = std_diff / np.sqrt(n)  # Standard error of the mean difference
    
    # Calculate 95% confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, dof)
    ci_lower = mean_diff - t_critical * se_diff
    ci_upper = mean_diff + t_critical * se_diff
    
    # Determine significance
    significant = p_value < alpha
    
    # Print results
    print(f"\n{component_labels[component]}:")
    print(f"  N pairs: {n} {unit_label}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  WHAM mean: {wham_paired.mean():.3f} ± {wham_paired.std(ddof=1):.3f} %BW")
    print(f"  Mono mean: {mono_paired.mean():.3f} ± {mono_paired.std(ddof=1):.3f} %BW")
    print(f"  Mean difference (WHAM - Mono): {mean_diff:.3f} ± {se_diff:.3f} %BW")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}] %BW")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant (α = {alpha}): {'Yes' if significant else 'No'}")

# Save the figures as both HTML and SVG
fig.write_html("validation/grf_component_comparison.html")
fig.write_image("validation/grf_component_comparison.svg")
fig2.write_html("validation/grf_overall_comparison.html") 
fig2.write_image("validation/grf_overall_comparison.svg")

print(f"\nPlots saved as:")
print(f"  - validation/grf_component_comparison.html (interactive)")
print(f"  - validation/grf_component_comparison.svg (vector graphics)")
print(f"  - validation/grf_overall_comparison.html (interactive)")
print(f"  - validation/grf_overall_comparison.svg (vector graphics)")