"""
Statistical comparison of methods using t-tests.
Compares OpenCap Mono vs WHAM and OpenCap Mono vs OpenCap Two-camera
for rotation (degrees) and translation (cm) metrics.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Significance level
ALPHA = 0.05

# Define DOF categories
TRANSLATIONS = ["pelvis_tx", "pelvis_ty", "pelvis_tz"]


def load_data(csv_dir="analysis_results", aggregate_to_subject=True):
    """
    Load the IK results CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        aggregate_to_subject: If True, average within subjects first (correct for nested data)
                             If False, use all trials (assumes independence)
    
    Returns:
        mono, wham, twocam DataFrames (either subject-level or trial-level)
    """
    mono_path = os.path.join(csv_dir, "ik_results_mono.csv")
    wham_path = os.path.join(csv_dir, "ik_results_wham.csv")
    twocam_path = os.path.join(csv_dir, "ik_results_twocam.csv")
    
    print(f"Loading data from {csv_dir}...")
    mono = pd.read_csv(mono_path)
    wham = pd.read_csv(wham_path)
    twocam = pd.read_csv(twocam_path)
    
    print(f"  Mono: {len(mono)} trials, {mono['Subject'].nunique()} subjects")
    print(f"  WHAM: {len(wham)} trials, {wham['Subject'].nunique()} subjects")
    print(f"  Two-camera: {len(twocam)} trials, {twocam['Subject'].nunique()} subjects")
    
    if aggregate_to_subject:
        print("\nAggregating to subject level (averaging trials within each subject)...")
        # Average within each subject
        mono = mono.groupby('Subject').agg({
            'global_mae_degrees': 'mean',
            'global_mae_mm': 'mean',
            'Subject': 'first'  # Keep subject name
        }).reset_index(drop=True)
        
        wham = wham.groupby('Subject').agg({
            'global_mae_degrees': 'mean',
            'global_mae_mm': 'mean',
            'Subject': 'first'
        }).reset_index(drop=True)
        
        twocam = twocam.groupby('Subject').agg({
            'global_mae_degrees': 'mean',
            'global_mae_mm': 'mean',
            'Subject': 'first'
        }).reset_index(drop=True)
        
        print(f"  After aggregation:")
        print(f"    Mono: {len(mono)} subjects")
        print(f"    WHAM: {len(wham)} subjects")
        print(f"    Two-camera: {len(twocam)} subjects")
    
    return mono, wham, twocam


def get_rotation_errors(df):
    """Extract rotation errors (degrees) from dataframe."""
    # Use global_mae_degrees if available, otherwise compute from rotation DOFs
    if "global_mae_degrees" in df.columns:
        return df["global_mae_degrees"].values
    else:
        # Compute mean across all rotation DOFs
        exclude_cols = ["Subject", "Movement", "Trial", "Camera", "Time", "time",
                       "global_mae_degrees", "global_mae_mm", "global_mae_weighted"]
        rotation_cols = [col for col in df.columns 
                        if col not in exclude_cols 
                        and pd.api.types.is_numeric_dtype(df[col])
                        and not any(x in col for x in ["_tx", "_ty", "_tz", "mm", "min", "max", "std", "mean"])]
        if len(rotation_cols) > 0:
            return df[rotation_cols].mean(axis=1).values
        else:
            return np.array([])


def get_translation_errors(df):
    """Extract translation errors (cm) from dataframe."""
    # Use global_mae_mm if available, convert to cm
    if "global_mae_mm" in df.columns:
        return df["global_mae_mm"].values / 10.0  # Convert mm to cm
    else:
        # Compute mean across translation DOFs and convert to cm
        translation_cols = [col for col in df.columns if any(x in col for x in ["_tx", "_ty", "_tz"])]
        if len(translation_cols) > 0:
            # Assuming translation errors are in meters, convert to cm
            # If they're already in mm, divide by 10
            translation_errors = df[translation_cols].mean(axis=1).values
            # Check if values are reasonable for mm (typically 10-200mm) or m (0.01-0.2m)
            if translation_errors.max() > 1.0:  # Likely in mm
                return translation_errors / 10.0
            else:  # Likely in meters
                return translation_errors * 100.0
        else:
            return np.array([])


def perform_shapiro_test(data, group_name):
    """
    Perform Shapiro-Wilk test for normality.
    
    Returns:
        statistic, p_value, is_normal
    """
    # Remove any NaN values
    data_clean = data[~np.isnan(data)]
    
    if len(data_clean) < 3:  # Shapiro-Wilk requires at least 3 samples
        return np.nan, np.nan, False
    
    # Perform Shapiro-Wilk test
    statistic, p_value = stats.shapiro(data_clean)
    
    # Data is considered normal if p >= alpha (fail to reject null hypothesis of normality)
    is_normal = p_value >= ALPHA
    
    return statistic, p_value, is_normal


def perform_wilcoxon_signed_rank_test(group1, group2, group1_name, group2_name):
    """
    Perform Wilcoxon signed-rank test for paired samples.
    This is the non-parametric alternative to the paired t-test.
    
    Returns:
        w_statistic, p_value, significant, median_diff
    """
    # Remove any NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan, np.nan, False, np.nan
    
    # Ensure same length (should be if same subjects)
    min_len = min(len(group1_clean), len(group2_clean))
    if len(group1_clean) != len(group2_clean):
        print(f"  Warning: Unequal sample sizes ({len(group1_clean)} vs {len(group2_clean)}), using first {min_len} values")
        group1_clean = group1_clean[:min_len]
        group2_clean = group2_clean[:min_len]
    
    # Calculate differences
    differences = group1_clean - group2_clean
    
    # Perform Wilcoxon signed-rank test
    w_stat, p_value = stats.wilcoxon(differences, alternative='two-sided')
    
    significant = p_value < ALPHA
    
    # Calculate median difference (group1 - group2)
    median_diff = np.median(differences)
    
    return w_stat, p_value, significant, median_diff


def compute_cohens_d_paired(group1, group2):
    """
    Compute Cohen's d (effect size) for paired samples.
    
    Cohen's d = mean(differences) / std(differences)
    
    Interpretation:
    - |d| < 0.2: negligible effect
    - 0.2 ≤ |d| < 0.5: small effect
    - 0.5 ≤ |d| < 0.8: medium effect
    - |d| ≥ 0.8: large effect
    
    Returns:
        cohens_d: Effect size
    """
    # Remove any NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan
    
    # Ensure same length
    min_len = min(len(group1_clean), len(group2_clean))
    if len(group1_clean) != len(group2_clean):
        group1_clean = group1_clean[:min_len]
        group2_clean = group2_clean[:min_len]
    
    # Calculate differences
    differences = group1_clean - group2_clean
    
    # Cohen's d for paired samples
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    if std_diff == 0:
        return np.nan
    
    cohens_d = mean_diff / std_diff
    
    return cohens_d


def perform_ttest(group1, group2, group1_name, group2_name, paired=True):
    """
    Perform t-test (paired or independent samples).
    
    Args:
        paired: If True, use paired t-test (for matched subjects). If False, use independent samples t-test.
    
    Returns:
        t_statistic, p_value, significant, group1_clean, group2_clean, mean_diff, ci_lower, ci_upper, cohens_d, df
    """
    # Remove any NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return np.nan, np.nan, False, group1_clean, group2_clean, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Ensure same length for paired test
    if paired:
        min_len = min(len(group1_clean), len(group2_clean))
        if len(group1_clean) != len(group2_clean):
            print(f"  Warning: Unequal sample sizes ({len(group1_clean)} vs {len(group2_clean)}), using first {min_len} values")
            group1_clean = group1_clean[:min_len]
            group2_clean = group2_clean[:min_len]
    
    if paired:
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(group1_clean, group2_clean)
        
        # Calculate differences
        differences = group1_clean - group2_clean
        mean_diff = np.mean(differences)
        
        # Standard error of the mean difference (for paired samples)
        se_diff = np.std(differences, ddof=1) / np.sqrt(len(differences))
        
        # Degrees of freedom for paired t-test: n - 1
        df = len(differences) - 1
        
        # Cohen's d for paired samples
        cohens_d = compute_cohens_d_paired(group1_clean, group2_clean)
    else:
        # Independent samples t-test (Welch's t-test for unequal variances)
        t_stat, p_value = stats.ttest_ind(group1_clean, group2_clean, equal_var=False)
        
        # Calculate mean difference (group1 - group2)
        mean_diff = np.mean(group1_clean) - np.mean(group2_clean)
        
        # Standard error of the difference for independent samples
        se1 = np.std(group1_clean, ddof=1) / np.sqrt(len(group1_clean))
        se2 = np.std(group2_clean, ddof=1) / np.sqrt(len(group2_clean))
        se_diff = np.sqrt(se1**2 + se2**2)
        
        # Degrees of freedom for Welch's t-test
        var1 = np.var(group1_clean, ddof=1)
        var2 = np.var(group2_clean, ddof=1)
        n1 = len(group1_clean)
        n2 = len(group2_clean)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Cohen's d for independent samples
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
    
    significant = p_value < ALPHA
    
    # t-critical value for 95% CI
    t_critical = stats.t.ppf(1 - ALPHA/2, df)
    
    # Confidence interval
    margin_error = t_critical * se_diff
    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error
    
    return t_stat, p_value, significant, group1_clean, group2_clean, mean_diff, ci_lower, ci_upper, cohens_d, df


def plot_residuals(group1, group2, group1_name, group2_name, metric_name, output_dir):
    """
    Plot residuals and distributions to visualize non-normality.
    Creates histograms, Q-Q plots, and box plots.
    """
    # Remove any NaN values
    group1_clean = group1[~np.isnan(group1)]
    group2_clean = group2[~np.isnan(group2)]
    
    if len(group1_clean) < 2 or len(group2_clean) < 2:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Histogram of residuals (deviations from group means)
    ax1 = plt.subplot(2, 3, 1)
    residuals1 = group1_clean - np.mean(group1_clean)
    residuals2 = group2_clean - np.mean(group2_clean)
    ax1.hist(residuals1, bins=20, alpha=0.6, label=group1_name, density=True, color='blue')
    ax1.hist(residuals2, bins=20, alpha=0.6, label=group2_name, density=True, color='red')
    ax1.set_xlabel('Residuals (deviation from mean)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Residual Histograms\n{metric_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot for group1
    ax2 = plt.subplot(2, 3, 2)
    stats.probplot(group1_clean, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot: {group1_name}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot for group2
    ax3 = plt.subplot(2, 3, 3)
    stats.probplot(group2_clean, dist="norm", plot=ax3)
    ax3.set_title(f'Q-Q Plot: {group2_name}')
    ax3.grid(True, alpha=0.3)
    
    # 4. Box plot
    ax4 = plt.subplot(2, 3, 4)
    box_data = [group1_clean, group2_clean]
    bp = ax4.boxplot(box_data, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax4.set_xticklabels([group1_name, group2_name])
    ax4.set_ylabel(metric_name)
    ax4.set_title('Box Plots')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Histogram of group1
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(group1_clean, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax5.axvline(np.mean(group1_clean), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(group1_clean):.3f}')
    ax5.axvline(np.median(group1_clean), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(group1_clean):.3f}')
    ax5.set_xlabel(metric_name)
    ax5.set_ylabel('Frequency')
    ax5.set_title(f'Distribution: {group1_name}')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Histogram of group2
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(group2_clean, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax6.axvline(np.mean(group2_clean), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(group2_clean):.3f}')
    ax6.axvline(np.median(group2_clean), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(group2_clean):.3f}')
    ax6.set_xlabel(metric_name)
    ax6.set_ylabel('Frequency')
    ax6.set_title(f'Distribution: {group2_name}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    safe_group1 = group1_name.replace(' ', '_').replace('-', '_')
    safe_group2 = group2_name.replace(' ', '_').replace('-', '_')
    safe_metric = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
    filename = f"residuals_{safe_group1}_vs_{safe_group2}_{safe_metric}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Residual plots saved to: {os.path.abspath(filepath)}")


def compare_methods(mono, wham, twocam, output_dir="analysis_results"):
    """Compare methods using t-tests for rotation and translation."""
    
    results = []
    
    # Get rotation and translation errors for each method
    mono_rot = get_rotation_errors(mono)
    mono_trans = get_translation_errors(mono)
    
    wham_rot = get_rotation_errors(wham)
    wham_trans = get_translation_errors(wham)
    
    twocam_rot = get_rotation_errors(twocam)
    twocam_trans = get_translation_errors(twocam)
    
    # Determine unit of analysis
    if len(mono) <= 20:  # Likely subject-level
        unit = "subjects"
        n_info = f"n={len(mono)} {unit}"
    else:  # Likely trial-level
        unit = "trials"
        n_info = f"n={len(mono)} {unit} (WARNING: may have pseudoreplication if multiple trials per subject)"
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON RESULTS")
    print(f"Unit of analysis: {n_info}")
    print("="*80)
    
    # Comparison 1: Mono vs WHAM - Rotation
    print("\n1. OpenCap Mono vs WHAM - Rotation (degrees)")
    print("-" * 80)
    
    # Perform Shapiro-Wilk tests for normality
    sw_stat1, sw_p1, is_normal1 = perform_shapiro_test(mono_rot, "Mono")
    sw_stat2, sw_p2, is_normal2 = perform_shapiro_test(wham_rot, "WHAM")
    
    t_stat, p_value, significant, mono_rot_clean, wham_rot_clean, mean_diff, ci_lower, ci_upper, cohens_d, df = perform_ttest(mono_rot, wham_rot, "Mono", "WHAM", paired=True)
    
    # Perform Wilcoxon signed-rank test (non-parametric alternative for paired data)
    w_stat, w_p, w_significant, median_diff = perform_wilcoxon_signed_rank_test(mono_rot, wham_rot, "Mono", "WHAM")
    
    # Plot residuals
    plot_residuals(mono_rot, wham_rot, "OpenCap Mono", "WHAM", "Rotation (degrees)", output_dir)
    
    # Interpret Cohen's d
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"  Mono:   Mean = {np.mean(mono_rot_clean):.3f}°, Median = {np.median(mono_rot_clean):.3f}°, Std = {np.std(mono_rot_clean):.3f}°, N = {len(mono_rot_clean)}")
    print(f"  WHAM:   Mean = {np.mean(wham_rot_clean):.3f}°, Median = {np.median(wham_rot_clean):.3f}°, Std = {np.std(wham_rot_clean):.3f}°, N = {len(wham_rot_clean)}")
    print(f"  Shapiro-Wilk test (normality):")
    print(f"    Mono: W = {sw_stat1:.4f}, p = {sw_p1:.4f}, Normal = {'Yes' if is_normal1 else 'No'}")
    print(f"    WHAM: W = {sw_stat2:.4f}, p = {sw_p2:.4f}, Normal = {'Yes' if is_normal2 else 'No'}")
    print(f"  Parametric test (paired t-test):")
    print(f"    Difference (Mono - WHAM): {mean_diff:.3f}° (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"    t-statistic = {t_stat:.4f}, df = {df:.1f}, p-value = {p_value:.4f}, Significant = {'Yes' if significant else 'No'}")
    print(f"    Cohen's d = {cohens_d:.3f} ({effect_size} effect)")
    print(f"  Non-parametric test (Wilcoxon signed-rank):")
    print(f"    Median difference (Mono - WHAM): {median_diff:.3f}°")
    print(f"    W-statistic = {w_stat:.4f}, p-value = {w_p:.4f}, Significant = {'Yes' if w_significant else 'No'}")
    
    results.append({
        'Comparison': 'Mono vs WHAM',
        'Metric': 'Rotation (degrees)',
        'Method1': 'OpenCap Mono',
        'Method2': 'WHAM',
        'Method1_Mean': np.mean(mono_rot_clean),
        'Method1_Median': np.median(mono_rot_clean),
        'Method1_Std': np.std(mono_rot_clean),
        'Method1_N': len(mono_rot_clean),
        'Method1_Shapiro_W': sw_stat1,
        'Method1_Shapiro_p': sw_p1,
        'Method1_Normal': is_normal1,
        'Method2_Mean': np.mean(wham_rot_clean),
        'Method2_Median': np.median(wham_rot_clean),
        'Method2_Std': np.std(wham_rot_clean),
        'Method2_N': len(wham_rot_clean),
        'Method2_Shapiro_W': sw_stat2,
        'Method2_Shapiro_p': sw_p2,
        'Method2_Normal': is_normal2,
        't_statistic': t_stat,
        't_df': df,
        't_p_value': p_value,
        't_significant': significant,
        'Cohens_d': cohens_d,
        'Effect_size': effect_size,
        'Wilcoxon_W_statistic': w_stat,
        'Wilcoxon_p_value': w_p,
        'Wilcoxon_significant': w_significant,
        'Alpha': ALPHA
    })
    
    # Comparison 2: Mono vs WHAM - Translation
    print("\n2. OpenCap Mono vs WHAM - Translation (cm)")
    print("-" * 80)
    
    # Perform Shapiro-Wilk tests for normality
    sw_stat1, sw_p1, is_normal1 = perform_shapiro_test(mono_trans, "Mono")
    sw_stat2, sw_p2, is_normal2 = perform_shapiro_test(wham_trans, "WHAM")
    
    t_stat, p_value, significant, mono_trans_clean, wham_trans_clean, mean_diff, ci_lower, ci_upper, cohens_d, df = perform_ttest(mono_trans, wham_trans, "Mono", "WHAM", paired=True)
    
    # Perform Wilcoxon signed-rank test (non-parametric alternative for paired data)
    w_stat, w_p, w_significant, median_diff = perform_wilcoxon_signed_rank_test(mono_trans, wham_trans, "Mono", "WHAM")
    
    # Plot residuals
    plot_residuals(mono_trans, wham_trans, "OpenCap Mono", "WHAM", "Translation (cm)", output_dir)
    
    # Interpret Cohen's d
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"  Mono:   Mean = {np.mean(mono_trans_clean):.3f} cm, Median = {np.median(mono_trans_clean):.3f} cm, Std = {np.std(mono_trans_clean):.3f} cm, N = {len(mono_trans_clean)}")
    print(f"  WHAM:   Mean = {np.mean(wham_trans_clean):.3f} cm, Median = {np.median(wham_trans_clean):.3f} cm, Std = {np.std(wham_trans_clean):.3f} cm, N = {len(wham_trans_clean)}")
    print(f"  Shapiro-Wilk test (normality):")
    print(f"    Mono: W = {sw_stat1:.4f}, p = {sw_p1:.4f}, Normal = {'Yes' if is_normal1 else 'No'}")
    print(f"    WHAM: W = {sw_stat2:.4f}, p = {sw_p2:.4f}, Normal = {'Yes' if is_normal2 else 'No'}")
    print(f"  Parametric test (paired t-test):")
    print(f"    Difference (Mono - WHAM): {mean_diff:.3f} cm (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"    t-statistic = {t_stat:.4f}, df = {df:.1f}, p-value = {p_value:.4f}, Significant = {'Yes' if significant else 'No'}")
    print(f"    Cohen's d = {cohens_d:.3f} ({effect_size} effect)")
    print(f"  Non-parametric test (Wilcoxon signed-rank):")
    print(f"    Median difference (Mono - WHAM): {median_diff:.3f} cm")
    print(f"    W-statistic = {w_stat:.4f}, p-value = {w_p:.4f}, Significant = {'Yes' if w_significant else 'No'}")
    
    results.append({
        'Comparison': 'Mono vs WHAM',
        'Metric': 'Translation (cm)',
        'Method1': 'OpenCap Mono',
        'Method2': 'WHAM',
        'Method1_Mean': np.mean(mono_trans_clean),
        'Method1_Median': np.median(mono_trans_clean),
        'Method1_Std': np.std(mono_trans_clean),
        'Method1_N': len(mono_trans_clean),
        'Method1_Shapiro_W': sw_stat1,
        'Method1_Shapiro_p': sw_p1,
        'Method1_Normal': is_normal1,
        'Method2_Mean': np.mean(wham_trans_clean),
        'Method2_Median': np.median(wham_trans_clean),
        'Method2_Std': np.std(wham_trans_clean),
        'Method2_N': len(wham_trans_clean),
        'Method2_Shapiro_W': sw_stat2,
        'Method2_Shapiro_p': sw_p2,
        'Method2_Normal': is_normal2,
        't_statistic': t_stat,
        't_df': df,
        't_p_value': p_value,
        't_significant': significant,
        'Cohens_d': cohens_d,
        'Effect_size': effect_size,
        'Wilcoxon_W_statistic': w_stat,
        'Wilcoxon_p_value': w_p,
        'Wilcoxon_significant': w_significant,
        'Alpha': ALPHA
    })
    
    # Comparison 3: Mono vs Two-camera - Rotation
    print("\n3. OpenCap Mono vs OpenCap Two-camera - Rotation (degrees)")
    print("-" * 80)
    
    # Perform Shapiro-Wilk tests for normality
    sw_stat1, sw_p1, is_normal1 = perform_shapiro_test(mono_rot, "Mono")
    sw_stat2, sw_p2, is_normal2 = perform_shapiro_test(twocam_rot, "Two-camera")
    
    t_stat, p_value, significant, mono_rot_clean2, twocam_rot_clean, mean_diff, ci_lower, ci_upper, cohens_d, df = perform_ttest(mono_rot, twocam_rot, "Mono", "Two-camera", paired=True)
    
    # Perform Wilcoxon signed-rank test (non-parametric alternative for paired data)
    w_stat, w_p, w_significant, median_diff = perform_wilcoxon_signed_rank_test(mono_rot, twocam_rot, "Mono", "Two-camera")
    
    # Plot residuals
    plot_residuals(mono_rot, twocam_rot, "OpenCap Mono", "OpenCap Two-camera", "Rotation (degrees)", output_dir)
    
    # Interpret Cohen's d
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"  Mono:      Mean = {np.mean(mono_rot_clean2):.3f}°, Median = {np.median(mono_rot_clean2):.3f}°, Std = {np.std(mono_rot_clean2):.3f}°, N = {len(mono_rot_clean2)}")
    print(f"  Two-camera: Mean = {np.mean(twocam_rot_clean):.3f}°, Median = {np.median(twocam_rot_clean):.3f}°, Std = {np.std(twocam_rot_clean):.3f}°, N = {len(twocam_rot_clean)}")
    print(f"  Shapiro-Wilk test (normality):")
    print(f"    Mono:      W = {sw_stat1:.4f}, p = {sw_p1:.4f}, Normal = {'Yes' if is_normal1 else 'No'}")
    print(f"    Two-camera: W = {sw_stat2:.4f}, p = {sw_p2:.4f}, Normal = {'Yes' if is_normal2 else 'No'}")
    print(f"  Parametric test (paired t-test):")
    print(f"    Difference (Mono - Two-camera): {mean_diff:.3f}° (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"    t-statistic = {t_stat:.4f}, df = {df:.1f}, p-value = {p_value:.4f}, Significant = {'Yes' if significant else 'No'}")
    print(f"    Cohen's d = {cohens_d:.3f} ({effect_size} effect)")
    print(f"  Non-parametric test (Wilcoxon signed-rank):")
    print(f"    Median difference (Mono - Two-camera): {median_diff:.3f}°")
    print(f"    W-statistic = {w_stat:.4f}, p-value = {w_p:.4f}, Significant = {'Yes' if w_significant else 'No'}")
    
    results.append({
        'Comparison': 'Mono vs Two-camera',
        'Metric': 'Rotation (degrees)',
        'Method1': 'OpenCap Mono',
        'Method2': 'OpenCap Two-camera',
        'Method1_Mean': np.mean(mono_rot_clean2),
        'Method1_Median': np.median(mono_rot_clean2),
        'Method1_Std': np.std(mono_rot_clean2),
        'Method1_N': len(mono_rot_clean2),
        'Method1_Shapiro_W': sw_stat1,
        'Method1_Shapiro_p': sw_p1,
        'Method1_Normal': is_normal1,
        'Method2_Mean': np.mean(twocam_rot_clean),
        'Method2_Median': np.median(twocam_rot_clean),
        'Method2_Std': np.std(twocam_rot_clean),
        'Method2_N': len(twocam_rot_clean),
        'Method2_Shapiro_W': sw_stat2,
        'Method2_Shapiro_p': sw_p2,
        'Method2_Normal': is_normal2,
        't_statistic': t_stat,
        't_df': df,
        't_p_value': p_value,
        't_significant': significant,
        'Cohens_d': cohens_d,
        'Effect_size': effect_size,
        'Wilcoxon_W_statistic': w_stat,
        'Wilcoxon_p_value': w_p,
        'Wilcoxon_significant': w_significant,
        'Alpha': ALPHA
    })
    
    # Comparison 4: Mono vs Two-camera - Translation
    print("\n4. OpenCap Mono vs OpenCap Two-camera - Translation (cm)")
    print("-" * 80)
    
    # Perform Shapiro-Wilk tests for normality
    sw_stat1, sw_p1, is_normal1 = perform_shapiro_test(mono_trans, "Mono")
    sw_stat2, sw_p2, is_normal2 = perform_shapiro_test(twocam_trans, "Two-camera")
    
    t_stat, p_value, significant, mono_trans_clean2, twocam_trans_clean, mean_diff, ci_lower, ci_upper, cohens_d, df = perform_ttest(mono_trans, twocam_trans, "Mono", "Two-camera", paired=True)
    
    # Perform Wilcoxon signed-rank test (non-parametric alternative for paired data)
    w_stat, w_p, w_significant, median_diff = perform_wilcoxon_signed_rank_test(mono_trans, twocam_trans, "Mono", "Two-camera")
    
    # Plot residuals
    plot_residuals(mono_trans, twocam_trans, "OpenCap Mono", "OpenCap Two-camera", "Translation (cm)", output_dir)
    
    # Interpret Cohen's d
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"
    
    print(f"  Mono:      Mean = {np.mean(mono_trans_clean2):.3f} cm, Median = {np.median(mono_trans_clean2):.3f} cm, Std = {np.std(mono_trans_clean2):.3f} cm, N = {len(mono_trans_clean2)}")
    print(f"  Two-camera: Mean = {np.mean(twocam_trans_clean):.3f} cm, Median = {np.median(twocam_trans_clean):.3f} cm, Std = {np.std(twocam_trans_clean):.3f} cm, N = {len(twocam_trans_clean)}")
    print(f"  Shapiro-Wilk test (normality):")
    print(f"    Mono:      W = {sw_stat1:.4f}, p = {sw_p1:.4f}, Normal = {'Yes' if is_normal1 else 'No'}")
    print(f"    Two-camera: W = {sw_stat2:.4f}, p = {sw_p2:.4f}, Normal = {'Yes' if is_normal2 else 'No'}")
    print(f"  Parametric test (paired t-test):")
    print(f"    Difference (Mono - Two-camera): {mean_diff:.3f} cm (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"    t-statistic = {t_stat:.4f}, df = {df:.1f}, p-value = {p_value:.4f}, Significant = {'Yes' if significant else 'No'}")
    print(f"    Cohen's d = {cohens_d:.3f} ({effect_size} effect)")
    print(f"  Non-parametric test (Wilcoxon signed-rank):")
    print(f"    Median difference (Mono - Two-camera): {median_diff:.3f} cm")
    print(f"    W-statistic = {w_stat:.4f}, p-value = {w_p:.4f}, Significant = {'Yes' if w_significant else 'No'}")
    
    results.append({
        'Comparison': 'Mono vs Two-camera',
        'Metric': 'Translation (cm)',
        'Method1': 'OpenCap Mono',
        'Method2': 'OpenCap Two-camera',
        'Method1_Mean': np.mean(mono_trans_clean2),
        'Method1_Median': np.median(mono_trans_clean2),
        'Method1_Std': np.std(mono_trans_clean2),
        'Method1_N': len(mono_trans_clean2),
        'Method1_Shapiro_W': sw_stat1,
        'Method1_Shapiro_p': sw_p1,
        'Method1_Normal': is_normal1,
        'Method2_Mean': np.mean(twocam_trans_clean),
        'Method2_Median': np.median(twocam_trans_clean),
        'Method2_Std': np.std(twocam_trans_clean),
        'Method2_N': len(twocam_trans_clean),
        'Method2_Shapiro_W': sw_stat2,
        'Method2_Shapiro_p': sw_p2,
        'Method2_Normal': is_normal2,
        't_statistic': t_stat,
        't_df': df,
        't_p_value': p_value,
        't_significant': significant,
        'Cohens_d': cohens_d,
        'Effect_size': effect_size,
        'Wilcoxon_W_statistic': w_stat,
        'Wilcoxon_p_value': w_p,
        'Wilcoxon_significant': w_significant,
        'Alpha': ALPHA
    })
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    
    output_path = os.path.join(output_dir, "statistical_comparison_results.csv")
    results_df.to_csv(output_path, index=False)
    print("\n" + "="*80)
    print(f"Results saved to: {os.path.abspath(output_path)}")
    print("="*80)
    
    return results_df


def main():
    """Main function to run statistical comparisons."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistical comparison of methods using t-tests')
    parser.add_argument('--csv_dir', type=str, default='analysis_results',
                       help='Directory containing the IK results CSV files (default: analysis_results)')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Directory to save output CSV (default: analysis_results)')
    parser.add_argument('--use_trials', action='store_true',
                       help='Use all trials instead of aggregating to subject level (not recommended)')
    
    args = parser.parse_args()
    
    # Load data (aggregate to subject level by default to avoid pseudoreplication)
    aggregate_to_subject = not args.use_trials
    mono, wham, twocam = load_data(args.csv_dir, aggregate_to_subject=aggregate_to_subject)
    
    # Perform comparisons
    results_df = compare_methods(mono, wham, twocam, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nParametric test (paired t-test) - Significant differences (p < 0.05):")
    t_significant = results_df[results_df['t_significant'] == True]
    if len(t_significant) > 0:
        for _, row in t_significant.iterrows():
            print(f"  ✓ {row['Comparison']} - {row['Metric']} (p = {row['t_p_value']:.4f}, d = {row['Cohens_d']:.3f})")
    else:
        print("  No significant differences found.")
    
    print("\nNon-parametric test (Wilcoxon signed-rank) - Significant differences (p < 0.05):")
    w_significant = results_df[results_df['Wilcoxon_significant'] == True]
    if len(w_significant) > 0:
        for _, row in w_significant.iterrows():
            print(f"  ✓ {row['Comparison']} - {row['Metric']} (p = {row['Wilcoxon_p_value']:.4f})")
    else:
        print("  No significant differences found.")
    
    print("\nParametric test (paired t-test) - Non-significant differences (p ≥ 0.05):")
    t_non_significant = results_df[results_df['t_significant'] == False]
    if len(t_non_significant) > 0:
        for _, row in t_non_significant.iterrows():
            print(f"  ✗ {row['Comparison']} - {row['Metric']} (p = {row['t_p_value']:.4f}, d = {row['Cohens_d']:.3f})")
    else:
        print("  All comparisons were significant.")
    
    print("\nNon-parametric test (Wilcoxon signed-rank) - Non-significant differences (p ≥ 0.05):")
    w_non_significant = results_df[results_df['Wilcoxon_significant'] == False]
    if len(w_non_significant) > 0:
        for _, row in w_non_significant.iterrows():
            print(f"  ✗ {row['Comparison']} - {row['Metric']} (p = {row['Wilcoxon_p_value']:.4f})")
    else:
        print("  All comparisons were significant.")


if __name__ == "__main__":
    main()

