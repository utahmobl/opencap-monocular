"""
Statistical Analysis: Paired t-tests for Normal vs Weak Conditions
Compares knee, hip, and ankle moments between normal and weak conditions
for both mocap and opencap_mono methods.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

# Configuration
alpha = 0.05
output_dir = "sts_kinetics_validation"
combined_csv = os.path.join(output_dir, "combined_kinetics_data_for_statistics.csv")

# Moment types to analyze
moment_types = {
    "knee": {
        "columns": ["knee_ext_l", "knee_ext_r", "knee_ext_combined"],
        "labels": ["Left Knee", "Right Knee", "Combined Knee"],
    },
    "hip": {
        "columns": ["hip_ext_l", "hip_ext_r", "hip_ext_combined"],
        "labels": ["Left Hip", "Right Hip", "Combined Hip"],
    },
    "ankle": {
        "columns": ["ankle_pf_l", "ankle_pf_r", "ankle_pf_combined"],
        "labels": ["Left Ankle", "Right Ankle", "Combined Ankle"],
    },
}

# Methods to analyze
methods = ["mocap", "opencap_mono"]


def load_data(csv_path):
    """Load the combined kinetics data."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Combined CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    print(f"Methods: {df['method'].unique()}")
    print(f"Conditions: {df['condition'].unique()}")
    return df


def perform_paired_ttest(normal_data, weak_data, moment_name, method_name):
    """
    Perform paired t-test between normal and weak conditions.
    
    Parameters:
    -----------
    normal_data : array-like
        Data for normal condition
    weak_data : array-like
        Data for weak condition (must be paired with normal_data)
    moment_name : str
        Name of the moment being tested
    method_name : str
        Method name (mocap or opencap_mono)
    
    Returns:
    --------
    dict : Dictionary with test results including residuals
    """
    # Remove any NaN values and ensure paired data
    pairs = list(zip(normal_data, weak_data))
    pairs = [(n, w) for n, w in pairs if not (np.isnan(n) or np.isnan(w))]
    
    if len(pairs) < 2:
        return {
            "n": len(pairs),
            "normal_mean": np.nan,
            "weak_mean": np.nan,
            "difference_mean": np.nan,
            "difference_std": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
            "significant": False,
            "error": "Insufficient data",
            "differences": None,
            "shapiro_statistic": np.nan,
            "shapiro_pvalue": np.nan,
        }
    
    normal_vals = [n for n, w in pairs]
    weak_vals = [w for n, w in pairs]
    differences = np.array([n - w for n, w in pairs])
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(normal_vals, weak_vals)
    
    # Calculate statistics
    normal_mean = np.mean(normal_vals)
    weak_mean = np.mean(weak_vals)
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)  # Sample standard deviation
    
    # Perform Shapiro-Wilk test for normality (on differences/residuals)
    if len(differences) >= 3 and len(differences) <= 5000:  # Shapiro-Wilk works for 3-5000 samples
        shapiro_stat, shapiro_p = stats.shapiro(differences)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Determine significance
    significant = p_value < alpha
    
    return {
        "n": len(pairs),
        "normal_mean": normal_mean,
        "weak_mean": weak_mean,
        "difference_mean": diff_mean,
        "difference_std": diff_std,
        "difference_se": diff_std / np.sqrt(len(pairs)),
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": significant,
        "error": None,
        "differences": differences,
        "shapiro_statistic": shapiro_stat,
        "shapiro_pvalue": shapiro_p,
    }


def analyze_method(df, method_name):
    """
    Analyze a specific method (mocap or opencap_mono).
    
    Parameters:
    -----------
    df : DataFrame
        Full dataset
    method_name : str
        Method to analyze
    
    Returns:
    --------
    dict : Results for all moment types
    """
    # Filter data for this method
    method_df = df[df["method"] == method_name].copy()
    
    if len(method_df) == 0:
        print(f"\nWarning: No data found for method: {method_name}")
        return None
    
    print(f"\n{'='*70}")
    print(f"ANALYZING METHOD: {method_name.upper()}")
    print(f"{'='*70}")
    
    results = {}
    
    # Analyze each moment type
    for moment_type, moment_info in moment_types.items():
        print(f"\n{'-'*70}")
        print(f"MOMENT TYPE: {moment_type.upper()}")
        print(f"{'-'*70}")
        
        moment_results = {}
        
        for col, label in zip(moment_info["columns"], moment_info["labels"]):
            # Get data for normal and weak conditions
            # Match by subject and simulation to ensure pairing
            normal_data = []
            weak_data = []
            
            # Group by subject and simulation to ensure proper pairing
            for (subject, sim), group in method_df.groupby(["subject", "simulation"]):
                normal_group = group[group["condition"] == "normal"]
                weak_group = group[group["condition"] == "weak"]
                
                if len(normal_group) > 0 and len(weak_group) > 0:
                    normal_val = normal_group[col].values[0]
                    weak_val = weak_group[col].values[0]
                    
                    if not (np.isnan(normal_val) or np.isnan(weak_val)):
                        normal_data.append(normal_val)
                        weak_data.append(weak_val)
            
            # Perform paired t-test
            test_result = perform_paired_ttest(normal_data, weak_data, label, method_name)
            moment_results[label] = test_result
            
            # Print results
            print(f"\n{label}:")
            print(f"  N pairs: {test_result['n']}")
            print(f"  Normal mean: {test_result['normal_mean']:.2f} Nm")
            print(f"  Weak mean: {test_result['weak_mean']:.2f} Nm")
            print(f"  Difference (Normal - Weak): {test_result['difference_mean']:.2f} ± {test_result['difference_se']:.2f} Nm")
            print(f"  t-statistic: {test_result['t_statistic']:.3f}")
            print(f"  p-value: {test_result['p_value']:.4f}")
            
            # Print normality test results
            if not np.isnan(test_result['shapiro_pvalue']):
                print(f"  Shapiro-Wilk test: W={test_result['shapiro_statistic']:.4f}, p={test_result['shapiro_pvalue']:.4f}")
                if test_result['shapiro_pvalue'] < alpha:
                    print(f"  Normality: NOT NORMAL (p < {alpha})")
                else:
                    print(f"  Normality: Normal (p >= {alpha})")
            
            if test_result['error']:
                print(f"  Error: {test_result['error']}")
            else:
                if test_result['significant']:
                    print(f"  Result: SIGNIFICANT (p < {alpha})")
                else:
                    print(f"  Result: Not significant (p >= {alpha})")
        
        results[moment_type] = moment_results
    
    return results


def create_summary_table(all_results):
    """Create a summary table of all results."""
    summary_data = []
    
    for method_name in methods:
        if method_name not in all_results or all_results[method_name] is None:
            continue
        
        method_results = all_results[method_name]
        
        for moment_type, moment_results in method_results.items():
            for label, result in moment_results.items():
                summary_data.append({
                    "Method": method_name,
                    "Moment Type": moment_type.capitalize(),
                    "Joint": label,
                    "N": result["n"],
                    "Normal Mean (Nm)": result["normal_mean"],
                    "Weak Mean (Nm)": result["weak_mean"],
                    "Difference (Nm)": result["difference_mean"],
                    "SE (Nm)": result["difference_se"],
                    "t-statistic": result["t_statistic"],
                    "p-value": result["p_value"],
                    "Significant": "Yes" if result["significant"] else "No",
                    "Shapiro-Wilk W": result["shapiro_statistic"],
                    "Shapiro-Wilk p": result["shapiro_pvalue"],
                    "Normality": "Normal" if not np.isnan(result["shapiro_pvalue"]) and result["shapiro_pvalue"] >= alpha else "Not Normal",
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def create_residual_plots(df, all_results, output_dir):
    """Create residual plots (Q-Q plots and histograms) to check normality."""
    
    moment_types_list = ["knee", "hip", "ankle"]
    
    for method_name in methods:
        if method_name not in all_results or all_results[method_name] is None:
            continue
        
        method_results = all_results[method_name]
        
        # Count how many plots we need
        plot_count = 0
        plot_info = []
        
        for moment_type in moment_types_list:
            if moment_type not in method_results:
                continue
            moment_results = method_results[moment_type]
            
            for label, result in moment_results.items():
                if result["n"] > 0 and result["error"] is None and result["differences"] is not None:
                    plot_count += 1
                    plot_info.append({
                        "moment_type": moment_type,
                        "label": label,
                        "differences": result["differences"],
                        "shapiro_p": result["shapiro_pvalue"]
                    })
        
        if plot_count == 0:
            continue
        
        # Create figure with subplots (2 columns: Q-Q plot and histogram)
        n_rows = plot_count
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Residual Plots for Normality Check: {method_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        for idx, info in enumerate(plot_info):
            differences = info["differences"]
            moment_type = info["moment_type"]
            label = info["label"]
            shapiro_p = info["shapiro_p"]
            
            # Q-Q plot
            ax_qq = axes[idx, 0]
            stats.probplot(differences, dist="norm", plot=ax_qq)
            ax_qq.set_title(f'Q-Q Plot: {moment_type.capitalize()} - {label}', fontweight='bold', fontsize=10)
            ax_qq.grid(True, alpha=0.3)
            
            # Add Shapiro-Wilk test result
            if not np.isnan(shapiro_p):
                norm_status = "Normal" if shapiro_p >= alpha else "Not Normal"
                ax_qq.text(0.05, 0.95, f'Shapiro-Wilk: p={shapiro_p:.3f}\n({norm_status})',
                          transform=ax_qq.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)
            
            # Histogram with normal curve overlay
            ax_hist = axes[idx, 1]
            n, bins, patches = ax_hist.hist(differences, bins=min(10, len(differences)), 
                                           density=True, alpha=0.7, edgecolor='black')
            
            # Overlay normal distribution
            mu = np.mean(differences)
            sigma = np.std(differences, ddof=1)
            x = np.linspace(differences.min(), differences.max(), 100)
            normal_curve = stats.norm.pdf(x, mu, sigma)
            ax_hist.plot(x, normal_curve, 'r-', linewidth=2, label='Normal distribution')
            
            ax_hist.set_xlabel('Difference (Normal - Weak) [Nm]', fontsize=10)
            ax_hist.set_ylabel('Density', fontsize=10)
            ax_hist.set_title(f'Histogram: {moment_type.capitalize()} - {label}', fontweight='bold', fontsize=10)
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"residual_plots_{method_name}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Residual plots for {method_name} saved to: {os.path.abspath(plot_path)}")


def create_visualizations(df, all_results, output_dir):
    """Create visualization plots for the results."""
    
    # Create figure with subplots for each moment type
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Paired t-test Results: Normal vs Weak Conditions', fontsize=16, fontweight='bold')
    
    moment_types_list = ["knee", "hip", "ankle"]
    method_positions = {"mocap": 0, "opencap_mono": 1}
    
    for moment_idx, moment_type in enumerate(moment_types_list):
        for method_name in methods:
            if method_name not in all_results or all_results[method_name] is None:
                continue
            
            ax = axes[moment_idx, method_positions[method_name]]
            method_results = all_results[method_name]
            
            if moment_type not in method_results:
                continue
            
            moment_results = method_results[moment_type]
            
            # Prepare data for plotting
            labels = []
            differences = []
            errors = []
            p_values = []
            significant = []
            
            for label, result in moment_results.items():
                if result["n"] > 0 and result["error"] is None:
                    labels.append(label.replace("Combined ", "").replace("Left ", "L ").replace("Right ", "R "))
                    differences.append(result["difference_mean"])
                    errors.append(result["difference_se"])
                    p_values.append(result["p_value"])
                    significant.append(result["significant"])
            
            if len(differences) > 0:
                # Create bar plot
                colors = ['green' if sig else 'gray' for sig in significant]
                bars = ax.barh(labels, differences, xerr=errors, color=colors, alpha=0.7, capsize=5)
                
                # Add p-value annotations
                for i, (diff, p_val, sig) in enumerate(zip(differences, p_values, significant)):
                    if sig:
                        ax.text(diff + errors[i] + 0.5, i, f'p={p_val:.3f}*', 
                               va='center', fontweight='bold', fontsize=9)
                    else:
                        ax.text(diff + errors[i] + 0.5, i, f'p={p_val:.3f}', 
                               va='center', fontsize=9)
                
                # Add zero line
                ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
                
                ax.set_xlabel('Difference (Normal - Weak) [Nm]', fontsize=10)
                ax.set_title(f'{moment_type.capitalize()} - {method_name.upper()}', fontweight='bold', fontsize=11)
                ax.grid(True, alpha=0.3, axis='x')
                ax.set_xlim(min(differences) - max(errors) - 5, max(differences) + max(errors) + 10)
            else:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{moment_type.capitalize()} - {method_name.upper()}', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "statistical_analysis_results.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {os.path.abspath(plot_path)}")


def main():
    """Main analysis function."""
    print("="*70)
    print("STATISTICAL ANALYSIS: PAIRED T-TESTS")
    print("Normal vs Weak Conditions")
    print("="*70)
    
    # Load data
    try:
        df = load_data(combined_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run kinetics_val_sts.py first to generate the combined CSV.")
        return
    
    # Analyze each method
    all_results = {}
    for method_name in methods:
        results = analyze_method(df, method_name)
        all_results[method_name] = results
    
    # Create summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    summary_df = create_summary_table(all_results)
    
    # Save summary table
    summary_csv = os.path.join(output_dir, "statistical_analysis_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSummary table saved to: {os.path.abspath(summary_csv)}")
    
    # Print summary table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("\n" + summary_df.to_string(index=False))
    
    # Create visualizations
    create_visualizations(df, all_results, output_dir)
    
    # Create residual plots for normality checks
    print(f"\n{'='*70}")
    print("CREATING RESIDUAL PLOTS FOR NORMALITY CHECK")
    print(f"{'='*70}")
    create_residual_plots(df, all_results, output_dir)
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for method_name in methods:
        if method_name not in all_results or all_results[method_name] is None:
            continue
        
        print(f"\n{method_name.upper()}:")
        method_results = all_results[method_name]
        
        total_tests = 0
        significant_tests = 0
        
        for moment_type, moment_results in method_results.items():
            for label, result in moment_results.items():
                if result["n"] > 0 and result["error"] is None:
                    total_tests += 1
                    if result["significant"]:
                        significant_tests += 1
        
        print(f"  Total tests: {total_tests}")
        print(f"  Significant (p < {alpha}): {significant_tests}")
        print(f"  Not significant: {total_tests - significant_tests}")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

