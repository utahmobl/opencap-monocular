#!/usr/bin/env python3
"""
Script to find and explore the folders containing results from the best hyperparameter weights.
"""

import json
import os
from pathlib import Path

def load_best_weights():
    """Load the best weights from the analysis results."""
    best_weights_file = Path(__file__).parent.parent / "output" / "best_hp_weights.json"
    
    if not best_weights_file.exists():
        print("❌ Best weights file not found. Run analyze_hp_results.py first.")
        return None
    
    with open(best_weights_file, 'r') as f:
        return json.load(f)

def find_matching_cases(best_weights):
    """Find the case folders that match the best weights."""
    repo_path = Path(__file__).parent.parent
    cases_files = {
        'walking': repo_path / "output" / "cases_walking.json",
        'squats': repo_path / "output" / "cases_squats.json", 
        'STS': repo_path / "output" / "cases_STS.json"
    }
    
    matching_cases = {}
    
    for movement, cases_file in cases_files.items():
        if not cases_file.exists():
            print(f"⚠️  Cases file not found for {movement}: {cases_file}")
            continue
            
        with open(cases_file, 'r') as f:
            cases = json.load(f)
        
        # Find the case that matches the best weights
        best_weights_for_movement = best_weights[movement]['weights']
        
        for case_name, case_data in cases.items():
            if case_data['weights'] == best_weights_for_movement:
                matching_cases[movement] = {
                    'case_name': case_name,
                    'case_path': repo_path / "output" / case_name,
                    'weights': case_data['weights'],
                    'filter_freq': case_data['filter_freq']
                }
                break
    
    return matching_cases

def explore_case_folder(case_path, movement):
    """Explore the contents of a case folder."""
    print(f"\n📁 Exploring {movement} case folder: {case_path}")
    
    if not case_path.exists():
        print(f"❌ Case folder does not exist: {case_path}")
        return
    
    # List main contents
    print(f"📋 Main contents:")
    for item in case_path.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")
    
    # Check for results.csv
    results_csv = case_path / "results.csv"
    if results_csv.exists():
        print(f"\n📊 Results CSV found: {results_csv}")
        # You could load and display results here if needed
    
    # Check for OpenSim results
    opensim_path = case_path / "OpenSim"
    if opensim_path.exists():
        print(f"\n🔬 OpenSim results found:")
        for item in opensim_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}/")
                # Check for specific result files
                if item.name == "IK":
                    ik_path = item / "shiftedIK"
                    if ik_path.exists():
                        print(f"    📁 shiftedIK/")
                        for ik_file in ik_path.iterdir():
                            if ik_file.name.endswith('.txt'):
                                print(f"      📄 {ik_file.name}")
    
    # Check for visualization files
    viz_files = list(case_path.rglob("*.json"))
    if viz_files:
        print(f"\n🎨 Visualization files found:")
        for viz_file in viz_files:
            print(f"  📄 {viz_file.relative_to(case_path)}")

def main():
    """Main function to find and explore best result folders."""
    print("🔍 Finding Folders with Best Hyperparameter Results")
    print("=" * 60)
    
    # Load best weights
    best_weights = load_best_weights()
    if not best_weights:
        return
    
    print("🏆 Best weights found:")
    for movement, data in best_weights.items():
        print(f"  {movement.capitalize()}: Score = {data['score']:.4f}")
        print(f"    Weights: {data['weights']}")
    
    # Find matching cases
    print(f"\n🔍 Finding matching case folders...")
    matching_cases = find_matching_cases(best_weights)
    
    if not matching_cases:
        print("❌ No matching cases found")
        return
    
    # Display matching cases
    print(f"\n✅ Matching cases found:")
    for movement, case_info in matching_cases.items():
        print(f"  {movement.capitalize()}: {case_info['case_name']}")
        print(f"    Path: {case_info['case_path']}")
        print(f"    Filter frequency: {case_info['filter_freq']}")
    
    # Explore each case folder
    for movement, case_info in matching_cases.items():
        explore_case_folder(case_info['case_path'], movement)
    
    # Summary
    print(f"\n🎯 Summary of Best Result Folders:")
    print("=" * 60)
    for movement, case_info in matching_cases.items():
        print(f"📁 {movement.capitalize()}: {case_info['case_name']}")
        print(f"   Path: {case_info['case_path']}")
        print(f"   Best Score: {best_weights[movement]['score']:.4f}")
        print()
    
    print("💡 You can now explore these folders to examine the detailed results!")
    print("   Each folder contains the complete pipeline output for that weight combination.")

if __name__ == "__main__":
    main()

