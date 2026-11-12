#!/usr/bin/env python3
"""
Script to analyze hyperparameter search results and find the best weights for each activity.
"""

import json
import os
import sys
from pathlib import Path

def load_hp_results(filepath):
    """
    Load hyperparameter search results from a JSON file.
    
    Parameters:
        filepath (str): Path to the results file
        
    Returns:
        dict: Loaded data or None if failed
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def find_best_weights_by_activity(data):
    """
    Find the best weights for each activity based on the lowest score.
    
    Parameters:
        data (dict): Loaded hyperparameter search data
        
    Returns:
        dict: Best weights for each activity
    """
    if not data or 'combinations' not in data:
        print("Invalid data format: 'combinations' key not found")
        return {}
    
    best_results = {}
    
    for movement, enhanced_combos in data['combinations'].items():
        print(f"\nAnalyzing {movement}...")
        
        # Filter combinations that have been run and have valid scores
        valid_results = []
        for combo in enhanced_combos:
            if combo.get('ran') and combo.get('score') is not None:
                valid_results.append({
                    'weights': combo['combination']['weights'],
                    'score': combo['score'],
                    'status': combo.get('status', 'unknown')
                })
        
        if not valid_results:
            print(f"  No valid results found for {movement}")
            continue
        
        # Sort by score (lowest is best)
        valid_results.sort(key=lambda x: x['score'])
        
        # Get the best result
        best_result = valid_results[0]
        best_results[movement] = best_result
        
        print(f"  Best score: {best_result['score']:.4f}")
        print(f"  Status: {best_result['status']}")
        print(f"  Weights: {best_result['weights']}")
        
        # Show top 3 results
        print(f"  Top 3 results:")
        for i, result in enumerate(valid_results[:3]):
            print(f"    {i+1}. Score: {result['score']:.4f} | Weights: {result['weights']}")
    
    return best_results

def save_best_weights(best_results, output_file):
    """
    Save the best weights to a JSON file.
    
    Parameters:
        best_results (dict): Best weights for each activity
        output_file (str): Output file path
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(best_results, f, indent=2)
        print(f"\nBest weights saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def generate_parameters_yaml_update(best_results, original_params_file):
    """
    Generate an updated parameters.yaml content with the best weights.
    
    Parameters:
        best_results (dict): Best weights for each activity
        original_params_file (str): Path to original parameters file
        
    Returns:
        str: Updated YAML content
    """
    try:
        with open(original_params_file, 'r') as f:
            original_content = f.read()
        
        # Create updated content
        updated_content = original_content
        
        # Update weights for each movement
        for movement, result in best_results.items():
            weights = result['weights']
            
            # Map movement names to parameter keys
            if movement == 'walking':
                param_key = 'weights_opt2_walking'
            elif movement == 'squats':
                param_key = 'weights_opt2_squats'
            elif movement == 'STS':
                param_key = 'weights_opt2_sts'
            else:
                continue
            
            # Find the section to update
            start_marker = f"{param_key}:"
            lines = updated_content.split('\n')
            
            for i, line in enumerate(lines):
                if line.strip() == start_marker:
                    # Update the weights section
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    # Replace the weights section
                    new_weights = [f"{indent_str}{param_key}:"]
                    for weight_name, weight_value in weights.items():
                        new_weights.append(f"{indent_str}  {weight_name}: {weight_value}")
                    
                    # Find where this section ends (next non-indented line)
                    end_idx = i + 1
                    while (end_idx < len(lines) and 
                           lines[end_idx].strip() and 
                           (lines[end_idx].startswith(' ' * (indent + 2)) or 
                            lines[end_idx].startswith(' ' * (indent + 1)))):
                        end_idx += 1
                    
                    # Replace the section
                    lines[i:end_idx] = new_weights
                    break
            
            updated_content = '\n'.join(lines)
        
        return updated_content
        
    except Exception as e:
        print(f"Error generating YAML update: {e}")
        return None

def main():
    """Main function to analyze HP results."""
    
    # File paths
    repo_path = Path(__file__).parent.parent  # Go up one level to get to the main repo
    hp_results_file = repo_path / "output" / "hp_combinations" / "hp_combinations_with_status_20250827_113632_ALL_ALL.json"
    original_params_file = repo_path / "params" / "parameters.yaml"
    
    print("🔍 Hyperparameter Search Results Analyzer")
    print("=" * 50)
    
    # Check if files exist
    if not hp_results_file.exists():
        print(f"❌ HP results file not found: {hp_results_file}")
        print("Please check the file path and try again.")
        return
    
    if not original_params_file.exists():
        print(f"⚠️  Original parameters file not found: {original_params_file}")
        print("Will not generate YAML update.")
    
    # Load the HP results
    print(f"📁 Loading results from: {hp_results_file}")
    data = load_hp_results(hp_results_file)
    
    if not data:
        print("❌ Failed to load HP results")
        return
    
    # Show metadata
    if 'metadata' in data:
        metadata = data['metadata']
        print(f"\n📊 Search Metadata:")
        print(f"  Movements: {', '.join(metadata.get('movements', []))}")
        print(f"  Subject: {metadata.get('subject', 'Unknown')}")
        print(f"  Session: {metadata.get('session', 'Unknown')}")
        print(f"  Total combinations: {metadata.get('total_combinations', 'Unknown')}")
        print(f"  Max combinations per movement: {metadata.get('max_combinations', 'Unknown')}")
        print(f"  Random seed: {metadata.get('random_seed', 'Unknown')}")
    
    # Find best weights
    print(f"\n🏆 Finding best weights for each activity...")
    best_results = find_best_weights_by_activity(data)
    
    if not best_results:
        print("❌ No valid results found")
        return
    
    # Save best weights
    output_file = repo_path / "output" / "best_hp_weights.json"
    save_best_weights(best_results, output_file)
    
    # Generate parameters.yaml update if original file exists
    if original_params_file.exists():
        print(f"\n📝 Generating parameters.yaml update...")
        updated_yaml = generate_parameters_yaml_update(best_results, original_params_file)
        
        if updated_yaml:
            # Save updated YAML
            updated_yaml_file = repo_path / "output" / "parameters_updated_with_best_weights.yaml"
            try:
                with open(updated_yaml_file, 'w') as f:
                    f.write(updated_yaml)
                print(f"✅ Updated parameters saved to: {updated_yaml_file}")
                print("💡 You can copy this content to replace your parameters.yaml file")
            except Exception as e:
                print(f"❌ Error saving updated YAML: {e}")
        else:
            print("❌ Failed to generate YAML update")
    
    # Summary
    print(f"\n🎯 Summary:")
    for movement, result in best_results.items():
        print(f"  {movement.capitalize()}: Best score = {result['score']:.4f}")
    
    print(f"\n✨ Analysis complete! Check the output files for detailed results.")

if __name__ == "__main__":
    main()
