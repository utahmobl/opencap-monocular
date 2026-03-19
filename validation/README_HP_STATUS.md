# Hyperparameter Status Tracking

This document describes the enhanced hyperparameter search functionality that tracks the run status and scores of combinations, allowing you to resume interrupted searches and avoid re-running completed combinations.

## Overview

The new system saves hyperparameter combinations with their run status (`ran: true/false`) and scores, enabling:

- **Resume interrupted searches**: Load existing combinations and continue from where you left off
- **Skip completed combinations**: Only run combinations that haven't been processed yet
- **Track progress**: See which combinations have been run, their scores, and any errors
- **Persistent storage**: Save progress to JSON files that can be shared or backed up

## Key Features

### 1. Enhanced Combination Storage

Combinations are now saved with additional metadata:

```json
{
  "metadata": {
    "timestamp": 1756309637.7104292,
    "movements": ["walking", "squats"],
    "subject": "test_subject",
    "session": "test_session",
    "max_combinations": 10,
    "random_seed": 42,
    "total_combinations": 5
  },
  "combinations": {
    "walking": [
      {
        "combination": {
          "movement": "walking",
          "weights": {"mono_mae_degrees": 1, "mono_mae_mm": 10, "mono_marker_mae": 100}
        },
        "ran": true,
        "score": 15.5,
        "error": null,
        "status": "processed"
      },
      {
        "combination": {
          "movement": "walking", 
          "weights": {"mono_mae_degrees": 10, "mono_mae_mm": 100, "mono_marker_mae": 1000}
        },
        "ran": false,
        "score": null,
        "error": null,
        "status": "pending"
      }
    ]
  }
}
```

### 2. Status Tracking

Each combination tracks:
- **`ran`**: Boolean indicating if the combination has been executed
- **`score`**: The total error score (lower is better)
- **`error`**: Error message if the combination failed
- **`status`**: Status string ("pending", "processed", "skipped_complete", "error")

### 3. Smart Resume Functionality

When starting a hyperparameter search:
1. **Load existing combinations**: Choose from saved combination files
2. **Check compatibility**: Verify movements match current selection
3. **Show progress**: Display completed vs pending combinations
4. **Run only unfinished**: Skip combinations that have already been processed

## Usage

### 1. Generate New Combinations

1. Go to the "Hyperparameter Search" tab in the validation app
2. Select movements, subject, and session
3. Click "Generate Parameter Combinations"
4. Combinations are automatically saved with status tracking

### 2. Load Existing Combinations

1. In the "Load Existing Combinations" section, you'll see:
   - 📊 Files with status tracking (new format)
   - 📁 Regular combination files (old format)

2. Select a file and click "Load Selected Combinations"
3. The system will show:
   - Progress summary (completed/pending/errors)
   - Best scores achieved so far
   - Compatibility with current settings

### 3. Start/Resume Search

1. Click "Start Hyperparameter Search"
2. The system will:
   - Show how many combinations are unfinished
   - Only process combinations that haven't been run
   - Update the status file after each combination completes

### 4. Monitor Progress

During execution, you can see:
- Real-time progress updates
- Status indicators for each combination (✅ Completed, ⏳ Pending, ❌ Error)
- Best scores achieved so far
- Overall completion percentage

## File Management

### File Types

1. **Status Files** (`hp_combinations_with_status_*.json`):
   - Include run status and scores
   - Can be resumed from any point
   - Show progress summary

2. **Regular Files** (`hp_combinations_*.json`):
   - Legacy format without status tracking
   - Can be converted to status format

### File Locations

- **Combination files**: `output/hp_combinations/`
- **Progress files**: `output/hp_search_progress/`

### File Naming

Files are named with timestamp and parameters:
```
hp_combinations_with_status_20250827_094717_test_subject_test_session.json
```

## API Functions

### Core Functions

```python
# Save combinations with status tracking
save_combinations_with_status(combinations, movements, subject, session, max_combinations, random_seed, hp_results=None)

# Load combinations with status
load_combinations_with_status(filepath)

# Update combination status
update_combination_status(filepath, movement, combo_index, ran=True, score=None, error=None, status="completed")

# Get summary statistics
get_combination_summary(enhanced_combinations)

# Get unfinished combinations
get_unfinished_combinations(enhanced_combinations)
```

### Utility Functions

```python
# Get list of saved files
get_saved_combination_files_with_status()
get_saved_combination_files()

# Check compatibility
check_combinations_compatibility(combinations, selected_movements)
```

## Example Workflow

### 1. Initial Setup

```python
# Generate combinations
combinations = generate_parameter_combinations(weight_ranges, max_combinations)

# Save with status tracking
filepath = save_combinations_with_status(
    combinations, movements, subject, session, 
    max_combinations, random_seed
)
```

### 2. Resume Search

```python
# Load existing combinations
combinations, metadata, run_status = load_combinations_with_status(filepath)

# Get unfinished combinations
unfinished = get_unfinished_combinations(enhanced_combinations)

# Process only unfinished combinations
for movement, combos in unfinished.items():
    for combo in combos:
        # Process combination
        result = process_combination(combo)
        
        # Update status
        update_combination_status(
            filepath, movement, combo_index,
            ran=True, score=result.score, status="processed"
        )
```

### 3. Monitor Progress

```python
# Get summary
summary = get_combination_summary(enhanced_combinations)
print(f"Progress: {summary['ran_combinations']}/{summary['total_combinations']}")
print(f"Best scores: {summary['best_scores']}")
```

## Benefits

1. **Time Savings**: Skip already completed combinations
2. **Fault Tolerance**: Resume from interruptions
3. **Progress Tracking**: Monitor search progress in real-time
4. **Resource Efficiency**: Avoid redundant computations
5. **Collaboration**: Share combination files with team members
6. **Backup**: Save progress for later analysis

## Migration from Old Format

Existing combination files can be loaded and converted to the new format:

1. Load the old combination file
2. Click "Save with Status and Scores" 
3. The new file will include status tracking fields


## Troubleshooting

### Common Issues

1. **File not found**: Check the `output/hp_combinations/` directory
2. **Compatibility issues**: Ensure movements match between files
3. **Status not updating**: Verify file permissions and disk space
4. **Progress not showing**: Check if the file has status tracking enabled

### Debug Information

The system logs detailed information about:
- File operations (save/load)
- Status updates
- Progress calculations
- Error conditions

Check the logs for troubleshooting information.
