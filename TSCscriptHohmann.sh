#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various force constants and modes
# Save console output using tee into appropriately named log files

# Force constants to test
force_constants=(10000 100 1)

# Modes to test (third argument to Python script)
modes=(4 2)

# Output directory (Git Bash format for C:\ drive)
output_dir="/c/Users/hu650776/SynologyDrive/TSCHohmannOutputLogs"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through modes and force constants
for mode in "${modes[@]}"; do
  for force_constant in "${force_constants[@]}"; do
    # Sanitize force constant for file name (replace '.' with '_')
    sanitized_fc=$(echo "$force_constant" | sed 's/\./_/')

    # Construct output filename
    output_file="${output_dir}/output_force_${sanitized_fc}_noise_pos_${mode}.log"

    echo "Running with force constant: $force_constant, only using pos data: $mode"

    # Execute the Python script with the force constant and mode
    python scripts/classification/mambaTimeSeriesBinaryClassificationHohmann.py --deltaV "$force_constant" --trainDim "$mode" --no-plot | tee "$output_file"
  done
done
