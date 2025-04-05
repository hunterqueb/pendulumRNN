#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various force constants and modes
# Save console output using tee into appropriately named log files

# Force constants to test
force_constants=(10 1 0.1 0.01)

# Modes to test (third argument to Python script)
modes=(1 2)

# Output directory (Git Bash format for C:\ drive)
output_dir="/c/hu650776/SynologyDrive/TSCoutputLogs"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through modes and force constants
for mode in "${modes[@]}"; do
  for force_constant in "${force_constants[@]}"; do
    # Sanitize force constant for file name (replace '.' with '_')
    sanitized_fc=$(echo "$force_constant" | sed 's/\./_/')

    # Construct output filename
    output_file="${output_dir}/output_force_${sanitized_fc}_layers_${mode}.log"

    echo "Running with force constant: $force_constant, number of layers: $mode"

    # Execute the Python script with the force constant and mode
    python scripts/conferences/ASC2025/mambaTimeSeriesBinaryClassification.py "$force_constant" 10000 "$mode" | tee "$output_file"
  done
done
