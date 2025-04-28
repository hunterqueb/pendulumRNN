#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various force constants and modes
# Save console output using tee into appropriately named log files

# Force constants to test
force_constants=(1)

# Modes to test (third argument to Python script)
modes=(3)

#  sequence lengths to test
seq_lengths=(500 250 100 50 10)

# Output directory (Git Bash format for C:\ drive)
output_dir="/e/SynologyDrive/TSCHohmannOutputLogs"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through modes and force constants
for mode in "${modes[@]}"; do
  for seq_length in "${seq_lengths[@]}"; do

    # Construct output filename
    output_file="${output_dir}/output_force_1_noise_pos_${mode}_sl_${seq_length}.log"

    echo "Running with sequence length: $seq_length, only using pos data: $mode"

    # Execute the Python script with the force constant and mode
    python scripts/classification/mambaTimeSeriesBinaryClassificationHohmannComplexForce.py --trainDim "$mode" --no-plot --seqLength "$seq_length" | tee "$output_file"
  done
done