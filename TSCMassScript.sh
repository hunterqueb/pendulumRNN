#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various force constants and modes
# Save console output using tee into appropriately named log files

# Force constants to test
num_classes=(3 4 5 6 7 8 9 10)
dampings=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Output directory (Git Bash format for C:\ drive)
output_dir="/c/Users/hu650776/SynologyDrive/TSCMassOutput"
# output_dir="/e/SynologyDrive/TSCMassOutput"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through modes and force constants
for damping in "${dampings[@]}"; do
  for num_class in "${num_classes[@]}"; do
    # Sanitize force constant for file name (replace '.' with '_')
    sanitized_nc=$(echo "$num_class" | sed 's/\./_/')
    sanitized_damp=$(echo "$damping" | sed 's/\./_/')

    # Construct output filename
    output_file="${output_dir}/output_classes_${sanitized_nc}_layers_2_${sanitized_damp}.log"

    echo "Running with number of classification bins: $num_class, with $damping damping coefficent."

    python scripts/classification/mambaTimeSeriesMassClassification.py --num_classes "$num_class" --layers 2 --damping "$damping" | tee "$output_file"
  done
done