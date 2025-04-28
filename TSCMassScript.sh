#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various force constants and modes
# Save console output using tee into appropriately named log files

# Force constants to test
num_classes=(3 4 5 6 7 8 9 10)


# Output directory (Git Bash format for C:\ drive)
output_dir="/c/Users/hu650776/SynologyDrive/TSCMassOutput"
# output_dir="/e/SynologyDrive/TSCMassOutput"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through modes and force constants
for num_class in "${num_classes[@]}"; do
  # Sanitize force constant for file name (replace '.' with '_')
  sanitized_nc=$(echo "$num_class" | sed 's/\./_/')

  # Construct output filename
  output_file="${output_dir}/output_classes_${sanitized_nc}.log"

  echo "Running with number of classification bins: $num_class"

  python scripts/classification/mambaTimeSeriesMassClassification.py --num_classes "$num_class"  | tee "$output_file"
done
