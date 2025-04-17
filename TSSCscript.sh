#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various force constants and modes
# Save console output using tee into appropriately named log files

# Force constants to test
force_constants=(10 1 0.1)


# Output directory (Git Bash format for C:\ drive)
# output_dir="/c/Users/hu650776/SynologyDrive/TSSCoutput"
output_dir="/e/SynologyDrive/TSSCoutput"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through modes and force constants
for force_constant in "${force_constants[@]}"; do
  # Sanitize force constant for file name (replace '.' with '_')
  sanitized_fc=$(echo "$force_constant" | sed 's/\./_/')

  # Construct output filename
  output_file="${output_dir}/output_force_${sanitized_fc}.log"

  echo "Running with force constant: $force_constant"

  # Execute the Python script with the force constant and mode
  python scripts/classification/seqClassifier.py --forceConst "$force_constant" --no-plot | tee "$output_file"
done
