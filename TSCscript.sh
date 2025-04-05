#!/bin/bash

# Run mambaTimeSeriesBinaryClassification.py for various input force constants
# and capture console output using tee, saving each run's output to a separate file

# Force constants to test
force_constants=(10 1 0.1 0.01)

# Target output directory (convert Windows path to Unix-style if using Git Bash or WSL)
output_dir="/c/hu650776/SynologyDrive/TSCoutputLogs"

# Make sure the directory exists
mkdir -p "$output_dir"

# Loop through force constants
for force_constant in "${force_constants[@]}"; do
  # Define output file based on current force constant
  output_file="${output_dir}/output_force_${force_constant}.log"

  echo "Running with force constant: $force_constant"
  
  # Run the script and save output
  python scripts/conferences/ASC2025/mambaTimeSeriesBinaryClassification.py "$force_constant" 1000 | tee "$output_file"
done
