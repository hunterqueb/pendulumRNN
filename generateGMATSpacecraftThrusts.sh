#!/bin/bash
# This script generates GMAT spacecraft thrust commands for a specified number of runs.
# Usage: ./generateGMATSpacecraftThrusts.sh <number_of_runs> <numMinsToProp>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <number_of_runs> <numMinsToProp>"
    exit 1
fi

num_runs=$1
numMinsToProp=$2

echo "Generating GMAT spacecraft thrust commands for $num_runs runs with $numMinsToProp minutes to propagate."

python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType chem --lowerAlt 300 --upperAlt 400
python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType elec --lowerAlt 300 --upperAlt 400
python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType imp --lowerAlt 300 --upperAlt 400
python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType none --lowerAlt 300 --upperAlt 400