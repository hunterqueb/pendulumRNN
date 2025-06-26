#!/bin/bash
# This script generates GMAT spacecraft thrust commands for a specified number of runs.
# Usage: ./generateGMATSpacecraftThrusts.sh <number_of_runs> <numMinsToProp> [lowerAlt] [upperAlt]

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <number_of_runs> <numMinsToProp> [lowerAlt] [upperAlt]"
    exit 1
fi

num_runs=$1
numMinsToProp=$2
lowerAlt=${3:-200}
upperAlt=${4:-250}

echo "Generating GMAT spacecraft thrust commands for $num_runs runs with $numMinsToProp minutes to propagate."
echo "Altitude bounds: lower=$lowerAlt, upper=$upperAlt"

python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType chem --lowerAlt $lowerAlt --upperAlt $upperAlt
python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType elec --lowerAlt $lowerAlt --upperAlt $upperAlt
python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType imp --lowerAlt $lowerAlt --upperAlt $upperAlt
python gmat/scripts/generateSpacecraftThrustOpt.py --numRandSys $num_runs --numMinProp $numMinsToProp --propType none --lowerAlt $lowerAlt --upperAlt $upperAlt