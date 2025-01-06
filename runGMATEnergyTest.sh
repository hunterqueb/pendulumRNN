#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_times to run GMAT script>"
  exit 1
fi

COUNT=$1

for i in $(seq 1 $COUNT); do
  python scripts/journals/orbitProp/mambaGMATSaveEnergy.py  > /dev/null 2>&1
  echo "Run $i is done."

done


