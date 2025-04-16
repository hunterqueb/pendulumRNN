#!/bin/bash

# Default argument value
arg=${1:-500}

# Run each propagation script with the provided argument
./runGMATShortProp.sh "$arg"
echo "GMAT Short Propagation completed."
./runGMATLongProp.sh "$arg"
echo "GMAT Long Propagation completed."
./runCR3BPShortProp.sh "$arg"
echo "CR3BP Short Propagation completed."
./runCR3BPLongProp.sh "$arg"
echo "CR3BP Long Propagation completed."
