#!/bin/bash



python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --propMin 3 --systems 10000 --classic --save



python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --propMin 30 --systems 10000 --classic --save
python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --propMin 100 --systems 10000 --classic --save


