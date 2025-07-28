#!/bin/bash

python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test leo --save

python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test geo --save

python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test leo --propMin 3 --save

python scripts/classification/mambaTimeSeriesClassificationGMATThrusts.py --OE --norm --energy --classic --nearest --test geo --propMin 3 --save

