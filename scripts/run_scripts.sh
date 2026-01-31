#!/bin/bash

python scripts/download_data.py

python scripts/merge_data.py

python -m scripts.remove_outliers
python -m scripts.make_dictionary
