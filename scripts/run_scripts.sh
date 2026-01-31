#!/bin/bash

python -m scripts.download_data

python -m scripts.merge_data

python -m scripts.remove_outliers
python -m scripts.make_dictionary
