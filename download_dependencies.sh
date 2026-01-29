#!/bin/bash

# Create virtual environment if the directory venv doesn't exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate the environment
source venv/bin/activate

# Ensure pip is update
python3 -m install --upgrade pip

# Install tqdm
python3 -m install tqdm

# Install current directory package
pip install .