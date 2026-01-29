@echo off
:: Ensure pip is up to date
python -m pip install --upgrade pip

:: Install tqdm
python -m pip install tqdm

:: Install the current directory package
pip install .

pause