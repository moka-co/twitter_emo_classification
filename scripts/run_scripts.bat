@echo off

python scripts\\download_data.py
python scripts\\merge_data.py
python -m scripts\\remove_outliers.py

pause
