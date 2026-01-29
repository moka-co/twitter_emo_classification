@echo off

:: Create a virtual environment if it doesn't exist
if not exist "venv" python -m venv venv

:: Activate the environment
call venv\Scripts\activate

:: Ensure pip is up to date
python -m pip install --upgrade pip

:: Install tqdm
python -m pip install tqdm

:: Install the current directory package
pip install .

pause