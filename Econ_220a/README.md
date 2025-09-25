PS1 - ECON 220a Problem Set 1

## Software:
The code is written in python and assumes that your device has python installed. Tested with Python 3.8.

## Files
- The data is stored in 01_Data
- The main script is 02_Code/ps1.py
- 02_Code/requirements.txt has all the packages required to run ps1.py. It is called using the instructions below
- All tables are saved to 03_Output
- Econ_220a_PS1.pdf has the written answers

## Instructions
Run the following commands in sequence in Command Prompt/Terminal to create the virtual environment, install the required packages, and run the full file.

Windows
1. Open Command Prompt or Powershell and set cd to this folder.
2. Run: python -m venv .venv
3. Run: .\.venv\Scripts\python -m pip install --upgrade pip
4. Run: .\.venv\Scripts\python -m pip install -r 02_Code\requirements.txt
5. Run: .\.venv\Scripts\python 02_Code\ps1.py

macOS/Linux
1. Open Terminal and set cd to this folder
2. Run: python3 -m venv .venv
3. Run: source .venv/bin/activate
4. Run: python -m pip install --upgrade pip
5. Run: python -m pip install -r 02_Code/requirements.txt
6. Run: python 02_Code/ps1.py

What each line does
- Create venv: makes an isolated Python environment in .venv so package versions don’t conflict with other projects.
- Upgrade pip: ensures the installer is up to date to avoid old resolver issues.
- Install requirements: installs only the packages listed in 02_Code/requirements.txt.
- Run script: executes the main file using the environment you just prepared.

