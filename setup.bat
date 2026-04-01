@echo off
echo ================================
echo   PolicyPal Environment Setup
echo ================================
echo.

echo [1/4] Creating virtual environment...
python -m venv .venv

echo.
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [4/4] Installing requirements...
python -m pip install -r requirements.txt

echo.
echo =================================
echo   Setup Complete!
echo =================================
pause