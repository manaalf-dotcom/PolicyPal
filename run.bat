@echo off
echo ================================
echo       Starting PolicyPal
echo ================================
echo.

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Running Streamlit...
python -m streamlit run app.py

pause