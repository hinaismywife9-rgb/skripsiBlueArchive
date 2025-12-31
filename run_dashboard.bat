@echo off
REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv_sentiment" (
    echo Creating virtual environment...
    python -m venv venv_sentiment
)

REM Activate virtual environment
call venv_sentiment\Scripts\activate.bat

REM Install/upgrade required packages
echo Installing dependencies...
pip install streamlit pandas numpy torch transformers matplotlib seaborn wordcloud -q

REM Run Streamlit app
echo.
echo ================================================
echo Starting Sentiment Analysis Dashboard...
echo ================================================
echo Open browser to: http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run app.py

pause
