@echo off
cd /d "d:\james BA"
call venv_sentiment\Scripts\activate.bat
pip install streamlit wordcloud -q
echo.
echo ================================================
echo Starting Sentiment Analysis Dashboard...
echo ================================================
echo Open browser to: http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run app.py
pause
