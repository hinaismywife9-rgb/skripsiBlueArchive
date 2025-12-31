#!/bin/bash
cd "d:\james BA"
source venv_sentiment/Scripts/activate
pip install streamlit wordcloud
streamlit run app.py
