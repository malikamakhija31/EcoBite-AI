@echo off
title EcoBite AI Dashboard
echo ========================================================
echo   Starting EcoBite AI Streamlit Dashboard (Dark Mode)
echo ========================================================
echo Dashboard will launch in your browser at: http://localhost:8501
echo.

if exist ".venv\Scripts\streamlit.exe" (
    ".venv\Scripts\streamlit.exe" run dashboard\app.py
) else (
    streamlit run dashboard\app.py
)

pause
