@echo off
cd /d "%~dp0"

echo ============================================================
echo   KnowledgeBase RAG - Launching two Streamlit apps
echo ============================================================
echo   Upload UI : http://127.0.0.1:8502
echo   Chat UI   : http://127.0.0.1:8501
echo.
echo   Close the two console windows to stop the servers.
echo ============================================================
echo.

start "KB-Upload-8502" cmd /k ".venv\Scripts\python.exe -m streamlit run app_upload.py --server.port 8502 --server.headless true"
start "KB-Chat-8501"   cmd /k ".venv\Scripts\python.exe -m streamlit run app_chat.py --server.port 8501 --server.headless true"

timeout /t 10 /nobreak >nul

start "" http://127.0.0.1:8502
start "" http://127.0.0.1:8501
