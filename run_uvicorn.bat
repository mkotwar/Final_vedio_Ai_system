@echo off
call "c:\Mukul K\vinfo1\video-search-engine\.venv\Scripts\activate.bat"
uvicorn app.main:app --host 127.0.0.1 --port 8000
