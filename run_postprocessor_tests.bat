@echo off
cd /d "c:\Mukul K\vinfo1\video-search-engine"
.venv\Scripts\python.exe -m pytest tests\test_metadata_postprocessor.py -v --tb=short > validation\reports\test_results.txt 2>&1
echo EXIT CODE: %ERRORLEVEL% >> validation\reports\test_results.txt
