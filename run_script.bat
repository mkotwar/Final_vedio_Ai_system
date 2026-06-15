@echo off
call .venv\Scripts\activate.bat
python analyze_13m.py > stdout.txt 2> stderr.txt
