@echo off
.\.venv\Scripts\python.exe test_gpu_diagnostic.py > diag_output_final.txt 2>&1
.\.venv\Scripts\pip.exe list > pip_list_final.txt 2>&1
echo Done.
