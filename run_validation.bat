@echo off
cd /d "c:\Mukul K\vinfo1\video-search-engine"
call "c:\Mukul K\vinfo1\video-search-engine\.venv\Scripts\activate.bat"
python "c:\Mukul K\vinfo1\video-search-engine\validation_runner.py" > "c:\Mukul K\vinfo1\video-search-engine\run_output.log" 2>&1
echo DONE >> "c:\Mukul K\vinfo1\video-search-engine\run_output.log"
