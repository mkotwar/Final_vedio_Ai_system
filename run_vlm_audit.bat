@echo off
echo Executing Frame Extraction...
python extract_vlm_frames_fixed.py
echo.
echo Executing VLM Validation Audit...
python validation_runner_vlm.py
pause
