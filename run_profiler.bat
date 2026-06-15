@echo off
cd "C:\Mukul K\vinfo1\video-search-engine"
"C:\Mukul K\vinfo1\video-search-engine\.venv\Scripts\pytest.exe" test_profiler.py -s > "C:\Mukul K\vinfo1\video-search-engine\profiler_out.txt" 2>&1
echo Done.
