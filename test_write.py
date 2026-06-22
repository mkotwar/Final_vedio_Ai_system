import subprocess
import os

print("Starting subprocess test run...")
# Target the virtual env python and run validation_runner.py
python_exe = r"c:\Mukul K\vinfo1\video-search-engine\.venv\Scripts\python.exe"
script_path = "validation_runner.py"

res = subprocess.run([python_exe, script_path], capture_output=True, text=True)

output_file = "test_results.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== STDOUT ===\n")
    f.write(res.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(res.stderr)

print("Subprocess run complete. Outputs written.")
