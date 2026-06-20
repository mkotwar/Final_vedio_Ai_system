import os
from pathlib import Path

REPORTS_DIR = Path(r"c:\Mukul K\vinfo1\video-search-engine\validation\vlm\reports")
file_path = REPORTS_DIR / "vlm_validation_audit.md"

if file_path.exists():
    print(f"File exists. Size: {file_path.stat().st_size} bytes")
else:
    print("File does not exist.")

with open(file_path, "a") as f:
    f.write("\n\nTEST INJECTION: SCRIPT REACHED THE END\n")

print("Appended test string.")
