import sys
import traceback
from pathlib import Path

ROOT = Path(r"C:\Mukul K\vinfo1\video-search-engine")
sys.path.insert(0, str(ROOT))

try:
    print("Importing app.main...")
    import app.main
    print("Import successful!")
except Exception as e:
    print("FAILED TO IMPORT APP.MAIN:")
    traceback.print_exc()
