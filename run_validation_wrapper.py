import os
import sys

# Force the CWD to be the project root
target_dir = r"c:\Mukul K\vinfo1\video-search-engine"
os.chdir(target_dir)

print(f"Current working directory: {os.getcwd()}")

# Import and run the validation script natively in the same process
try:
    import validation_runner_vlm
    import asyncio
    
    asyncio.run(validation_runner_vlm.run_vlm_validation())
except Exception as e:
    import traceback
    with open("debug_trace.txt", "w") as f:
        f.write("FATAL ERROR:\n")
        f.write(traceback.format_exc())
    print("FATAL ERROR:", e)
