import torch
import json
import os
import subprocess
import sys

output_path = r"c:\Mukul K\vinfo1\video-search-engine\hardware_out.txt"

with open(output_path, "w") as f:
    f.write(f"PyTorch Version: {torch.__version__}\n")
    f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"CUDA Version: {torch.version.cuda}\n")
        f.write(f"Device Count: {torch.cuda.device_count()}\n")
        for i in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024**3)
            f.write(f"GPU {i}: {torch.cuda.get_device_name(i)} VRAM: {vram:.2f} GB Capability: {cap[0]}.{cap[1]}\n")
            
    f.write("\nNVIDIA-SMI Output:\n")
    try:
        smi = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        f.write(smi)
    except Exception as e:
        f.write(f"Failed to run nvidia-smi: {e}\n")
        
print("Hardware info dumped successfully.")
sys.stdout.flush()
