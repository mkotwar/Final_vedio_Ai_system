import sys
import os

output_file = r"c:\Mukul K\vinfo1\video-search-engine\env_baseline.txt"

with open(output_file, "w") as f:
    try:
        f.write("--- Runtime Environment Baseline ---\n\n")
        f.write(f"Python version: {sys.version}\n")
        
        import torch
        f.write(f"Torch version: {torch.__version__}\n")
        f.write(f"CUDA version: {torch.version.cuda}\n")
        f.write(f"CUDA Available: {torch.cuda.is_available()}\n")
        
        if torch.cuda.is_available():
            f.write(f"GPU Detected: {torch.cuda.get_device_name(0)}\n")
            f.write(f"BF16 Supported: {torch.cuda.is_bf16_supported()}\n")
        else:
            f.write("GPU Detected: None\n")
            f.write("BF16 Supported: False\n")
            
        import transformers
        f.write(f"Transformers version: {transformers.__version__}\n")
        
    except Exception as e:
        f.write(f"Error during baseline check: {str(e)}\n")
