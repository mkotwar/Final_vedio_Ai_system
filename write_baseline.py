import sys
import torch
import transformers

with open(r"C:\Users\Vinfocom\.gemini\antigravity-ide\brain\5a327ec8-e4a1-4023-a28a-14cde3ab0c24\baseline.txt", "w") as f:
    f.write(f"Python version: {sys.version}\n")
    f.write(f"Torch version: {torch.__version__}\n")
    f.write(f"Transformers version: {transformers.__version__}\n")
    f.write(f"CUDA version: {torch.version.cuda}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"GPU Name: {torch.cuda.get_device_name(0)}\n")
        f.write(f"BF16 Supported: {torch.cuda.is_bf16_supported()}\n")
