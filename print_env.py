import sys
import torch
import transformers

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
