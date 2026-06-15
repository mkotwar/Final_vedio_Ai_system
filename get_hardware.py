import torch
import json
import os

info = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "gpus": []
}

if info["cuda_available"]:
    for i in range(info["gpu_count"]):
        cap = torch.cuda.get_device_capability(i)
        props = torch.cuda.get_device_properties(i)
        info["gpus"].append({
            "id": i,
            "name": torch.cuda.get_device_name(i),
            "vram_gb": round(props.total_memory / (1024**3), 2),
            "capability": f"{cap[0]}.{cap[1]}"
        })

with open("hardware_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("Hardware info saved.")
