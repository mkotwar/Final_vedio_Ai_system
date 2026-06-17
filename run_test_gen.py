import sys
import traceback
from pathlib import Path

def main():
    try:
        import torch
        import time
        # Add project root to path
        sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
        from app.services.qwen_vlm_hf import NativeQwenTransformersService
        
        print("Loading model...")
        NativeQwenTransformersService.load_model()
        print("Model loaded.")
        
        # create a dummy image
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8))
        dummy_path = Path(r"c:\Mukul K\vinfo1\video-search-engine\dummy_test_img.jpg")
        dummy_img.save(dummy_path)
        
        print("Running generate_batch...")
        results = NativeQwenTransformersService.generate_batch(
            [dummy_path], 
            "Describe the scene."
        )
        print("Generation Results:", results)
        
    except Exception as e:
        print("CRASHED:", e)
        traceback.print_exc()

if __name__ == "__main__":
    with open(r"c:\Mukul K\vinfo1\video-search-engine\py_test_out.txt", "w") as f:
        sys.stdout = f
        sys.stderr = f
        main()
