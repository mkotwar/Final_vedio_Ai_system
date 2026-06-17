import sys
import traceback
from pathlib import Path
import time
import json
import asyncio

def main():
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("WARNING: CUDA is NOT available to PyTorch!")
            
        # Add project root to path
        sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
        from app.services.qwen_vlm_hf import NativeQwenTransformersService
        from app.schemas.frame import FrameRichMetadata
        
        print("\nLoading Native HF model...")
        start = time.perf_counter()
        NativeQwenTransformersService.load_model()
        print(f"Model loaded in {time.perf_counter() - start:.2f}s")
        
        if torch.cuda.is_available():
            print(f"VRAM Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
            print(f"VRAM Reserved:  {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
        
        # Check model device
        model = NativeQwenTransformersService._model
        print(f"\nModel Dtype: {model.dtype}")
        print(f"Model Device: {next(model.parameters()).device}")
        
        # create a dummy image
        from PIL import Image
        import numpy as np
        dummy_img = Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8))
        dummy_path = Path(r"c:\Mukul K\vinfo1\video-search-engine\dummy_test_img.jpg")
        dummy_img.save(dummy_path)
        
        print("\nRunning async batch generation...")
        batch = [("test_frame_1", "test_video", 1.0, dummy_path)]
        
        async def run_test():
            start_gen = time.perf_counter()
            results = await NativeQwenTransformersService.generate_metadata_batch(batch)
            print(f"Batch inference complete in {time.perf_counter() - start_gen:.2f}s")
            
            for meta, timings in results:
                print("\n--- Timings ---")
                print(json.dumps(timings, indent=2))
                print("\n--- Metadata Result ---")
                print(meta.model_dump_json(indent=2))
                
        asyncio.run(run_test())

    except Exception as e:
        print("CRASHED:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
