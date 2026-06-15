import os
import sys
import time
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

# Ensure VLM_ENGINE_TYPE is set so config loads properly
os.environ["VLM_ENGINE_TYPE"] = "native_hf"
from app.services.qwen_vlm_hf import NativeQwenTransformersService

def create_dummy_images(num_images, tmp_dir):
    paths = []
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        p = tmp_dir / f"dummy_frame_{i}.jpg"
        if not p.exists():
            img = Image.new('RGB', (1280, 720), color = (i*10 % 255, i*20 % 255, i*30 % 255))
            img.save(p)
        paths.append(p)
    return paths

def main():
    print("=== Native HF Runtime Benchmark ===")
    print("Initializing NativeQwenTransformersService...")
    start_init = time.perf_counter()
    NativeQwenTransformersService.load_model()
    print(f"Loaded in {time.perf_counter() - start_init:.2f}s")
    
    tmp_dir = Path(__file__).resolve().parent.parent / "data" / "frames" / "benchmark"
    
    batch_sizes = [1, 4, 8]
    prompt = "Describe this image in detail. Be extremely descriptive."
    
    print("\nStarting benchmarks...")
    for bs in batch_sizes:
        print(f"\n--- Batch Size: {bs} ---")
        paths = create_dummy_images(bs, tmp_dir)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            vram_before = torch.cuda.memory_allocated() / (1024**3)
        
        start_inf = time.perf_counter()
        
        # We catch exceptions to prevent crash if bs=8 OOMs
        try:
            results = NativeQwenTransformersService.generate_batch(paths, prompt)
            end_inf = time.perf_counter()
            
            inf_time = end_inf - start_inf
            fps = bs / inf_time
            
            from loguru import logger
            logger.info(f"--- BATCH SIZE {bs} RESULTS ---")
            logger.info(f"Inference Time: {inf_time:.2f}s")
            logger.info(f"Throughput: {fps:.2f} frames/sec")
            
            if torch.cuda.is_available():
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
                logger.info(f"Peak VRAM: {peak_vram:.2f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                from loguru import logger
                logger.error(f"Out of memory on batch size {bs}!")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e

if __name__ == '__main__':
    main()
