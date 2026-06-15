import time
import json
import psutil
import torch
from pathlib import Path
from loguru import logger
from app.services.qwen_vlm_hf import NativeQwenTransformersService

def run_hf_benchmark():
    logger.info("Starting HuggingFace Benchmark...")
    
    frames_dir = Path("data/frames")
    all_frames = list(frames_dir.glob("*/*.jpg"))
    if not all_frames:
        logger.error("No frames found in data/frames/")
        return
        
    test_frames = all_frames[:50]
    logger.info(f"Selected {len(test_frames)} frames.")
    
    prompt = "Describe the contents of this image in a strict JSON format matching: {\"scene_type\": \"\", \"objects\": [], \"events\": []}"
    
    batch_sizes = [1, 4, 8, 16]
    results = {}
    
    for bs in batch_sizes:
        logger.info(f"--- Running HF benchmark with Batch Size {bs} ---")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        
        for i in range(0, len(test_frames), bs):
            batch_paths = test_frames[i:i+bs]
            logger.info(f"Processing batch of {len(batch_paths)} frames...")
            _ = NativeQwenTransformersService.generate_batch(batch_paths, prompt)
            
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        sec_per_frame = total_time / len(test_frames)
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        cpu_util = psutil.cpu_percent()
        
        metrics = {
            "frames": len(test_frames),
            "batch_size": bs,
            "runtime": f"{total_time:.2f} sec",
            "sec_per_frame": f"{sec_per_frame:.2f} sec/frame",
            "peak_vram_gb": f"{peak_vram_gb:.2f} GB",
            "cpu_util": f"{cpu_util}%"
        }
        
        results[f"batch_{bs}"] = metrics
        print(json.dumps(metrics, indent=2))
        
    with open("benchmark_hf_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_hf_benchmark()
