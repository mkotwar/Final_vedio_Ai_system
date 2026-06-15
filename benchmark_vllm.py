import time
import json
import psutil
import torch
from pathlib import Path
from loguru import logger
from app.services.qwen_vlm_vllm import NativeQwenVLLMService

def run_vllm_benchmark():
    logger.info("Starting vLLM Benchmark...")
    
    frames_dir = Path("data/frames")
    all_frames = list(frames_dir.glob("*/*.jpg"))
    if not all_frames:
        logger.error("No frames found in data/frames/")
        return
        
    test_frames = all_frames[:50]
    logger.info(f"Selected {len(test_frames)} frames.")
    
    prompt = "Describe the contents of this image in a strict JSON format matching: {\"scene_type\": \"\", \"objects\": [], \"events\": []}"
    
    # Pre-load to avoid measuring load time
    NativeQwenVLLMService.load_model()
    
    batch_sizes = [8, 16, 32]
    results = {}
    
    for bs in batch_sizes:
        logger.info(f"--- Running vLLM benchmark with Max Batch Size {bs} ---")
        
        start_time = time.perf_counter()
        
        # For vLLM, we can just throw all requests at once and let its continuous batching handle it,
        # but to simulate batch sizes we chunk it or just set max_num_seqs if we restarted engine.
        # Since vLLM manages it, we will just chunk it to measure pure throughput of that chunk.
        for i in range(0, len(test_frames), bs):
            batch_paths = test_frames[i:i+bs]
            logger.info(f"Processing chunk of {len(batch_paths)} frames...")
            _ = NativeQwenVLLMService.generate_batch(batch_paths, prompt)
            
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        sec_per_frame = total_time / len(test_frames)
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        cpu_util = psutil.cpu_percent()
        
        metrics = {
            "frames": len(test_frames),
            "chunk_size": bs,
            "runtime": f"{total_time:.2f} sec",
            "sec_per_frame": f"{sec_per_frame:.2f} sec/frame",
            "peak_vram_gb": f"Managed by vLLM (Reserved approx 90% default)",
            "cpu_util": f"{cpu_util}%"
        }
        
        results[f"batch_{bs}"] = metrics
        print(json.dumps(metrics, indent=2))
        
    with open("benchmark_vllm_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_vllm_benchmark()
