import asyncio
import time
from pathlib import Path
from loguru import logger
import json

from app.services.qwen_vlm import QwenVLMService
from app.core.config import settings

async def run_benchmark():
    logger.info("Starting Ollama Benchmark...")
    
    # 1. Collect 50 frames
    frames_dir = Path("data/frames")
    all_frames = list(frames_dir.glob("*/*.jpg"))
    if not all_frames:
        logger.error("No frames found in data/frames/")
        return
        
    test_frames = all_frames[:50]
    logger.info(f"Selected {len(test_frames)} frames for benchmarking.")
    
    batch_data = []
    for i, path in enumerate(test_frames):
        video_id = path.parent.name
        frame_id = f"{video_id}_f{i}"
        ts = float(i)
        batch_data.append((frame_id, video_id, ts, path))
        
    # 2. Run baseline benchmark
    start_time = time.perf_counter()
    results = await QwenVLMService.generate_metadata_batch(batch_data)
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    sec_per_frame = total_time / len(test_frames) if test_frames else 0
    
    metrics = {
        "frames": len(test_frames),
        "runtime": f"{total_time:.2f} sec",
        "sec_per_frame": f"{sec_per_frame:.2f} sec/frame",
        "gpu_util": "N/A (Ollama manages its own GPU process)",
        "vram": "N/A (Ollama manages its own VRAM)",
        "cpu": "N/A"
    }
    
    print("\n--- OLLAMA BASELINE BENCHMARK RESULTS ---")
    print(json.dumps(metrics, indent=2))
    
    with open("benchmark_ollama_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    asyncio.run(run_benchmark())
