import asyncio
import json
import time
from pathlib import Path
from loguru import logger

from app.services.qwen_vlm import QwenVLMService
from app.services.native_qwen_vlm import NativeQwenVLMService

def calculate_jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

async def validate_models():
    logger.info("Starting Side-by-Side Validation: Ollama vs vLLM...")
    
    frames_dir = Path("data/frames")
    all_frames = list(frames_dir.glob("*/*.jpg"))
    if not all_frames:
        logger.error("No frames found in data/frames/")
        return
        
    # Select 10 random frames (or first 10 for deterministic testing)
    test_frames = all_frames[:10]
    
    batch_data = []
    for i, path in enumerate(test_frames):
        video_id = path.parent.name
        frame_id = f"{video_id}_f{i}"
        ts = float(i)
        batch_data.append((frame_id, video_id, ts, path))
        
    logger.info("Running baseline Ollama Service...")
    start_time = time.time()
    ollama_results = await QwenVLMService.generate_metadata_batch(batch_data)
    ollama_time = time.time() - start_time
    
    logger.info("Running Native vLLM Service...")
    start_time = time.time()
    vllm_results = await NativeQwenVLMService.generate_metadata_batch(batch_data)
    vllm_time = time.time() - start_time
    
    # Compare results
    logger.info("--- COMPARISON REPORT ---")
    logger.info(f"Ollama Time: {ollama_time:.2f}s | vLLM Time: {vllm_time:.2f}s")
    
    total_actor_sim = 0.0
    total_object_sim = 0.0
    total_activity_sim = 0.0
    
    for i in range(len(test_frames)):
        ollama_meta = ollama_results[i][0] if isinstance(ollama_results[i], tuple) else ollama_results[i]
        vllm_meta = vllm_results[i][0] if isinstance(vllm_results[i], tuple) else vllm_results[i]
        
        o_actors = set()
        for evt in ollama_meta.events:
            o_actors.update(evt.actors)
            
        v_actors = set()
        for evt in vllm_meta.events:
            v_actors.update(evt.actors)
            
        o_objects = set([obj.subtype for obj in ollama_meta.objects])
        v_objects = set([obj.subtype for obj in vllm_meta.objects])
        
        o_activities = set(ollama_meta.activities)
        v_activities = set(vllm_meta.activities)
        
        actor_sim = calculate_jaccard_similarity(o_actors, v_actors)
        object_sim = calculate_jaccard_similarity(o_objects, v_objects)
        activity_sim = calculate_jaccard_similarity(o_activities, v_activities)
        
        total_actor_sim += actor_sim
        total_object_sim += object_sim
        total_activity_sim += activity_sim
        
        logger.info(f"Frame {i}: Actors Sim: {actor_sim:.2f} | Objects Sim: {object_sim:.2f} | Activities Sim: {activity_sim:.2f}")
        logger.info(f"  Ollama Caption: {ollama_meta.caption}")
        logger.info(f"  vLLM Caption:   {vllm_meta.caption}")
        
    n = len(test_frames)
    logger.info("--- FINAL AVERAGE SIMILARITY ---")
    logger.info(f"Actors:     {total_actor_sim/n:.2%}")
    logger.info(f"Objects:    {total_object_sim/n:.2%}")
    logger.info(f"Activities: {total_activity_sim/n:.2%}")
    
    if (total_object_sim/n) > 0.8:
        logger.info("Validation PASSED: Quality parity is acceptable.")
    else:
        logger.warning("Validation FAILED: Quality disparity detected.")

if __name__ == "__main__":
    asyncio.run(validate_models())
