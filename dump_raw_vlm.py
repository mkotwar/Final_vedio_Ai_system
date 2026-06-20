import asyncio
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.config import settings
from app.services.vlm_factory import get_vlm_service

async def dump_raw_vlm():
    vlm_service = get_vlm_service()
    FRAMES_DIR = Path(PROJECT_ROOT) / "validation" / "vlm" / "frames"
    
    # Let's just grab all images in the frames directory
    images = list(FRAMES_DIR.glob("*.jpg"))
    if not images:
        print("No images found.")
        return
        
    batch = []
    for idx, img_path in enumerate(images):
        batch.append((f"test_f{idx}", "test_vid", 0.0, img_path))
        
    print(f"Running VLM on {len(batch)} frames...")
    try:
        results = await vlm_service.generate_metadata_batch(batch)
    except Exception as e:
        print(f"Error: {e}")
        return
        
    output_data = {}
    for (frame_id, vid, ts, path), (rich_meta, timings) in zip(batch, results):
        output_data[path.name] = rich_meta.model_dump()
        
    with open("raw_vlm_outputs.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
        
    print("Dumped to raw_vlm_outputs.json")

if __name__ == "__main__":
    asyncio.run(dump_raw_vlm())
