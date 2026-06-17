import asyncio
import sys
from pathlib import Path

from app.services.vlm_factory import get_vlm_service
from app.core.config import settings

async def main():
    print("Starting test_qwen_metadata...", flush=True)
    video_dir = settings.DATA_DIR / "frames"
    sample_img = None
    for p in video_dir.rglob("*.jpg"):
        sample_img = p
        break
    
    if not sample_img:
        print("No sample frames found!", flush=True)
        sys.exit(1)
        
    print(f"Testing with image: {sample_img}", flush=True)
    
    # Initialize mock batch of 1 frame
    batch = [("test_frame_01", "test_video", 5.0, sample_img)]
    
    try:
        service = get_vlm_service()
        print("Calling generate_metadata_batch...", flush=True)
        results = await service.generate_metadata_batch(batch)
        print("Success! Generated rich metadata.", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
