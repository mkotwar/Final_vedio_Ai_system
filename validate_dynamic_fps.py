import asyncio
from app.services.frame import FrameExtractionService
from app.services.video import VideoService
from app.core.config import settings
from loguru import logger
import sys
import json

logger.remove()
logger.add(sys.stdout, level="INFO")

# We use mock model to avoid waiting for Qwen during this unit test
settings.MOCK_MODEL = True

async def main():
    video_id = "03301eee-50a4-4a3a-b5db-11f29b339233" # 4MB test video
    print(f"Testing video: {video_id} with PROFILE={settings.VIDEO_PROFILE}")
    
    stats = await FrameExtractionService.extract_frames(video_id)
    
    print("\n--- TEST RESULTS ---")
    with open("test_out.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    with open("data/metadata/" + video_id + "_frames.json", "r") as f:
        frames = json.load(f)
        print(f"Extracted {len(frames)} frames to metadata catalog.")

if __name__ == "__main__":
    asyncio.run(main())
