import sys
import json
import traceback

def write_result(data):
    with open(r"c:\Mukul K\vinfo1\video-search-engine\VALIDATION_OUTPUT.json", "w") as f:
        json.dump(data, f, indent=2)

data = {"status": "started", "logs": []}
write_result(data)

try:
    import torch
    import asyncio
    sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
    
    from app.services.frame import FrameService
    from app.services.summary_service import SummaryService
    from app.services.incident_engine import IncidentEngine
    from app.services.narrative_builder import NarrativeBuilder
    from app.services.poster_service import PosterService
    from app.services.qwen_vlm_hf import NativeQwenTransformersService
    from app.core.qdrant_manager import QdrantManager
    
    data["logs"].append("Imports successful")
    
    video_id = "32ac5bc9-91ea-4cfe-8d82-a383f6d608c4"
    
    async def run_pipeline():
        try:
            data["logs"].append("Loading model...")
            write_result(data)
            NativeQwenTransformersService.load_model()
            data["logs"].append("Model loaded successfully")
            
            data["logs"].append("Extracting frames...")
            write_result(data)
            extraction_result = await FrameService.extract_frames(video_id)
            data["extraction"] = extraction_result
            data["logs"].append("Frames extracted")
            
            data["logs"].append("Aggregating events...")
            write_result(data)
            events = SummaryService.generate_summary(video_id)
            data["events"] = [e for e in events]
            
            data["logs"].append("Correlating incidents...")
            write_result(data)
            incidents = IncidentEngine.correlate_events(events)
            data["incidents"] = incidents
            
            data["logs"].append("Generating poster...")
            write_result(data)
            poster = await PosterService.generate_poster(video_id)
            data["poster"] = poster
            
            data["logs"].append("Verifying search indexing...")
            write_result(data)
            qm = QdrantManager()
            points, _ = qm.client.scroll(collection_name="video_events", limit=500)
            data["indexed_events"] = [p.payload for p in points if p.payload.get("video_id") == video_id]
            
            if torch.cuda.is_available():
                data["vram"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                    "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "peak_gb": torch.cuda.max_memory_allocated() / (1024**3)
                }
            
            data["status"] = "success"
            write_result(data)
        except Exception as e:
            data["status"] = "error"
            data["error"] = str(e)
            data["traceback"] = traceback.format_exc()
            write_result(data)

    asyncio.run(run_pipeline())

except Exception as e:
    data["status"] = "fatal"
    data["error"] = str(e)
    data["traceback"] = traceback.format_exc()
    write_result(data)
