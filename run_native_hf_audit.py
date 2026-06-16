import os
import sys
import time
import json
import torch
import asyncio
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")

log_path = r"c:\Mukul K\vinfo1\video-search-engine\py_audit_log.txt"
f_out = open(log_path, "w", encoding="utf-8")

def my_print(msg):
    print(msg)
    f_out.write(str(msg) + "\n")
    f_out.flush()

try:
    from app.services.frame import FrameExtractionService
    from app.services.summary_service import SummaryService
    from app.services.incident_engine import IncidentEngine
    from app.services.narrative_builder import NarrativeBuilderService
    from app.services.poster_service import PosterService
    from app.services.qwen_vlm_hf import NativeQwenTransformersService
    from app.core.qdrant_manager import QdrantManager
    from app.core.config import settings
except Exception as e:
    my_print(f"IMPORT ERROR: {e}")
    sys.exit(1)

def print_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        my_print(f"[{label}] VRAM Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak Alloc: {max_alloc:.2f}GB")
        return max_alloc
    else:
        my_print(f"[{label}] CUDA not available")
        return 0.0

async def run_audit():
    try:
        video_path = r"c:\Mukul K\vinfo1\video-search-engine\data\videos\32ac5bc9-91ea-4cfe-8d82-a383f6d608c4.mp4"
        if not os.path.exists(video_path):
            my_print(f"Error: Test video not found at {video_path}")
            return

        my_print("\n--- INITIALIZING AUDIT ---")
        my_print(f"Video: {video_path}")
        
        video_id = Path(video_path).stem
        my_print(f"Assigned Video ID: {video_id}")
        
        # Task 5: VRAM Before Load
        vram_before = print_memory("VRAM BEFORE MODEL LOAD")
        
        # Task 2: Native HF Activation Audit & Task 5: VRAM After Model Load
        my_print("\n--- LOADING NATIVE HF MODEL ---")
        start = time.time()
        try:
            NativeQwenTransformersService.load_model()
        except Exception as e:
            my_print(f"Model loading crashed: {e}")
            raise e
        my_print(f"Model loaded in {time.time() - start:.2f}s")
        vram_after_load = print_memory("VRAM AFTER MODEL LOAD")
        
        my_print("\n--- FRAME EXTRACTION & DYNAMIC FPS ---")
        start = time.time()
        try:
            extraction_result = await FrameExtractionService.extract_frames(video_id)
            my_print(f"Frame processing took {time.time() - start:.2f}s")
        except Exception as e:
            my_print(f"Frame extraction failed! {e}")
            import traceback
            my_print(traceback.format_exc())
            return

        processed_frames = extraction_result.get("processed_frames", 0)
        success_frames = extraction_result.get("successful_frames", 0)
        failed_frames = extraction_result.get("failed_frames", 0)
        
        vram_peak = print_memory("VRAM AFTER BATCH PROCESSING (PEAK)")

        my_print(f"\nCandidate Frames: {processed_frames}")
        my_print(f"Metadata Success: {success_frames}")
        my_print(f"Metadata Failed: {failed_frames}")
        
        my_print("\n--- EVENT AGGREGATION ---")
        start = time.time()
        events = []
        try:
            events = SummaryService.generate_summary(video_id)
            my_print(f"Event Aggregation took {time.time() - start:.2f}s")
            my_print(f"Events Generated: {len(events)}")
        except Exception as e:
            my_print(f"Event Aggregation Failed: {e}")
        
        my_print("\n--- INCIDENT CORRELATION ---")
        start = time.time()
        incidents = []
        try:
            incidents = IncidentEngine.correlate_events(events)
            my_print(f"Incident Correlation took {time.time() - start:.2f}s")
            my_print(f"Incidents Generated: {len(incidents)}")
        except Exception as e:
             my_print(f"Incident Correlation Failed: {e}")
             
        my_print("\n--- NARRATIVE BUILDER ---")
        start = time.time()
        narrative_mode = "Incident Report"
        narrative = ""
        try:
            narrative = NarrativeBuilderService.generate_investigation_report(str(events))
            my_print(f"Narrative Builder took {time.time() - start:.2f}s")
            my_print(f"Narrative Generated: {len(narrative)} characters")
        except Exception as e:
            my_print(f"Narrative Failed: {e}")
            
        my_print("\n--- POSTER GENERATION ---")
        start = time.time()
        poster_path = ""
        try:
            poster_result = await PosterService.generate_poster(video_id)
            my_print(f"Poster Generation took {time.time() - start:.2f}s")
            poster_path = poster_result.get("poster_path")
            my_print(f"Poster Path: {poster_path}")
            my_print(f"Poster Dimensions: {poster_result.get('width', 0)}x{poster_result.get('height', 0)}")
            my_print(f"Source Frame Selected: {poster_result.get('source_frame')}")
        except Exception as e:
             my_print(f"Poster Failed: {e}")
             
        my_print("\n--- SEARCH INDEXING VERIFICATION ---")
        video_points = []
        try:
            qm = QdrantManager()
            points, _ = qm.client.scroll(
                collection_name="video_events",
                scroll_filter=None,
                limit=500
            )
            video_points = [p for p in points if p.payload.get("video_id") == video_id]
            my_print(f"Search Indexed (Event points found for this video): {len(video_points)}")
        except Exception as e:
            my_print(f"Search indexing verification failed: {e}")
            
        my_print("\n--- FINAL RUNTIME SUMMARY METRICS ---")
        my_print(f"| Candidate Frames    | {processed_frames} |")
        my_print(f"| Metadata Success    | {success_frames} |")
        my_print(f"| Metadata Failed     | {failed_frames} |")
        my_print(f"| Events Generated    | {len(events)} |")
        my_print(f"| Incidents Generated | {len(incidents)} |")
        my_print(f"| Posters Generated   | {1 if poster_path else 0} |")
        my_print(f"| Search Indexed      | {len(video_points)} |")
        my_print(f"| Peak VRAM           | {vram_peak:.2f} GB |")
        my_print(f"| Backend Used        | native_hf |")
    except Exception as e:
        my_print(f"UNHANDLED EXCEPTION: {e}")
        import traceback
        my_print(traceback.format_exc())

if __name__ == "__main__":
    logger.add(f_out, format="{time} {level} {message}", filter="app", level="DEBUG")
    asyncio.run(run_audit())
    f_out.close()
