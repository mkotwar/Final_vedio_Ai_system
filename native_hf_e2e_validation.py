"""
PHASE DEMO-7C FIX-D — Native HF End-to-End Pipeline Validation
Standalone script. Runs the EXACT production code path in-process.
No Uvicorn needed. Uses the venv Python directly.

Usage: .\.venv\Scripts\python.exe native_hf_e2e_validation.py
"""
import os
import sys
import time
import json
import traceback

# Force project root on path
PROJECT_ROOT = r"c:\Mukul K\vinfo1\video-search-engine"
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Output file for evidence
REPORT_PATH = os.path.join(PROJECT_ROOT, "E2E_EVIDENCE.txt")

def log(msg):
    print(msg)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

def main():
    # Clear previous report
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
    
    log("=" * 70)
    log("PHASE DEMO-7C FIX-D: NATIVE HF END-TO-END PIPELINE VALIDATION")
    log("=" * 70)
    log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # ----------------------------------------------------------------
    # TASK 2: Verify Active Backend
    # ----------------------------------------------------------------
    log("--- TASK 2: VERIFY ACTIVE BACKEND ---")
    try:
        from app.core.config import settings
        log(f"settings.VLM_ENGINE_TYPE = '{settings.VLM_ENGINE_TYPE}'")
        log(f"settings.MOCK_MODEL = {settings.MOCK_MODEL}")
        log(f"settings.BATCH_SIZE = {settings.BATCH_SIZE}")
        log(f"settings.QWEN_MAX_NEW_TOKENS = {settings.QWEN_MAX_NEW_TOKENS}")
        
        from app.services.vlm_factory import get_vlm_service
        vlm_class = get_vlm_service()
        log(f"Factory returned class: {vlm_class.__name__}")
        log(f"Factory module: {vlm_class.__module__}")
        
        if vlm_class.__name__ != "NativeQwenTransformersService":
            log("FAIL: Factory did NOT return NativeQwenTransformersService!")
            return
        log("PASS: Native HF backend is correctly routed.")
    except Exception as e:
        log(f"FAIL: Backend verification error: {e}")
        log(traceback.format_exc())
        return
    
    log("")

    # ----------------------------------------------------------------
    # TASK 11: Native HF Model Loading + Tensor Shape Proof
    # ----------------------------------------------------------------
    log("--- TASK 11: NATIVE HF MODEL LOADING ---")
    try:
        import torch
        log(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log(f"GPU: {torch.cuda.get_device_name(0)}")
            log(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
            log(f"VRAM before model load: {torch.cuda.memory_allocated() / (1024**3):.3f} GB")
        
        from app.services.qwen_vlm_hf import NativeQwenTransformersService
        
        t0 = time.time()
        NativeQwenTransformersService.load_model()
        load_time = time.time() - t0
        log(f"Model loaded in {load_time:.2f}s")
        
        if torch.cuda.is_available():
            log(f"VRAM after model load: {torch.cuda.memory_allocated() / (1024**3):.3f} GB")
            log(f"VRAM reserved: {torch.cuda.memory_reserved() / (1024**3):.3f} GB")
        
        log(f"Model object exists: {NativeQwenTransformersService._model is not None}")
        log(f"Processor object exists: {NativeQwenTransformersService._processor is not None}")
        log("PASS: Model loaded successfully.")
    except Exception as e:
        log(f"FAIL: Model loading error: {e}")
        log(traceback.format_exc())
        return
    
    log("")

    # ----------------------------------------------------------------
    # TASK 1 & 4: Execute Real Ingestion (Full Pipeline)
    # ----------------------------------------------------------------
    log("--- TASK 1 & 4: EXECUTE REAL INGESTION ---")
    
    # Pick a valid video: must have BOTH .mp4 AND registry metadata JSON
    from app.services.video import VideoService
    
    videos_dir = settings.VIDEOS_DIR
    video_files = sorted(videos_dir.glob("*.mp4"), key=lambda p: p.stat().st_size)
    if not video_files:
        log("FAIL: No video files found in data/videos/")
        return
    
    video_id = None
    video_path = None
    
    log(f"Scanning {len(video_files)} video files for a valid candidate...")
    for candidate_path in video_files:
        candidate_id = candidate_path.stem
        try:
            VideoService.get_video(candidate_id)
            video_id = candidate_id
            video_path = candidate_path
            log(f"  VALID: {candidate_id} (registry OK, file OK)")
            break
        except Exception as e:
            log(f"  SKIP:  {candidate_id} ({type(e).__name__}: {e})")
            continue
    
    if video_id is None:
        log("FAIL: No video with a valid registry metadata found!")
        log("      Videos must be uploaded via the API to create a registry entry.")
        log("      Run: POST /videos/upload with a .mp4 file first.")
        return
    
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    log(f"Selected video: {video_path.name}")
    log(f"Video ID: {video_id}")
    log(f"File size: {video_size_mb:.2f} MB")
    log("")
    
    # Trace: File → Class → Function → Line
    log("--- TASK 1: RUNTIME PATH TRACE ---")
    log("1. Video Upload       → app/services/video.py → VideoService.save_video()")
    log("2. Frame Extraction   → app/services/frame.py → FrameExtractionService.extract_frames() [line 80]")
    log("3. Dynamic FPS        → app/services/frame.py → State Machine [lines 204-244]")
    log("4. Event Candidate    → app/services/frame.py → VNS accumulation [lines 273-302]")
    log("5. VLM Factory        → app/services/vlm_factory.py → get_vlm_service() [line 16]")
    log("6. Native HF Backend  → app/services/qwen_vlm_hf.py → NativeQwenTransformersService.generate_metadata_batch()")
    log("7. Metadata Gen       → app/services/qwen_vlm_hf.py → _generate_single() / batch processing")
    log("8. Event Aggregation  → app/services/event_aggregation.py → EventAggregationService.process_events() [line 505]")
    log("9. Poster Generation  → app/services/poster_service.py → PosterService.select_event_poster() [line 114]")
    log("10. Search Indexing   → app/services/search_service.py → SearchService.index_events() [line 65]")
    log("11. Incident Engine   → app/services/incident_engine.py → IncidentEngine.correlate_events() [called from search_service.py line 81]")
    log("")
    
    # Clean previous INGESTION artifacts (frames, events, catalogs) but PRESERVE registry metadata
    import shutil
    registry_file = settings.METADATA_DIR / f"{video_id}.json"  # MUST preserve this
    
    meta_dir = settings.METADATA_DIR / video_id
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    frames_dir_v = settings.FRAMES_DIR / video_id
    if frames_dir_v.exists():
        shutil.rmtree(frames_dir_v)
    events_dir = settings.DATA_DIR / "events" / video_id
    if events_dir.exists():
        shutil.rmtree(events_dir)
    thumb_dir_clean = settings.DATA_DIR / "thumbnails" / video_id
    if thumb_dir_clean.exists():
        shutil.rmtree(thumb_dir_clean)
    # Remove catalog/derivative files but NOT the registry JSON
    for p in settings.METADATA_DIR.glob(f"{video_id}*"):
        if p == registry_file:
            continue  # NEVER delete the registry
        p.unlink()
    
    log("Cleaned previous ingestion artifacts (preserved registry). Starting fresh.")
    log("")
    
    # Run the EXACT production path
    import asyncio
    from app.services.frame import FrameExtractionService
    from app.services.status_service import JobStatusService
    
    JobStatusService.initialize(video_id)
    
    log(">>> STARTING FRAME EXTRACTION + VLM ANALYSIS <<<")
    pipeline_start = time.time()
    
    try:
        stats = asyncio.run(FrameExtractionService.extract_frames(video_id))
        pipeline_duration = time.time() - pipeline_start
    except Exception as e:
        pipeline_duration = time.time() - pipeline_start
        log(f"FAIL: Pipeline crashed after {pipeline_duration:.2f}s")
        log(f"Exception: {e}")
        log(traceback.format_exc())
        
        # Task 5: Failure Analysis
        log("")
        log("--- TASK 5: FAILURE ANALYSIS ---")
        log(f"Exception type: {type(e).__name__}")
        log(f"Message: {str(e)}")
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            last = tb[-1]
            log(f"File: {last.filename}")
            log(f"Line: {last.lineno}")
            log(f"Function: {last.name}")
            log(f"Code: {last.line}")
        
        # Classify
        err_str = str(e).lower()
        if "cuda" in err_str or "vram" in err_str or "out of memory" in err_str:
            log("Classification: A. HF generation (CUDA/VRAM)")
        elif "ocr" in err_str:
            log("Classification: B. OCR")
        elif "json" in err_str or "parse" in err_str or "validation" in err_str:
            log("Classification: C. Metadata parsing")
        elif "event" in err_str or "aggregat" in err_str:
            log("Classification: D. Event aggregation")
        elif "poster" in err_str or "thumbnail" in err_str:
            log("Classification: E. Poster generation")
        elif "qdrant" in err_str or "index" in err_str:
            log("Classification: F. Search indexing")
        else:
            log("Classification: G. Other")
        return
    
    log(f"Pipeline completed in {pipeline_duration:.2f}s")
    log("")
    
    # ----------------------------------------------------------------
    # TASK 3: Instrumentation Evidence (from stats)
    # ----------------------------------------------------------------
    log("--- TASK 3: PIPELINE OUTPUT EVIDENCE ---")
    candidate_frames = stats.get("processed_frames", 0)
    metadata_success = stats.get("successful_frames", 0)
    metadata_failed = stats.get("failed_frames", 0)
    log(f"Candidate Frames:   {candidate_frames}")
    log(f"Metadata Success:   {metadata_success}")
    log(f"Metadata Failed:    {metadata_failed}")
    log("")
    
    # ----------------------------------------------------------------
    # TASK 6: Metadata Verification
    # ----------------------------------------------------------------
    log("--- TASK 6: METADATA VERIFICATION ---")
    frames = stats.get("frames", [])
    if frames:
        log(f"Total FrameRichMetadata objects: {len(frames)}")
        log("")
        log("FIRST metadata object:")
        f0 = frames[0]
        log(f"  frame_id: {f0.get('frame_id')}")
        log(f"  caption: {f0.get('caption', '')[:120]}")
        log(f"  scene_type: {f0.get('scene_type')}")
        log(f"  people_count: {f0.get('people_count')}")
        log(f"  objects: {len(f0.get('objects', []))}")
        log(f"  events: {len(f0.get('events', []))}")
        log(f"  activities: {f0.get('activities', [])}")
        log(f"  keywords: {f0.get('keywords', [])[:5]}")
        log(f"  search_text exists: {'Yes' if f0.get('search_text') else 'No'}")
        log(f"  ocr text: {f0.get('ocr', {}).get('text', 'N/A')[:80]}")
        
        if len(frames) > 1:
            log("")
            log("LAST metadata object:")
            fl = frames[-1]
            log(f"  frame_id: {fl.get('frame_id')}")
            log(f"  caption: {fl.get('caption', '')[:120]}")
            log(f"  scene_type: {fl.get('scene_type')}")
            log(f"  objects: {len(fl.get('objects', []))}")
        log("")
        log(f"PASS: metadata_success={metadata_success}, metadata_failed={metadata_failed}")
    else:
        log("FAIL: No FrameRichMetadata objects produced!")
    log("")
    
    # ----------------------------------------------------------------
    # TASK 7: Event Aggregation Verification
    # ----------------------------------------------------------------
    log("--- TASK 7: EVENT AGGREGATION VERIFICATION ---")
    events_file = settings.METADATA_DIR / f"{video_id}_events_v2.json"
    events_data = []
    if events_file.exists():
        with open(events_file, "r", encoding="utf-8") as f:
            events_data = json.load(f)
        log(f"Events file: {events_file}")
        log(f"Total AggregatedEvent objects: {len(events_data)}")
        for ev in events_data:
            log(f"  {ev.get('event_id')} | type={ev.get('event_type')} | severity={ev.get('event_severity')} | duration={ev.get('duration_seconds')}s")
        log("PASS: Event aggregation succeeded.") if events_data else log("FAIL: 0 events.")
    else:
        log(f"FAIL: Events file not found at {events_file}")
    log("")
    
    # ----------------------------------------------------------------
    # TASK 8: Incident Verification
    # ----------------------------------------------------------------
    log("--- TASK 8: INCIDENT VERIFICATION ---")
    try:
        from app.services.incident_engine import IncidentEngine
        from app.schemas.summary import AggregatedEvent
        if events_data:
            agg_events = [AggregatedEvent(**e) for e in events_data]
            chains = IncidentEngine.correlate_events(agg_events)
            log(f"IncidentChain objects: {len(chains)}")
            for ch in chains:
                log(f"  {ch.incident_id} | type={ch.incident_type} | severity={ch.severity}")
                log(f"    description: {ch.description[:100]}")
            log("PASS: Incident correlation succeeded.") if chains else log("INFO: 0 incidents (may be normal for simple scene).")
        else:
            log("SKIP: No events to correlate.")
    except Exception as e:
        log(f"FAIL: Incident correlation error: {e}")
        log(traceback.format_exc())
    log("")
    
    # ----------------------------------------------------------------
    # TASK 9: Poster Verification
    # ----------------------------------------------------------------
    log("--- TASK 9: POSTER VERIFICATION ---")
    thumb_dir = settings.DATA_DIR / "thumbnails" / video_id
    poster_files = list(thumb_dir.glob("*.jpg")) if thumb_dir.exists() else []
    log(f"Thumbnail directory: {thumb_dir}")
    log(f"Poster files found: {len(poster_files)}")
    for pf in poster_files:
        import cv2
        img = cv2.imread(str(pf))
        h, w = img.shape[:2] if img is not None else (0, 0)
        log(f"  {pf.name} | size={pf.stat().st_size} bytes | dimensions={w}x{h} | exists=True")
    if poster_files:
        log("PASS: Poster generation succeeded.")
    else:
        log("FAIL: No poster thumbnails found.")
    log("")
    
    # ----------------------------------------------------------------
    # TASK 10: Search (Qdrant) Verification
    # ----------------------------------------------------------------
    log("--- TASK 10: SEARCH INDEXING VERIFICATION ---")
    try:
        from app.services.search_service import SearchService
        client = SearchService.get_client()
        collection = settings.QDRANT_COLLECTION
        points, _ = client.scroll(collection_name=collection, limit=500)
        video_points = [p for p in points if p.payload.get("video_id") == video_id]
        log(f"Collection: {collection}")
        log(f"Total points in collection: {len(points)}")
        log(f"Points for this video: {len(video_points)}")
        if video_points:
            log(f"Embedding dimensions: {len(video_points[0].vector) if video_points[0].vector else 'N/A'}")
            log(f"Sample payload keys: {list(video_points[0].payload.keys())}")
        log("PASS: Search indexing succeeded.") if video_points else log("FAIL: No Qdrant points for this video.")
    except Exception as e:
        log(f"FAIL: Search verification error: {e}")
        log(traceback.format_exc())
    log("")
    
    # ----------------------------------------------------------------
    # TASK 11: VRAM Peak
    # ----------------------------------------------------------------
    log("--- TASK 11: VRAM PEAK ---")
    if torch.cuda.is_available():
        log(f"Peak VRAM allocated: {torch.cuda.max_memory_allocated() / (1024**3):.3f} GB")
        log(f"Current VRAM allocated: {torch.cuda.memory_allocated() / (1024**3):.3f} GB")
    log("")
    
    # ----------------------------------------------------------------
    # TASK 13: Final Summary Table
    # ----------------------------------------------------------------
    log("=" * 70)
    log("TASK 13: FINAL VALIDATION SUMMARY")
    log("=" * 70)
    log(f"| {'Metric':<25} | {'Value':<30} |")
    log(f"|{'-'*27}|{'-'*32}|")
    log(f"| {'Candidate Frames':<25} | {candidate_frames:<30} |")
    log(f"| {'Metadata Generated':<25} | {metadata_success:<30} |")
    log(f"| {'Metadata Failed':<25} | {metadata_failed:<30} |")
    log(f"| {'Events Generated':<25} | {len(events_data):<30} |")
    log(f"| {'Incidents Generated':<25} | {len(chains) if 'chains' in dir() else 'N/A':<30} |")
    log(f"| {'Posters Generated':<25} | {len(poster_files):<30} |")
    log(f"| {'Qdrant Points':<25} | {len(video_points) if 'video_points' in dir() else 'N/A':<30} |")
    log(f"| {'Runtime (seconds)':<25} | {pipeline_duration:.2f}{'':>24} |")
    log(f"| {'Backend':<25} | {'native_hf':<30} |")
    if torch.cuda.is_available():
        log(f"| {'Peak VRAM (GB)':<25} | {torch.cuda.max_memory_allocated() / (1024**3):.3f}{'':>25} |")
    log("")
    
    # ----------------------------------------------------------------
    # PASS / FAIL VERDICT
    # ----------------------------------------------------------------
    all_pass = (
        metadata_success > 0
        and len(events_data) > 0
        and len(poster_files) > 0
        and ('video_points' in dir() and len(video_points) > 0)
        and vlm_class.__name__ == "NativeQwenTransformersService"
    )
    
    log("=" * 70)
    if all_pass:
        log("VERDICT: *** PASS ***")
        log("All criteria met: metadata > 0, events > 0, posters exist,")
        log("Qdrant indexed, native_hf backend confirmed.")
    else:
        log("VERDICT: *** FAIL ***")
        reasons = []
        if metadata_success <= 0: reasons.append("metadata_success = 0")
        if len(events_data) <= 0: reasons.append("events = 0")
        if len(poster_files) <= 0: reasons.append("no posters")
        if 'video_points' not in dir() or len(video_points) <= 0: reasons.append("no Qdrant points")
        log(f"Failed checks: {', '.join(reasons)}")
    log("=" * 70)

if __name__ == "__main__":
    main()
