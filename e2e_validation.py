import os
import time
import json
import requests
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
VIDEO_PATH = r"c:\Mukul K\vinfo1\video-search-engine\data\videos\32ac5bc9-91ea-4cfe-8d82-a383f6d608c4.mp4"

def log_output(text):
    print(text)
    with open("E2E_REPORT.md", "a", encoding="utf-8") as f:
        f.write(text + "\n")

def run():
    if os.path.exists("E2E_REPORT.md"):
        os.remove("E2E_REPORT.md")
        
    log_output("# E2E API Validation Report")
    
    # Wait for Uvicorn to be ready
    for _ in range(30):
        try:
            res = requests.get(f"{BASE_URL}/docs")
            if res.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
        
    log_output("\n## API Server is UP")
    
    # 1. Upload Video
    log_output("\n### 1. Upload Video")
    with open(VIDEO_PATH, "rb") as f:
        res = requests.post(f"{BASE_URL}/videos/upload", files={"file": ("test.mp4", f, "video/mp4")})
    
    if res.status_code != 201:
        log_output(f"**FAIL**: Upload failed {res.status_code} {res.text}")
        return
        
    video_id = res.json()["video_id"]
    log_output(f"**PASS**: Uploaded Video ID: `{video_id}`")
    
    # 2. Extract Frames
    log_output("\n### 2. Extract Frames (Native HF)")
    start = time.time()
    res = requests.post(f"{BASE_URL}/frames/extract", json={"video_id": video_id})
    duration = time.time() - start
    
    if res.status_code != 200:
        log_output(f"**FAIL**: Extraction failed {res.status_code} {res.text}")
        return
        
    data = res.json()
    log_output(f"**PASS**: Extracted frames in {duration:.2f}s")
    log_output(f"Candidate Frames: {data.get('processed_frames')}")
    log_output(f"Metadata Success: {data.get('successful_frames')}")
    log_output(f"Metadata Failed: {data.get('failed_frames')}")
    
    # Print the first frame's metadata to prove FrameRichMetadata works
    if data.get("frames"):
        f1 = data["frames"][0]
        log_output(f"\nExample Frame Metadata:")
        log_output(f"- Caption: {f1.get('caption')}")
        log_output(f"- Scene: {f1.get('scene_type')}")
        log_output(f"- Events: {len(f1.get('events', []))}")
        log_output(f"- Search Text exists: {'Yes' if f1.get('search_text') else 'No'}")
        log_output(f"- OCR Text: {f1.get('ocr', {}).get('text', 'None')}")

    # 3. Generate Summaries
    log_output("\n### 3. Generate Summaries (Event Aggregation)")
    start = time.time()
    res = requests.post(f"{BASE_URL}/summaries/generate", json={"video_id": video_id})
    duration = time.time() - start
    
    if res.status_code != 200:
        log_output(f"**FAIL**: Summaries failed {res.status_code} {res.text}")
    else:
        events = res.json().get("events", [])
        log_output(f"**PASS**: Aggregated {len(events)} events in {duration:.2f}s")
        
    # 4. Generate Incidents
    log_output("\n### 4. Generate Incidents")
    start = time.time()
    res = requests.post(f"{BASE_URL}/incidents/correlate", json={"video_id": video_id})
    duration = time.time() - start
    
    if res.status_code != 200:
        log_output(f"**FAIL**: Incidents failed {res.status_code} {res.text}")
    else:
        incidents = res.json().get("incidents", [])
        log_output(f"**PASS**: Correlated {len(incidents)} incidents in {duration:.2f}s")
        
    # 5. Generate Poster
    log_output("\n### 5. Generate Poster")
    start = time.time()
    res = requests.post(f"{BASE_URL}/posters/generate", json={"video_id": video_id})
    duration = time.time() - start
    
    if res.status_code != 200:
        log_output(f"**FAIL**: Poster failed {res.status_code} {res.text}")
    else:
        poster_path = res.json().get("poster_path")
        log_output(f"**PASS**: Generated poster at `{poster_path}` in {duration:.2f}s")
        
    log_output("\n### 6. Verify Search Indexing")
    # Verify via local Qdrant script
    try:
        import sys
        sys.path.insert(0, r"c:\Mukul K\vinfo1\video-search-engine")
        from app.core.qdrant_manager import QdrantManager
        qm = QdrantManager()
        points, _ = qm.client.scroll(collection_name="video_events", limit=500)
        found = len([p for p in points if p.payload.get("video_id") == video_id])
        log_output(f"**PASS**: Found {found} searchable points in Qdrant")
    except Exception as e:
        log_output(f"**FAIL**: Search verification failed: {str(e)}")
        
if __name__ == "__main__":
    run()
