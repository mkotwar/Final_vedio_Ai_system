"""
VLM Performance Regression — Root Cause Diagnostic Script
==========================================================
This script measures everything needed for an evidence-based diagnosis.
It does NOT modify any code — read-only diagnostics only.
"""

import json
import os
import subprocess
import sys
import time
import base64
import io
from pathlib import Path

# ─── SECTION 0: Utilities ─────────────────────────────────────────────

def section_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def run_cmd(cmd, shell=True):
    """Run a shell command and return stdout."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=30)
        return result.stdout.strip()
    except Exception as e:
        return f"ERROR: {e}"

# ─── SECTION 1: GPU Environment ───────────────────────────────────────

def audit_gpu():
    section_header("PART 1A: GPU ENVIRONMENT")

    # nvidia-smi full output
    smi_output = run_cmd("nvidia-smi")
    print(smi_output)

    print("\n--- Structured GPU Query ---")
    csv_output = run_cmd(
        'nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.draw,power.limit --format=csv'
    )
    print(csv_output)

    # CUDA version
    cuda_ver = run_cmd("nvidia-smi --query-gpu=cuda_version --format=csv,noheader")
    print(f"\nCUDA Version: {cuda_ver}")

    # Check if torch sees GPU
    try:
        import torch
        print(f"\nPyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            total_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"PyTorch reported VRAM: {total_mem:.2f} GB")
    except ImportError:
        print("PyTorch not available in this environment")

# ─── SECTION 1B: Ollama Environment ───────────────────────────────────

def audit_ollama():
    section_header("PART 1B: OLLAMA ENVIRONMENT")

    version = run_cmd("ollama --version")
    print(f"Ollama Version: {version}")

    models = run_cmd("ollama list")
    print(f"\nInstalled Models:\n{models}")

    # Get currently running models
    ps_output = run_cmd("ollama ps")
    print(f"\nRunning Models:\n{ps_output}")

    # Try to get model info for configured model
    from app.core.config import settings
    model_id = settings.QWEN_MODEL_ID
    print(f"\nConfigured Model: {model_id}")

    show_output = run_cmd(f"ollama show {model_id}")
    print(f"\nModel Details:\n{show_output}")

    # Check Ollama environment variables
    print("\n--- Ollama Environment Variables ---")
    for var in ["OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_ORIGINS",
                "OLLAMA_HOST", "OLLAMA_MODELS", "OLLAMA_GPU_OVERHEAD",
                "OLLAMA_MAX_QUEUE", "OLLAMA_NUM_GPU", "OLLAMA_FLASH_ATTENTION",
                "OLLAMA_KV_CACHE_TYPE", "OLLAMA_CONTEXT_LENGTH"]:
        val = os.environ.get(var, "NOT SET")
        print(f"  {var}: {val}")

# ─── SECTION 1C: VLM Configuration ───────────────────────────────────

def audit_vlm_config():
    section_header("PART 1C: VLM CONFIGURATION (.env)")

    from app.core.config import settings
    print(f"QWEN_MODEL_ID:       {settings.QWEN_MODEL_ID}")
    print(f"BATCH_SIZE:          {settings.BATCH_SIZE}")
    print(f"QWEN_MAX_NEW_TOKENS: {settings.QWEN_MAX_NEW_TOKENS}")
    print(f"MOCK_MODEL:          {settings.MOCK_MODEL}")
    print(f"ENABLE_ADAPTIVE_SAMPLING: {settings.ENABLE_ADAPTIVE_SAMPLING}")
    print(f"ENABLE_MOTION_WINDOWING:  {settings.ENABLE_MOTION_WINDOWING}")

# ─── SECTION 2: Single Frame Timing Breakdown ─────────────────────────

def benchmark_single_frame():
    section_header("PART 2: SINGLE FRAME TIMING BREAKDOWN")

    import cv2
    import httpx
    from app.core.config import settings

    # Find a test video and extract one frame
    videos_dir = settings.VIDEOS_DIR
    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print("ERROR: No video files found for benchmarking!")
        return {}

    # Pick a small video for test
    video_file = None
    for vf in video_files:
        if vf.stat().st_size < 10_000_000:  # Under 10MB
            video_file = vf
            break
    if not video_file:
        video_file = video_files[0]

    print(f"Test video: {video_file.name} ({video_file.stat().st_size / 1024 / 1024:.1f} MB)")

    # === Step 1: Extract a frame ===
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(video_file))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)  # Frame 30 (1 second in)
    success, frame = cap.read()
    cap.release()
    t_extract = (time.perf_counter() - t0) * 1000

    if not success:
        print("ERROR: Could not extract frame!")
        return {}

    original_h, original_w = frame.shape[:2]
    print(f"\n--- Image Pipeline ---")
    print(f"Original Resolution:   {original_w}x{original_h}")

    # === Step 2: Resize (matching qwen_vlm.py logic) ===
    t0 = time.perf_counter()
    max_dim = 800
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = frame
        new_w, new_h = w, h
    t_resize = (time.perf_counter() - t0) * 1000
    print(f"Resized Resolution:    {new_w}x{new_h}")

    # === Step 3: JPEG Compression ===
    t0 = time.perf_counter()
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
    _, jpeg_buf = cv2.imencode(".jpg", resized, encode_params)
    jpeg_bytes = jpeg_buf.tobytes()
    t_compress = (time.perf_counter() - t0) * 1000
    jpeg_size_kb = len(jpeg_bytes) / 1024
    print(f"JPEG Size:             {jpeg_size_kb:.1f} KB (quality=75)")

    # === Step 4: Base64 Encoding ===
    t0 = time.perf_counter()
    b64_str = base64.b64encode(jpeg_bytes).decode("utf-8")
    t_b64 = (time.perf_counter() - t0) * 1000
    b64_size_kb = len(b64_str) / 1024
    print(f"Base64 Payload Size:   {b64_size_kb:.1f} KB")

    # === Step 5: Prompt Construction ===
    t0 = time.perf_counter()
    prompt_text = """You are a forensic video analyst AI. Analyze this CCTV frame in strict JSON format.
Return ONLY valid JSON with these fields:
{
  "scene_type": "string",
  "scene_description": "string",
  "objects": [{"id": "string", "type": "string", "subtype": "string", "color": "string", "condition": "string", "attributes": ["string"]}],
  "events": [{"event_type": "string", "description": "string", "actors": ["string"], "severity": "string"}],
  "people_count": 0,
  "activities": ["string"],
  "keywords": ["string"],
  "caption": "string"
}"""

    payload = {
        "model": settings.QWEN_MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": prompt_text,
                "images": [b64_str],
            }
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_predict": settings.QWEN_MAX_NEW_TOKENS,
        },
    }
    payload_json = json.dumps(payload)
    t_prompt = (time.perf_counter() - t0) * 1000
    payload_size_mb = len(payload_json) / (1024 * 1024)
    print(f"Full HTTP Payload:     {payload_size_mb:.2f} MB")

    # === Step 6: Ollama HTTP Request (the actual inference) ===
    print(f"\n--- VLM Inference ---")
    print(f"Sending to Ollama ({settings.QWEN_MODEL_ID})...")

    t0_http = time.perf_counter()
    try:
        with httpx.Client(timeout=300.0) as client:
            t0_send = time.perf_counter()
            response = client.post(
                "http://localhost:11434/api/chat",
                content=payload_json,
                headers={"Content-Type": "application/json"},
            )
            t_total_http = (time.perf_counter() - t0_send) * 1000

        response_data = response.json()

        # Extract timing from Ollama response
        prompt_eval_count = response_data.get("prompt_eval_count", 0)
        eval_count = response_data.get("eval_count", 0)
        prompt_eval_duration = response_data.get("prompt_eval_duration", 0) / 1e6  # ns -> ms
        eval_duration = response_data.get("eval_duration", 0) / 1e6  # ns -> ms
        total_duration = response_data.get("total_duration", 0) / 1e6  # ns -> ms
        load_duration = response_data.get("load_duration", 0) / 1e6  # ns -> ms

        content = response_data.get("message", {}).get("content", "")
        content_len = len(content)

        print(f"\nOllama Response Timings:")
        print(f"  Model Load:          {load_duration:.1f} ms")
        print(f"  Prompt Eval:         {prompt_eval_duration:.1f} ms  ({prompt_eval_count} tokens)")
        print(f"  Generation:          {eval_duration:.1f} ms  ({eval_count} tokens)")
        print(f"  Total (Ollama):      {total_duration:.1f} ms")
        print(f"  Total (HTTP round):  {t_total_http:.1f} ms")

        if prompt_eval_count > 0:
            prompt_tok_per_sec = prompt_eval_count / (prompt_eval_duration / 1000.0) if prompt_eval_duration > 0 else 0
            print(f"  Prompt Speed:        {prompt_tok_per_sec:.1f} tokens/sec")
        if eval_count > 0:
            gen_tok_per_sec = eval_count / (eval_duration / 1000.0) if eval_duration > 0 else 0
            print(f"  Generation Speed:    {gen_tok_per_sec:.1f} tokens/sec")

        print(f"\n  Response Length:      {content_len} chars")
        print(f"  Prompt Tokens:       {prompt_eval_count}")
        print(f"  Output Tokens:       {eval_count}")

        # Queue wait = total_http - total_ollama_reported
        queue_wait = t_total_http - total_duration
        if queue_wait < 0:
            queue_wait = 0

    except Exception as e:
        print(f"ERROR during Ollama call: {e}")
        t_total_http = 0
        queue_wait = 0
        total_duration = 0
        prompt_eval_duration = 0
        eval_duration = 0
        load_duration = 0
        content = ""

    # === Step 7: OCR Timing ===
    print(f"\n--- OCR Pipeline ---")
    # Save frame to temp file for OCR
    temp_frame_path = Path("diagnostic_temp_frame.jpg")
    cv2.imwrite(str(temp_frame_path), frame)

    t0 = time.perf_counter()
    try:
        from app.services.ocr import OCRService
        ocr_result = OCRService.extract_text(str(temp_frame_path))
        t_ocr = (time.perf_counter() - t0) * 1000
        print(f"OCR Time:              {t_ocr:.1f} ms")
        print(f"OCR Text Found:        {len(ocr_result.get('detected_text', []))} items")
        print(f"License Plates Found:  {len(ocr_result.get('license_plates', []))} items")
    except Exception as e:
        t_ocr = 0
        print(f"OCR Error: {e}")

    # Cleanup temp file
    if temp_frame_path.exists():
        temp_frame_path.unlink()

    # === Step 8: JSON Repair Timing ===
    print(f"\n--- JSON Repair ---")
    t0 = time.perf_counter()
    try:
        import json_repair
        repaired = json_repair.loads(content if content else '{}')
        t_json_repair = (time.perf_counter() - t0) * 1000
        print(f"JSON Repair Time:      {t_json_repair:.1f} ms")
    except Exception as e:
        t_json_repair = 0
        print(f"JSON Repair Error: {e}")

    # === Step 9: Pydantic Validation Timing ===
    print(f"\n--- Pydantic Validation ---")
    t0 = time.perf_counter()
    try:
        from app.schemas.frame import FrameRichMetadata
        # We can't fully validate without all fields, just time the schema construction
        t_validation = (time.perf_counter() - t0) * 1000
        print(f"Validation Time:       {t_validation:.2f} ms")
    except Exception as e:
        t_validation = 0
        print(f"Validation Error: {e}")

    # === SUMMARY ===
    section_header("FRAME TIMING SUMMARY")

    timings = {
        "frame_extract_ms": round(t_extract, 1),
        "resize_ms": round(t_resize, 1),
        "jpeg_compress_ms": round(t_compress, 1),
        "base64_encode_ms": round(t_b64, 1),
        "prompt_construction_ms": round(t_prompt, 1),
        "model_load_ms": round(load_duration, 1),
        "prompt_eval_ms": round(prompt_eval_duration, 1),
        "generation_ms": round(eval_duration, 1),
        "ollama_total_ms": round(total_duration, 1),
        "http_roundtrip_ms": round(t_total_http, 1),
        "queue_wait_ms": round(queue_wait, 1),
        "ocr_ms": round(t_ocr, 1),
        "json_repair_ms": round(t_json_repair, 1),
        "validation_ms": round(t_validation, 2),
    }

    grand_total = sum(timings.values()) - timings["ollama_total_ms"]  # avoid double-counting

    print(f"Frame Extract:         {timings['frame_extract_ms']:>10.1f} ms")
    print(f"Resize:                {timings['resize_ms']:>10.1f} ms")
    print(f"JPEG Compress:         {timings['jpeg_compress_ms']:>10.1f} ms")
    print(f"Base64 Encode:         {timings['base64_encode_ms']:>10.1f} ms")
    print(f"Prompt Construction:   {timings['prompt_construction_ms']:>10.1f} ms")
    print(f"─── Ollama Breakdown ───")
    print(f"  Model Load:          {timings['model_load_ms']:>10.1f} ms")
    print(f"  Prompt Eval:         {timings['prompt_eval_ms']:>10.1f} ms")
    print(f"  Generation:          {timings['generation_ms']:>10.1f} ms")
    print(f"  Queue Wait:          {timings['queue_wait_ms']:>10.1f} ms")
    print(f"HTTP Roundtrip Total:  {timings['http_roundtrip_ms']:>10.1f} ms")
    print(f"OCR:                   {timings['ocr_ms']:>10.1f} ms")
    print(f"JSON Repair:           {timings['json_repair_ms']:>10.1f} ms")
    print(f"Validation:            {timings['validation_ms']:>10.2f} ms")
    print(f"{'─'*45}")
    print(f"GRAND TOTAL:           {grand_total:>10.1f} ms ({grand_total/1000:.1f}s)")

    # Percentage breakdown
    if grand_total > 0:
        print(f"\n--- Percentage Breakdown ---")
        for k, v in sorted(timings.items(), key=lambda x: x[1], reverse=True):
            if k != "ollama_total_ms":
                pct = (v / grand_total) * 100
                print(f"  {k:<25s}: {pct:>6.2f}%")

    return timings

# ─── SECTION 3: GPU Utilization During Inference ──────────────────────

def audit_gpu_during_inference():
    section_header("PART 3: GPU UTILIZATION (snapshot)")

    # Take a snapshot right now
    output = run_cmd("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.current.graphics --format=csv")
    print(output)

    # Also check processes using GPU
    print("\n--- GPU Process List ---")
    proc_output = run_cmd("nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv")
    print(proc_output)

# ─── SECTION 5: Image Pipeline Audit ──────────────────────────────────

def audit_image_pipeline():
    section_header("PART 5: IMAGE PIPELINE AUDIT")

    import cv2
    from app.core.config import settings

    videos_dir = settings.VIDEOS_DIR
    video_files = sorted(videos_dir.glob("*.mp4"))

    # Check a few videos for resolution diversity
    print(f"Total videos on disk: {len(video_files)}\n")

    print(f"{'Video ID':<40s} {'Resolution':<15s} {'FPS':<8s} {'Frames':<8s} {'Size MB':<10s}")
    print("─" * 85)

    resolutions = []
    for vf in video_files[:10]:  # Sample first 10
        cap = cv2.VideoCapture(str(vf))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        size_mb = vf.stat().st_size / (1024 * 1024)
        resolutions.append((w, h))
        print(f"{vf.stem:<40s} {w}x{h:<10d} {fps:<8.1f} {total:<8d} {size_mb:<10.1f}")

    # What the VLM receives
    print(f"\n--- VLM Input Analysis ---")
    print(f"Max resize dimension (from qwen_vlm.py): 800px")
    for w, h in resolutions[:3]:
        if max(w, h) > 800:
            scale = 800 / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            print(f"  {w}x{h} → {new_w}x{new_h} (scale={scale:.2f})")
        else:
            print(f"  {w}x{h} → {w}x{h} (no resize needed)")

# ─── SECTION 6: Throughput Calculation ────────────────────────────────

def throughput_projection(avg_frame_ms):
    section_header("PART 6: THROUGHPUT PROJECTIONS")

    avg_frame_sec = avg_frame_ms / 1000.0
    frames_per_min = 60.0 / avg_frame_sec if avg_frame_sec > 0 else 0

    print(f"Average Time Per Frame:     {avg_frame_sec:.1f} seconds")
    print(f"Effective Frames/Minute:    {frames_per_min:.1f}")
    print(f"")

    scenarios = [
        ("10 second video (1fps)", 10),
        ("1 minute video (1fps)", 60),
        ("10 minute video (1fps)", 600),
        ("1 hour video (1fps)", 3600),
        ("8 hour CCTV (1fps)", 28800),
    ]

    print(f"{'Scenario':<30s} {'Frames':<10s} {'Process Time':<20s} {'Realtime Ratio':<15s}")
    print("─" * 80)

    for name, frames in scenarios:
        total_sec = frames * avg_frame_sec
        hours = total_sec / 3600
        mins = total_sec / 60
        if hours >= 1:
            time_str = f"{hours:.1f} hours"
        else:
            time_str = f"{mins:.1f} minutes"
        ratio = total_sec / frames if frames > 0 else 0
        print(f"{name:<30s} {frames:<10d} {time_str:<20s} {ratio:.1f}x realtime")

    print(f"\n--- With 50% Adaptive Sampling ---")
    for name, frames in scenarios:
        effective_frames = frames // 2
        total_sec = effective_frames * avg_frame_sec
        hours = total_sec / 3600
        mins = total_sec / 60
        if hours >= 1:
            time_str = f"{hours:.1f} hours"
        else:
            time_str = f"{mins:.1f} minutes"
        print(f"{name:<30s} {effective_frames:<10d} {time_str:<20s}")

# ─── SECTION 4: Ollama Configuration Audit ────────────────────────────

def audit_ollama_config():
    section_header("PART 4: OLLAMA CONFIGURATION AUDIT")

    from app.core.config import settings

    # Check Ollama API for model config
    try:
        import httpx
        resp = httpx.post("http://localhost:11434/api/show", json={"name": settings.QWEN_MODEL_ID}, timeout=10)
        data = resp.json()

        print("--- Model Parameters ---")
        params = data.get("parameters", "")
        if params:
            print(params)
        else:
            print("(no parameters reported)")

        print("\n--- Model Template ---")
        template = data.get("template", "")
        print(template[:500] if template else "(no template)")

        print("\n--- Model Details ---")
        details = data.get("details", {})
        for k, v in details.items():
            print(f"  {k}: {v}")

        # Check modelfile for quantization info
        modelfile = data.get("modelfile", "")
        if "Q4" in modelfile or "q4" in modelfile:
            print("\n  Quantization: Q4 (4-bit)")
        elif "Q5" in modelfile or "q5" in modelfile:
            print("\n  Quantization: Q5 (5-bit)")
        elif "Q8" in modelfile or "q8" in modelfile:
            print("\n  Quantization: Q8 (8-bit)")
        elif "F16" in modelfile or "fp16" in modelfile.lower():
            print("\n  Quantization: FP16 (16-bit)")
        else:
            print(f"\n  Quantization: Could not determine from modelfile")

        # Model size
        model_info = data.get("model_info", {})
        if model_info:
            print("\n--- Model Architecture Info ---")
            for k, v in sorted(model_info.items()):
                print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error querying Ollama API: {e}")

    # Check running model state
    print("\n--- Running Model State ---")
    ps = run_cmd("ollama ps")
    print(ps)

# ─── MAIN ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  VLM PERFORMANCE REGRESSION — ROOT CAUSE ANALYSIS")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Part 1
    audit_gpu()
    audit_ollama()
    audit_vlm_config()
    audit_ollama_config()

    # Part 5 (before the long benchmark)
    audit_image_pipeline()

    # Part 3 (GPU state before inference)
    print("\n>>> GPU state BEFORE inference:")
    audit_gpu_during_inference()

    # Part 2 (the main benchmark — this takes time)
    timings = benchmark_single_frame()

    # Part 3 (GPU state during/after inference)
    print("\n>>> GPU state AFTER inference:")
    audit_gpu_during_inference()

    # Part 6
    if timings:
        avg_frame = timings.get("http_roundtrip_ms", 0) + timings.get("ocr_ms", 0)
        throughput_projection(avg_frame)

    section_header("DIAGNOSTIC COMPLETE")
    print("All measurements collected. No code was modified.")
