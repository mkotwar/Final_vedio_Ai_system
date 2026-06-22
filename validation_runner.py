import os
import sys
import traceback

# Force working directory to the script's folder
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

LOG_PATH = os.path.join(PROJECT_ROOT, "debug_run.txt")
BOUNDARY_TOLERANCE_SECONDS = 2.0

def debug_log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")

# Clear previous debug log
if os.path.exists(LOG_PATH):
    os.remove(LOG_PATH)

debug_log("Script started. Initializing imports...")

try:
    import json
    import time
    import shutil
    import asyncio
    from pathlib import Path
    import cv2

    from app.core.config import settings
    from app.services.frame import FrameExtractionService

    debug_log("Imports successful.")
except Exception as e:
    debug_log(f"FAILED DURING IMPORTS: {e}")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    sys.exit(1)

# Force mock model
settings.MOCK_MODEL = True

async def run_validation(custom_video_path=None):
    try:
        debug_log("Starting run_validation async loop...")
        
        # Directories
        val_dir = Path(PROJECT_ROOT) / "validation"
        gt_dir = val_dir / "ground_truth"
        videos_dir = val_dir / "videos"
        reports_dir = val_dir / "reports"

        val_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        videos_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        if custom_video_path:
            custom_path = Path(custom_video_path)
            if not custom_path.exists():
                debug_log(f"Error: Custom video path does not exist: {custom_video_path}")
                return
            
            dest_video_path = videos_dir / custom_path.name
            if not dest_video_path.exists() or dest_video_path.stat().st_size != custom_path.stat().st_size:
                debug_log(f"Copying {custom_path} to {dest_video_path}...")
                shutil.copy(custom_path, dest_video_path)
            
            import uuid
            video_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, custom_path.name))
            gt_file = gt_dir / f"{custom_path.stem}.json"
            
            if not gt_file.exists():
                cap = cv2.VideoCapture(str(dest_video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0.0
                cap.release()
                
                expected_base_frames = int(duration) + 1
                gt_data = {
                    "video_id": video_id,
                    "filename": custom_path.name,
                    "original_filename": custom_path.name,
                    "duration_seconds": round(duration, 2),
                    "fps": round(fps, 2),
                    "total_frames": total_frames,
                    "expected_base_frames": expected_base_frames,
                    "expected_events": [
                        {
                            "event_id": "evt_001",
                            "event_type": "motion",
                            "start_seconds": 0.0,
                            "end_seconds": round(duration, 1),
                            "description": "General motion event spanning the entire video."
                        }
                    ]
                }
                debug_log(f"Generating default ground truth for custom video: {gt_file}")
                with open(gt_file, "w", encoding="utf-8") as f:
                    json.dump(gt_data, f, indent=4)
            
            gt_files = [gt_file]
        else:
            gt_files = list(gt_dir.glob("*.json"))
            if not gt_files:
                debug_log("No ground truth files found in validation/ground_truth/")
                return

        for gt_path in gt_files:
            debug_log(f"Loading ground truth: {gt_path.name}")
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)

            video_id = gt["video_id"]
            filename = gt["filename"]
            original_filename = gt.get("original_filename", filename)
            expected_base_frames = gt["expected_base_frames"]
            expected_events = gt.get("expected_events", [])

            # Ensure video_id is a valid UUID
            import re
            import uuid
            UUID_REGEX = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
            if not UUID_REGEX.match(video_id):
                new_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
                debug_log(f"Correcting non-UUID video_id '{video_id}' to deterministic UUID '{new_uuid}'")
                video_id = new_uuid
                gt["video_id"] = video_id
                with open(gt_path, "w", encoding="utf-8") as f:
                    json.dump(gt, f, indent=4)

            video_src = videos_dir / filename
            if not video_src.exists():
                video_src = videos_dir / f"{video_id}.mp4"
                if not video_src.exists():
                    video_src = videos_dir / original_filename
                    if not video_src.exists():
                        debug_log(f"Error: Video file {filename} not found in validation/videos/")
                        continue

            debug_log(f"Processing Video: {original_filename} (ID: {video_id})")

            # SETUP: Mock upload
            app_video_path = settings.VIDEOS_DIR / f"{video_id}.mp4"
            app_meta_path = settings.METADATA_DIR / f"{video_id}.json"

            # Copy video
            shutil.copy(video_src, app_video_path)

            # Write app metadata
            app_meta = {
                "video_id": video_id,
                "filename": original_filename,
                "upload_time": "2026-06-20T12:00:00Z",
                "file_size": video_src.stat().st_size,
                "upload_duration_ms": 100.0
            }
            with open(app_meta_path, "w", encoding="utf-8") as f:
                json.dump(app_meta, f, indent=4)

            # Read actual video specs
            cap = cv2.VideoCapture(str(app_video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0.0
            cap.release()

            debug_log(f"Specs - Duration: {duration:.2f}s | FPS: {fps:.2f} | Total Frames: {total_frames}")

            # PASS 1: Base Frame Extraction (No Sampling)
            debug_log("Running Pass 1: Base Frame Extraction...")
            settings.ENABLE_ADAPTIVE_SAMPLING = False
            settings.ENABLE_MOTION_WINDOWING = False

            frame_dir = settings.FRAMES_DIR / video_id
            if frame_dir.exists():
                shutil.rmtree(frame_dir)

            start_time = time.perf_counter()
            stats_pass1 = await FrameExtractionService.extract_frames(video_id)
            pass1_duration = time.perf_counter() - start_time

            base_frames_count = len(stats_pass1.get("frames", []))
            debug_log(f"Pass 1 complete. Extracted frames: {base_frames_count}")

            # Validate Timestamps
            timestamp_errors = 0
            duplicate_timestamps = 0
            seen_timestamps = set()
            for idx, f in enumerate(stats_pass1.get("frames", [])):
                ts = f["timestamp_seconds"]
                frame_interval = max(1, int(round(fps)))
                expected_ts = (idx * frame_interval) / fps
                if abs(ts - expected_ts) > 0.01:
                    timestamp_errors += 1
                if ts in seen_timestamps:
                    duplicate_timestamps += 1
                seen_timestamps.add(ts)

            missing_frames = max(0, expected_base_frames - base_frames_count)
            pass1_status = "PASSED" if (missing_frames == 0 and timestamp_errors == 0) else "FAILED"

            # PASS 2: Dynamic FPS & Adaptive Sampling (Enabled)
            debug_log("Running Pass 2: Dynamic FPS & Adaptive Sampling...")
            settings.ENABLE_ADAPTIVE_SAMPLING = True
            settings.ENABLE_MOTION_WINDOWING = True

            if frame_dir.exists():
                shutil.rmtree(frame_dir)

            start_time = time.perf_counter()
            stats_pass2 = await FrameExtractionService.extract_frames(video_id)
            pass2_duration = time.perf_counter() - start_time

            retained_frames = stats_pass2.get("frames", [])
            retained_count = len(retained_frames)
            reduction_pct = stats_pass2.get("reduction_percent", 0.0)
            retention_pct = 100.0 - reduction_pct

            debug_log(f"Pass 2 complete. Retained frames: {retained_count} (Reduction: {reduction_pct:.2f}%)")

            # Event Metrics Verification
            retained_timestamps = sorted([f["timestamp_seconds"] for f in retained_frames])
            event_reports = []
            md_event_reports = []
            pass2_status = "PASSED"

            if not expected_events:
                fp_retention = len(retained_frames)
                fp_status = "PASS" if reduction_pct >= 80.0 else "FAIL"
                if fp_status == "FAIL": pass2_status = "FAILED"
                
                event_report = f"EMPTY SCENE VALIDATION\nOriginal Frames: {base_frames_count}\nRetained Frames: {fp_retention}\nReduction %: {reduction_pct:.2f}%\nStatus: {fp_status}\n"
                event_reports.append(event_report)
                
                md_event = f"### EMPTY SCENE VALIDATION\n* **Original Frames:** {base_frames_count}\n* **Retained Frames:** {fp_retention}\n* **Reduction %:** {reduction_pct:.2f}%\n* **Status:** **{fp_status}**\n"
                md_event_reports.append(md_event)
            else:
                for evt in expected_events:
                    evt_name = evt["event_type"]
                    evt_start = evt["start_seconds"]
                    evt_end = evt["end_seconds"]

                    orig_in_evt = len([f for f in stats_pass1.get("frames", []) if evt_start <= f["timestamp_seconds"] <= evt_end])
                    if orig_in_evt == 0:
                        orig_in_evt = int(evt_end - evt_start) + 1

                    retained_in_evt = [ts for ts in retained_timestamps if evt_start <= ts <= evt_end]
                    ret_count = len(retained_in_evt)
                    
                    event_recall = "PASS" if ret_count > 0 else "FAIL"
                    coverage_pct = (ret_count / orig_in_evt) * 100.0 if orig_in_evt > 0 else 0.0

                    if retained_in_evt:
                        start_dist = min(abs(ts - evt_start) for ts in retained_in_evt)
                        end_dist = min(abs(ts - evt_end) for ts in retained_in_evt)
                    else:
                        start_dist = float('inf')
                        end_dist = float('inf')
                        
                    start_boundary_pass = start_dist <= BOUNDARY_TOLERANCE_SECONDS
                    end_boundary_pass = end_dist <= BOUNDARY_TOLERANCE_SECONDS
                    
                    if ret_count == 0:
                        max_gap = evt_end - evt_start
                        max_gap_str = f"{max_gap:.1f}s"
                    elif ret_count == 1:
                        max_gap_str = "N/A"
                    else:
                        max_gap = max(retained_in_evt[i] - retained_in_evt[i-1] for i in range(1, ret_count))
                        max_gap_str = f"{max_gap:.1f}s"

                    if not start_boundary_pass or not end_boundary_pass or coverage_pct < 10.0 or event_recall == "FAIL":
                        pass2_status = "FAILED"

                    event_report = f"EVENT VALIDATION\nEvent: {evt_name}\nWindow: {evt_start}s - {evt_end}s\nOriginal Frames: {orig_in_evt}\nRetained Frames: {ret_count}\nEvent Recall: {event_recall}\nCoverage: {coverage_pct:.1f}%\nMax Temporal Gap: {max_gap_str}\nStart Boundary Distance: {start_dist:.1f}s\nStart Boundary Status: {'PASS' if start_boundary_pass else 'FAIL'}\nEnd Boundary Distance: {end_dist:.1f}s\nEnd Boundary Status: {'PASS' if end_boundary_pass else 'FAIL'}\n"
                    event_reports.append(event_report)
                    
                    md_event = f"### Event: {evt_name}\n* **Window:** {evt_start}s - {evt_end}s\n* **Original Frames:** {orig_in_evt}\n* **Retained Frames:** {ret_count}\n* **Event Recall:** {event_recall}\n* **Coverage:** {coverage_pct:.1f}%\n* **Max Temporal Gap:** {max_gap_str}\n* **Start Boundary Distance:** {start_dist:.1f}s\n* **Start Boundary Status:** {'PASS' if start_boundary_pass else 'FAIL'}\n* **End Boundary Distance:** {end_dist:.1f}s\n* **End Boundary Status:** {'PASS' if end_boundary_pass else 'FAIL'}\n"
                    md_event_reports.append(md_event)

            event_reports_str = "\n".join(event_reports)
            md_events_str = "\n".join(md_event_reports)

            # GENERATE REPORTS
            report_pass1 = f"""FRAME EXTRACTION REPORT

Video: {original_filename}
Duration: {duration:.2f} seconds
Expected Frames: {expected_base_frames}
Actual Frames: {base_frames_count}
Missing Frames: {missing_frames}
Duplicate Frames: {duplicate_timestamps}
Timestamp Accuracy: {100.0 - (timestamp_errors / max(1, base_frames_count)) * 100.0:.2f}%
Extraction Time: {pass1_duration:.4f} seconds
Status: {pass1_status}
"""

            report_pass2 = f"""DYNAMIC FPS REPORT

Video: {original_filename}
Original Frames: {base_frames_count}
Retained Frames: {retained_count}
Reduction %: {reduction_pct:.2f}%
Retention %: {retention_pct:.2f}%

{event_reports_str}Status: {pass2_status}
"""

            # Save reports
            report_pass1_path = reports_dir / f"{video_id}_frame_extraction_report.txt"
            with open(report_pass1_path, "w", encoding="utf-8") as f:
                f.write(report_pass1)

            report_pass2_path = reports_dir / f"{video_id}_dynamic_fps_report.txt"
            with open(report_pass2_path, "w", encoding="utf-8") as f:
                f.write(report_pass2)

            md_report = f"""# Validation Report for {original_filename}

## Frame Extraction Pass
* **Video:** `{original_filename}`
* **Duration:** {duration:.2f} seconds
* **Expected Frames:** {expected_base_frames}
* **Actual Frames:** {base_frames_count}
* **Missing Frames:** {missing_frames}
* **Duplicate Frames:** {duplicate_timestamps}
* **Timestamp Accuracy:** {100.0 - (timestamp_errors / max(1, base_frames_count)) * 100.0:.2f}%
* **Extraction Time:** {pass1_duration:.4f} seconds
* **Status:** **{pass1_status}**

---

## Dynamic FPS / Adaptive Sampling Pass
* **Original Frames:** {base_frames_count}
* **Retained Frames:** {retained_count}
* **Reduction %:** {reduction_pct:.2f}%
* **Retention %:** {retention_pct:.2f}%
{md_events_str}
* **Overall Status:** **{pass2_status}**
"""
            report_md_path = reports_dir / f"{video_id}_validation_report.md"
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(md_report)

            debug_log("Reports generated successfully.")

            # CLEANUP
            debug_log("Running database cleanups...")
            if app_video_path.exists():
                app_video_path.unlink()
            if app_meta_path.exists():
                app_meta_path.unlink()
            if frame_dir.exists():
                shutil.rmtree(frame_dir)
            catalog_path = settings.METADATA_DIR / f"{video_id}_frames.json"
            if catalog_path.exists():
                catalog_path.unlink()
            ind_meta_dir = settings.METADATA_DIR / video_id
            if ind_meta_dir.exists():
                shutil.rmtree(ind_meta_dir)
            events_dir = settings.EVENTS_DIR / video_id
            if events_dir.exists():
                shutil.rmtree(events_dir)
            status_file = settings.METADATA_DIR / f"{video_id}_status.json"
            if status_file.exists():
                status_file.unlink()

            for p in [Path(PROJECT_ROOT) / "PERFORMANCE_REPORT.md", Path(PROJECT_ROOT) / "ADAPTIVE_SAMPLING_REPORT.md", Path(PROJECT_ROOT) / "EVENT_AGGREGATION_REPORT.md", Path(PROJECT_ROOT) / "ADAPTIVE_SAMPLING_DIAGNOSTIC_REPORT.md"]:
                if p.exists():
                    p.unlink()

        debug_log("Validation complete!")

    except Exception as run_exc:
        debug_log(f"CRITICAL RUNTIME ERROR: {run_exc}")
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)

    print(retained_timestamps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Video Search Engine Validation")
    parser.add_argument("--video", type=str, help="Path to a custom video file to test")
    args = parser.parse_args()
    
    asyncio.run(run_validation(custom_video_path=args.video))


