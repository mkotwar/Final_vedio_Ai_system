import asyncio
import json
import os
import sys
from pathlib import Path
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.core.config import settings
from app.services.vlm_factory import get_vlm_service

GT_DIR = Path(PROJECT_ROOT) / "validation" / "vlm" / "ground_truth"
FRAMES_DIR = Path(PROJECT_ROOT) / "validation" / "vlm" / "frames"
REPORTS_DIR = Path(PROJECT_ROOT) / "validation" / "vlm" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

async def run_vlm_validation():
    logger.info("Starting VLM Validation Audit")
    
    vlm_service = get_vlm_service()
    logger.info(f"Active VLM Service: {vlm_service.__name__}")
    
    gt_files = list(GT_DIR.glob("*.json"))
    if not gt_files:
        logger.error("No ground truth files found in validation/vlm/ground_truth/")
        return
        
    total_frames_analyzed = 0
    total_expected_objects = 0
    total_objects_found = 0
    total_expected_events = 0
    total_events_found = 0
    hallucination_incidents = 0
    actor_consistency_failures = 0
    total_latency_ms = 0.0
    schema_compliance_failures = 0
    schema_regurgitations = 0
    
    report_lines = []
    report_lines.append("# VLM OUTPUT VALIDATION AUDIT")
    report_lines.append(f"**VLM Engine**: {settings.VLM_ENGINE_TYPE}")
    report_lines.append(f"**Model**: {settings.QWEN_MODEL_ID if settings.VLM_ENGINE_TYPE == 'ollama' else 'Qwen/Qwen2.5-VL-7B-Instruct'}\n")
    
    for gt_file in gt_files:
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            
        video_id = gt_data["video_id"]
        frames = gt_data["frames"]
        
        report_lines.append(f"## Video: {video_id}")
        
        batch_frames = []
        for idx, frame_gt in enumerate(frames):
            ts = frame_gt["timestamp_seconds"]
            # Look for prefix matching video_id + "_ts" + int(ts) + ".jpg"
            # Our extraction script used prefixes: empy_room, person_walk, customer_int
            # Let's map video_id to prefix
            prefix_map = {
                "empy_room_15sec": "empy_room",
                "person_walking_30sec": "person_walk",
                "customer_interaticio_60sec": "customer_int"
            }
            prefix = prefix_map.get(video_id, video_id)
            frame_name = f"{prefix}_ts{int(ts)}.jpg"
            frame_path = FRAMES_DIR / frame_name
            
            if not frame_path.exists():
                logger.error(f"Missing frame image: {frame_path}")
                continue
                
            frame_id = f"{video_id}_f{idx}"
            batch_frames.append((frame_id, video_id, ts, frame_path))
            
        if not batch_frames:
            continue
            
        # Send to VLM
        try:
            results = await vlm_service.generate_metadata_batch(batch_frames)
        except Exception as e:
            logger.error(f"VLM batch generation failed for {video_id}: {e}")
            schema_compliance_failures += len(batch_frames)
            total_frames_analyzed += len(batch_frames)
            continue
            
        result_map = {res[0].frame_id: res for res in results}
        
        for idx, frame_gt in enumerate(frames):
            ts = frame_gt["timestamp_seconds"]
            frame_id = f"{video_id}_f{idx}"
            
            total_frames_analyzed += 1
            
            if frame_id not in result_map:
                schema_compliance_failures += 1
                report_lines.append(f"### Frame @ {ts}s")
                report_lines.append(f"* **Schema Compliance:** FAIL (Parsing/Validation Error)")
                report_lines.append("")
                continue
                
            rich_meta, timings = result_map[frame_id]
            
            vlm_ms = timings.get("vlm_ms", 0.0)
            total_latency_ms += vlm_ms
            
            meta_str = rich_meta.model_dump_json().lower()
            if any(p in meta_str for p in ["e.g.", "example", "unique id"]):
                schema_regurgitations += 1
            
            # Extract detected entities
            detected_obj_types = [obj.subtype.lower() if obj.subtype else obj.type.lower() for obj in rich_meta.objects]
            detected_obj_types.extend([obj.type.lower() for obj in rich_meta.objects])
            detected_obj_ids = [obj.id for obj in rich_meta.objects]
            
            detected_event_types = [evt.event_type.lower() for evt in rich_meta.events if evt.event_type and evt.event_type.lower() != "none"]
            
            # Metrics Calculation
            expected_objects = frame_gt["expected_objects"]
            objects_found = 0
            for obj in expected_objects:
                total_expected_objects += 1
                if any(obj.lower() in d_obj for d_obj in detected_obj_types):
                    objects_found += 1
                    total_objects_found += 1
                    
            expected_events = frame_gt["expected_events"]
            events_found = 0
            for evt in expected_events:
                total_expected_events += 1
                if any(evt.lower() in d_evt for d_evt in detected_event_types):
                    events_found += 1
                    total_events_found += 1
                    
            must_not_contain = frame_gt["must_not_contain_events"]
            hallucinated_this_frame = []
            for bad_evt in must_not_contain:
                if any(bad_evt.lower() in d_evt for d_evt in detected_event_types):
                    hallucinated_this_frame.append(bad_evt)
                    hallucination_incidents += 1
                    
            # Actor Consistency
            actors_valid = True
            for evt in rich_meta.events:
                for actor in evt.actors:
                    if actor not in detected_obj_ids:
                        actors_valid = False
                        actor_consistency_failures += 1
                        
            report_lines.append(f"### Frame @ {ts}s")
            report_lines.append(f"* **Latency:** {vlm_ms:.2f} ms")
            report_lines.append(f"* **Expected Objects:** {expected_objects} | Found: {objects_found}/{len(expected_objects)}")
            report_lines.append(f"* **Expected Events:** {expected_events} | Found: {events_found}/{len(expected_events)}")
            report_lines.append(f"* **Hallucinations Detected:** {hallucinated_this_frame if hallucinated_this_frame else 'None'} ({'FAIL' if hallucinated_this_frame else 'PASS'})")
            report_lines.append(f"* **Actor Consistency:** {'PASS' if actors_valid else 'FAIL'}")
            report_lines.append(f"* **Detected Objects Dump:** {detected_obj_types}")
            report_lines.append(f"* **Detected Events Dump:** {detected_event_types}")
            report_lines.append("")
            
    # Final Summary
    report_lines.append("## OVERALL METRICS")
    
    obj_recall = (total_objects_found / total_expected_objects * 100.0) if total_expected_objects > 0 else 100.0
    evt_recall = (total_events_found / total_expected_events * 100.0) if total_expected_events > 0 else 100.0
    hallucination_rate = (hallucination_incidents / total_frames_analyzed * 100.0) if total_frames_analyzed > 0 else 0.0
    actor_fail_rate = (actor_consistency_failures / total_frames_analyzed * 100.0) if total_frames_analyzed > 0 else 0.0
    
    compliance_rate = ((total_frames_analyzed - schema_compliance_failures) / total_frames_analyzed * 100.0) if total_frames_analyzed > 0 else 100.0
    regurgitation_rate = (schema_regurgitations / total_frames_analyzed * 100.0) if total_frames_analyzed > 0 else 0.0
    
    valid_frames = total_frames_analyzed - schema_compliance_failures
    avg_latency = total_latency_ms / valid_frames if valid_frames > 0 else 0.0
    
    report_lines.append(f"* **Total Frames Analyzed:** {total_frames_analyzed}")
    report_lines.append(f"* **Schema Compliance Rate:** {compliance_rate:.1f}%")
    report_lines.append(f"* **Schema Regurgitation Rate:** {regurgitation_rate:.1f}%")
    report_lines.append(f"* **Object Recall:** {obj_recall:.1f}%")
    report_lines.append(f"* **Event Recall:** {evt_recall:.1f}%")
    report_lines.append(f"* **Hallucination Rate (Frames with false events):** {hallucination_rate:.1f}%")
    report_lines.append(f"* **Actor Consistency Failure Rate:** {actor_fail_rate:.1f}%")
    report_lines.append(f"* **Average Latency:** {avg_latency:.2f} ms")
    
    final_report = "\n".join(report_lines)
    report_path = REPORTS_DIR / "vlm_validation_audit.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
        
    logger.info(f"Validation complete. Report saved to {report_path}")
    print(final_report)

if __name__ == "__main__":
    asyncio.run(run_vlm_validation())
