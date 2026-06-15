import asyncio
import time
import json
import shutil
import traceback
from pathlib import Path
import cv2

from app.core.config import settings
from app.services.video import VideoService
from app.services.frame import FrameExtractionService
from app.services.qwen_vlm import QwenVLMService
from app.services.event_aggregation import EventAggregationService
from app.services.search_service import SearchService
from app.services.summary_service import SummaryService

async def main():
    try:
        video_id = "05715892-0b34-4fcb-9f62-608caeaadd42"
        metadata, video_path = VideoService.get_video(video_id)
        
        timings = {}
        
        start = time.perf_counter()
        dummy_path = video_path.with_name("dummy_upload.mp4")
        shutil.copy(video_path, dummy_path)
        dummy_path.unlink()
        timings["Video upload time"] = time.perf_counter() - start
        
        start = time.perf_counter()
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        frame_interval = max(1, int(round(fps)))
        
        extracted_tuples = []
        last_sent_frame = None
        frame_idx = 1
        current_raw_frame = 0
        total_extracted = 0
        
        frame_dir = settings.FRAMES_DIR / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        extract_time = 0.0
        sampling_time = 0.0
        
        while True:
            if current_raw_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_raw_frame)
            t0 = time.perf_counter()
            success, frame = cap.read()
            extract_time += (time.perf_counter() - t0)
            if not success: break
            
            second = current_raw_frame / fps
            total_extracted += 1
            frame_id = f"{video_id}_f{frame_idx:04d}"
            
            should_send = True
            t0 = time.perf_counter()
            if settings.ENABLE_ADAPTIVE_SAMPLING and last_sent_frame is not None:
                hist_diff, ssim_diff, motion_score = FrameExtractionService.compute_similarity_metrics(frame, last_sent_frame)
                ssim_score = 1.0 - ssim_diff
                is_scene_change = (hist_diff > settings.HISTOGRAM_THRESHOLD) or (ssim_score < settings.SSIM_THRESHOLD)
                is_motion = motion_score > settings.MOTION_THRESHOLD
                if not (is_scene_change or is_motion):
                    should_send = False
            sampling_time += (time.perf_counter() - t0)
            
            if should_send:
                t0 = time.perf_counter()
                out_path = frame_dir / f"frame_{frame_idx:04d}.jpg"
                cv2.imwrite(str(out_path), frame)
                extracted_tuples.append((frame_id, video_id, second, out_path))
                last_sent_frame = frame.copy()
                extract_time += (time.perf_counter() - t0)
                
            frame_idx += 1
            current_raw_frame += frame_interval
            
        cap.release()
        timings["Frame extraction time"] = extract_time
        timings["Adaptive sampling time"] = sampling_time
        
        start = time.perf_counter()
        rich_frames = []
        batch_size = settings.BATCH_SIZE
        processed_count = len(extracted_tuples)
        for i in range(0, processed_count, batch_size):
            batch = extracted_tuples[i : i + batch_size]
            batch_results = await QwenVLMService.generate_metadata_batch(batch)
            for rich_meta, t in batch_results:
                rich_frames.append(rich_meta.model_dump())
        timings["OCR/VLM metadata generation time"] = time.perf_counter() - start
        
        start = time.perf_counter()
        events = EventAggregationService.process_events(video_id, rich_frames)
        timings["Event aggregation time"] = time.perf_counter() - start
        
        from app.services.incident_engine import IncidentEngine
        start = time.perf_counter()
        incidents = IncidentEngine.correlate_events(events)
        timings["Incident correlation time"] = time.perf_counter() - start
        
        from app.services.narrative_builder import NarrativeBuilderService
        start = time.perf_counter()
        timeline_text = NarrativeBuilderService._format_events_for_prompt(events)
        report_data = NarrativeBuilderService.generate_investigation_report(timeline_text)
        timings["Narrative generation time"] = time.perf_counter() - start
        
        start = time.perf_counter()
        try:
            SearchService.index_events(video_id, events)
        except Exception as search_exc:
            pass
        timings["Search indexing time"] = time.perf_counter() - start
        
        total_time = sum(timings.values())
        sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
        
        report = "# PIPELINE PROFILING AUDIT REPORT\n\n"
        report += f"**Video ID:** {video_id}\n"
        report += f"**Total Pipeline Time:** {total_time:.3f} seconds\n\n"
        
        report += "| Stage Name | Duration (s) | % of Runtime |\n"
        report += "| :--- | :--- | :--- |\n"
        for name, duration in sorted_timings:
            pct = (duration / total_time) * 100
            report += f"| {name} | {duration:.3f} | {pct:.2f}% |\n"
            
        report += "\n## Top 3 Bottlenecks\n\n"
        for i in range(min(3, len(sorted_timings))):
            report += f"{i+1}. **{sorted_timings[i][0]}** ({sorted_timings[i][1]:.3f}s)\n"
            
        with open("C:/Users/Vinfocom/.gemini/antigravity-ide/brain/eff278fc-3d96-4836-ba15-99ae18434618/PIPELINE_PROFILING_AUDIT.md", "w") as f:
            f.write(report)
            
    except Exception as e:
        with open("C:/Users/Vinfocom/.gemini/antigravity-ide/brain/eff278fc-3d96-4836-ba15-99ae18434618/PIPELINE_PROFILING_AUDIT.md", "w") as f:
            f.write(f"ERROR: {traceback.format_exc()}")

if __name__ == "__main__":
    asyncio.run(main())
