"""Frame extraction service utilizing OpenCV to extract frames and Qwen2.5-VL VLM to generate rich metadata.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
from loguru import logger

from app.core.config import settings, PROJECT_ROOT
from app.core.exceptions import FrameExtractionError, VideoNotFoundError, MetadataGenerationError
from app.services.video import VideoService
from app.services.vlm_factory import get_vlm_service
from app.services.motion_window_service import MotionWindowService
from app.core.profiler import PerformanceTracker
from app.services.status_service import JobStatusService
from app.schemas.telemetry import SamplingMetrics


class FrameExtractionService:
    """Service layer managing video frame extraction and coordinating VLM analysis runs."""

    @classmethod
    def compute_similarity_metrics(
        cls, img1: Any, img2: Any
    ) -> Tuple[float, float, float]:
        """Computes similarity metrics between two BGR frame images.

        Returns:
            Tuple[float, float, float]: (histogram_difference, ssim_difference, motion_score)
        """
        import cv2
        import numpy as np

        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 1. Histogram Difference (using correlation)
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        # Normalize histograms to avoid scaling issues
        hist1 = hist1 / (hist1.sum() + 1e-7)
        hist2 = hist2 / (hist2.sum() + 1e-7)

        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_diff = max(0.0, 1.0 - correlation)

        # 2. SSIM (Structural Similarity Index Measure)
        C1 = 6.5025
        C2 = 58.5225

        g1 = gray1.astype(np.float32)
        g2 = gray2.astype(np.float32)

        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_score = float(np.mean(ssim_map))
        ssim_diff = max(0.0, 1.0 - ssim_score)

        # 3. Motion Detection (Frame Differencing)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = float(np.count_nonzero(thresh) / thresh.size)

        return hist_diff, ssim_diff, motion_score

    @classmethod
    async def extract_frames(cls, video_id: str) -> Dict[str, Any]:
        """Processes video, extracts 1 frame per second of playtime, and triggers VLM rich metadata generation.

        Args:
            video_id: Unique UUID4 string representing the uploaded video.

        Returns:
            dict: Processing stats and rich frame metadata collection list.

        Raises:
            VideoNotFoundError: If the source video doesn't exist.
            FrameExtractionError: If OpenCV capture, JPEG writes, or VLM analysis fail.
        """
        # 1. Retrieve raw video path and verify existence
        metadata, video_path = VideoService.get_video(video_id)

        # Initialize performance profiling
        tracker = PerformanceTracker(video_id)
        tracker.set_upload_time(metadata.get("upload_duration_ms", 0.0))
        tracker.start_pipeline()

        frame_dir = settings.FRAMES_DIR / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initiated frame extraction and VLM analysis for video ID: {video_id} | Path: {video_path}")
        meta_logger = logger.bind(context="metadata")
        meta_logger.info(f"Initiated frame extraction pipeline for video_id: {video_id}")
        
        JobStatusService.update(video_id, current_step="Extracting keyframes...", progress_percent=0.0)

        # 2. Open Video Capture using OpenCV and extract frame JPEGs
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"OpenCV failed to open video file: {video_path}")
            raise FrameExtractionError(f"Failed to open video file: {metadata.get('filename')}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        estimated_analysis_frames = int(total_frames / fps) if fps > 0 else 0
        JobStatusService.update(video_id, total_frames=estimated_analysis_frames)

        logger.debug(f"Video specs: FPS={fps} | Total Frames={total_frames}")

        motion_windows: List[Tuple[float, float]] = []
        is_motion_windowing_enabled = settings.ENABLE_MOTION_WINDOWING
        window_stats = {}
        if is_motion_windowing_enabled:
            logger.info(f"Running Motion Window Service for video {video_id}...")
            motion_windows, window_metrics = MotionWindowService.detect_motion_windows(video_path)
            window_stats = {idx: {"start": w[0], "end": w[1], "frames_in": 0, "frames_selected": 0} for idx, w in enumerate(motion_windows)}
            logger.info(f"Detected {len(motion_windows)} motion windows.")

        sampling_metrics = SamplingMetrics(
            video_id=video_id,
            motion_windows_detected=len(motion_windows)
        )

        # Temporary list of tuples to pass to VLM batching: (frame_id, video_id, timestamp_seconds, frame_absolute_path)
        extracted_tuples: List[Tuple[str, str, float, Path]] = []
        extract_timings: Dict[str, float] = {}
        frame_idx = 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        current_fps = 0.1
        current_state = "IDLE"
        burst_timer_seconds = 0.0

        # Telemetry trackers
        state_durations = {"IDLE": 0.0, "LOW_ACTIVITY": 0.0, "NORMAL_ACTIVITY": 0.0, "HIGH_ACTIVITY": 0.0, "BURST_CAPTURE": 0.0, "COOLDOWN": 0.0}
        fps_transitions = 0
        burst_activations = 0

        saved_paths: List[Path] = []
        current_raw_frame = 0

        last_sent_frame = None
        last_eval_frame = None
        total_extracted = 0
        skipped_count = 0
        
        # Event Candidate Layer Configuration
        last_candidate_second = 0.0
        accumulated_vns = 0.0
        candidate_frames_generated = 0
        candidate_frames_rejected = 0
        VNS_THRESHOLD = 45.0
        MAX_GAP_SECONDS = 10.0
        
        try:
            while True:
                frame_interval = max(1, int(round(fps / current_fps)))
                
                # Seek to exact frame position
                if current_raw_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_raw_frame)

                extract_start = time.perf_counter()
                success, frame = cap.read()
                if not success:
                    break

                second = current_raw_frame / fps
                total_extracted += 1
                frame_id = f"{video_id}_f{frame_idx:04d}"

                # Calculate ADS
                ads = 0
                hist_diff, ssim_diff, motion_score = 0.0, 0.0, 0.0
                if last_eval_frame is not None:
                    hist_diff, ssim_diff, motion_score = cls.compute_similarity_metrics(frame, last_eval_frame)
                    
                    # Normalize inputs based on practical maximums
                    norm_motion = min(1.0, motion_score / 0.4) 
                    norm_ssim = min(1.0, ssim_diff / 0.2)
                    norm_hist = min(1.0, hist_diff / 0.5)
                    
                    ads = int(((norm_motion * 0.6) + (norm_ssim * 0.3) + (norm_hist * 0.1)) * 100)
                
                last_eval_frame = frame.copy()

                # Dynamic FPS State Machine
                prev_state = current_state
                
                if current_state == "BURST_CAPTURE":
                    burst_timer_seconds += (1.0 / current_fps)
                    if burst_timer_seconds >= 3.0 and ads < 80:
                        current_state = "COOLDOWN"
                        burst_timer_seconds = 0.0
                elif current_state == "COOLDOWN":
                    burst_timer_seconds += (1.0 / current_fps)
                    if burst_timer_seconds >= 2.0:
                        current_state = "NORMAL_ACTIVITY"
                        burst_timer_seconds = 0.0
                    elif ads > 80:
                        current_state = "BURST_CAPTURE"
                        burst_timer_seconds = 0.0
                        burst_activations += 1
                else:
                    if ads > 80:
                        current_state = "BURST_CAPTURE"
                        burst_activations += 1
                        burst_timer_seconds = 0.0
                    elif ads > 50:
                        current_state = "HIGH_ACTIVITY"
                    elif ads > 20:
                        current_state = "NORMAL_ACTIVITY"
                    elif ads > 5:
                        current_state = "LOW_ACTIVITY"
                    else:
                        current_state = "IDLE"
                        
                if current_state != prev_state:
                    fps_transitions += 1
                    
                # Map State to FPS
                if current_state == "IDLE": current_fps = 0.1
                elif current_state == "LOW_ACTIVITY": current_fps = 0.5
                elif current_state == "NORMAL_ACTIVITY": current_fps = 1.0
                elif current_state == "HIGH_ACTIVITY": current_fps = 2.0
                elif current_state == "BURST_CAPTURE": current_fps = 5.0
                elif current_state == "COOLDOWN": current_fps = 1.0
                
                # Track duration in state
                if current_state in state_durations:
                    state_durations[current_state] += (1.0 / current_fps)
                elif current_state == "COOLDOWN":
                    state_durations["NORMAL_ACTIVITY"] += (1.0 / current_fps)

                should_send = True

                current_window_idx = None
                if is_motion_windowing_enabled:
                    in_window = False
                    for idx, (start_w, end_w) in enumerate(motion_windows):
                        if start_w <= second <= end_w:
                            in_window = True
                            window_stats[idx]["frames_in"] += 1
                            current_window_idx = idx
                            break
                    if not in_window:
                        should_send = False
                        sampling_metrics.dropped_by_motion_window += 1

                if should_send and settings.ENABLE_ADAPTIVE_SAMPLING and last_sent_frame is not None:
                    hist_diff, ssim_diff, motion_score = cls.compute_similarity_metrics(frame, last_sent_frame)

                    ssim_score = 1.0 - ssim_diff
                    
                    # Accumulate Visual Novelty Score
                    vns_delta = (
                        0.45 * (1.0 - ssim_score) +  # Structural change
                        0.35 * motion_score +        # Raw pixel difference
                        0.20 * hist_diff             # Lighting/color shifts
                    ) * 100
                    
                    accumulated_vns += vns_delta
                    time_since_last_candidate = second - last_candidate_second
                    
                    # Candidate Selection Rules
                    is_candidate = False
                    if accumulated_vns >= VNS_THRESHOLD:
                        is_candidate = True
                    elif time_since_last_candidate >= MAX_GAP_SECONDS:
                        is_candidate = True
                        
                    if not is_candidate:
                        should_send = False
                        candidate_frames_rejected += 1
                        # Maintain legacy metrics based on strict thresholds for backward compatibility
                        if ssim_score >= settings.SSIM_THRESHOLD:
                            sampling_metrics.dropped_by_ssim += 1
                        elif hist_diff <= settings.HISTOGRAM_THRESHOLD:
                            sampling_metrics.dropped_by_histogram += 1
                        else:
                            sampling_metrics.dropped_by_motion_threshold += 1
                    else:
                        candidate_frames_generated += 1
                        accumulated_vns = 0.0
                        last_candidate_second = second

                if should_send:
                    if is_motion_windowing_enabled and current_window_idx is not None:
                        window_stats[current_window_idx]["frames_selected"] += 1

                    # Formulate paths
                    frame_filename = f"frame_{frame_idx:04d}.jpg"
                    out_path = frame_dir / frame_filename

                    # Save frame image as JPEG
                    write_success = cv2.imwrite(str(out_path), frame)
                    if not write_success:
                        raise FrameExtractionError(f"Failed to write frame image disk buffer to: {out_path}")
                    extract_duration_ms = (time.perf_counter() - extract_start) * 1000.0

                    saved_paths.append(out_path)
                    extracted_tuples.append((frame_id, video_id, second, out_path))
                    extract_timings[frame_id] = extract_duration_ms

                    last_sent_frame = frame.copy()
                    
                    if total_extracted % 2 == 0 and estimated_analysis_frames > 0:
                        prog = min(10.0, (total_extracted / estimated_analysis_frames) * 10.0)
                        JobStatusService.update(video_id, progress_percent=round(prog, 1))
                else:
                    skipped_count += 1
                    logger.debug(
                        f"Skipped frame {frame_id} at {second}s (similarity: hist_diff={hist_diff:.3f}, "
                        f"ssim_diff={ssim_diff:.3f}, motion={motion_score:.3f})"
                    )

                # Increment states
                frame_idx += 1
                current_raw_frame += max(1, int(round(fps / current_fps)))

        except Exception as exc:
            # Perform disk rollbacks of saved images on failures
            logger.warning(f"Error occurred during frame JPEG extraction for ID {video_id}. Cleaning up saved images.")
            for path in saved_paths:
                if path.exists():
                    path.unlink()
            if frame_dir.exists() and not any(frame_dir.iterdir()):
                frame_dir.rmdir()
            logger.exception(f"Exception during frame extraction pipeline for video: {video_id}")
            JobStatusService.update(video_id, status="failed", current_step="Failed during frame extraction")
            raise FrameExtractionError(f"Extraction failed: {str(exc)}")
        finally:
            cap.release()

        processed_count = len(extracted_tuples)
        reduction_percentage = 0.0
        if total_extracted > 0:
            reduction_percentage = ((total_extracted - processed_count) / total_extracted) * 100.0

        if is_motion_windowing_enabled:
            for idx, stats in window_stats.items():
                logger.info(
                    f"Motion Window {idx+1}:\n"
                    f"Start={stats['start']:.1f}s\n"
                    f"End={stats['end']:.1f}s\n"
                    f"FramesInWindow={stats['frames_in']}\n"
                    f"FramesSelected={stats['frames_selected']}"
                )

        logger.info(
            f"--- Pipeline Report ---\n"
            f"Video ID: {video_id}\n"
            f"Total Frames Extracted (1fps base): {total_extracted}\n"
            f"Motion Windows Detected: {len(motion_windows) if is_motion_windowing_enabled else 'N/A (Disabled)'}\n"
            f"Frames Sent To VLM: {processed_count}\n"
            f"Estimated Workload Reduction: {reduction_percentage:.1f}%\n"
            f"-----------------------"
        )
        
        sampling_metrics.total_frames_seen = total_extracted
        sampling_metrics.kept_frames = processed_count
        sampling_metrics.reduction_percent = reduction_percentage
        sampling_metrics.vlm_calls_saved = skipped_count
        if total_extracted > 0:
            sampling_metrics.video_duration_seconds = total_extracted / fps if fps > 0 else 0
            
        sampling_metrics.mode_idle_seconds = state_durations["IDLE"]
        sampling_metrics.mode_low_seconds = state_durations["LOW_ACTIVITY"]
        sampling_metrics.mode_normal_seconds = state_durations["NORMAL_ACTIVITY"]
        sampling_metrics.mode_high_seconds = state_durations["HIGH_ACTIVITY"]
        sampling_metrics.mode_burst_seconds = state_durations["BURST_CAPTURE"]
        sampling_metrics.fps_transitions = fps_transitions
        sampling_metrics.burst_activations = burst_activations
        
        # Approximate average FPS
        total_duration = sum(state_durations.values())
        if total_duration > 0:
            sampling_metrics.average_extraction_fps = total_extracted / total_duration
            
        # Candidate Layer Telemetry
        sampling_metrics.candidate_frames_generated = candidate_frames_generated
        sampling_metrics.candidate_frames_rejected = candidate_frames_rejected
        sampling_metrics.candidate_frames_sent_to_vlm = processed_count
        if candidate_frames_generated + candidate_frames_rejected > 0:
            sampling_metrics.candidate_reduction_percent = (candidate_frames_rejected / (candidate_frames_generated + candidate_frames_rejected)) * 100.0
        if total_duration > 0:
            sampling_metrics.average_candidate_density = candidate_frames_generated / total_duration
        
        logger.info(f"Successfully extracted {processed_count} candidate frames from video: {video_id}. Starting VLM batch processing...")
        JobStatusService.update(video_id, current_step=f"Starting VLM Analysis (0/{processed_count})...", total_frames=processed_count, progress_percent=10.0)

        # 3. Process extracted frames in batches using QwenVLMService
        rich_frames: List[Dict[str, Any]] = []
        successful_count = 0
        failed_count = 0

        # Create video-specific metadata directory: data/metadata/{video_id}/
        video_metadata_dir = settings.METADATA_DIR / video_id
        video_metadata_dir.mkdir(parents=True, exist_ok=True)

        batch_size = settings.BATCH_SIZE
        for i in range(0, processed_count, batch_size):
            batch = extracted_tuples[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (processed_count + batch_size - 1) // batch_size
            logger.info(f"Processing VLM batch {current_batch_num}/{total_batches} (frames {i} to {i + len(batch)} of {processed_count}) for video ID {video_id}...")

            try:
                # Generate rich metadata for batch
                vlm_service = get_vlm_service()
                batch_results = await vlm_service.generate_metadata_batch(batch)
                logger.info(f"Successfully analyzed VLM batch {current_batch_num}/{total_batches} | Generated {len(batch_results)} metadata profiles.")

                # Process batch results
                for rich_meta, timings in batch_results:
                    # Convert Pydantic object to dict
                    meta_dict = rich_meta.model_dump()
                    rich_frames.append(meta_dict)
                    successful_count += 1

                    # Save individual JSON file to: data/metadata/{video_id}/{frame_id}.json
                    individual_json_path = video_metadata_dir / f"{rich_meta.frame_id}.json"
                    write_start = time.perf_counter()
                    try:
                        with open(individual_json_path, "w", encoding="utf-8") as ind_file:
                            json.dump(meta_dict, ind_file, indent=4)
                        write_duration_ms = (time.perf_counter() - write_start) * 1000.0
                    except Exception as ind_exc:
                        logger.warning(f"Save failure on individual metadata file {individual_json_path}: {str(ind_exc)}")
                        write_duration_ms = (time.perf_counter() - write_start) * 1000.0

                    # Record frame-level performance metrics
                    tracker.add_frame_timing(
                        frame_id=rich_meta.frame_id,
                        extract_ms=extract_timings.get(rich_meta.frame_id, 0.0),
                        ocr_ms=timings.get("ocr_ms", 0.0),
                        vlm_ms=timings.get("vlm_ms", 0.0),
                        json_repair_ms=timings.get("json_repair_ms", 0.0),
                        validation_ms=timings.get("validation_ms", 0.0),
                        write_ms=write_duration_ms,
                    )
                # Update status per batch
                vlm_progress = (successful_count / processed_count) * 70.0 if processed_count > 0 else 70.0
                JobStatusService.update(
                    video_id, 
                    current_step=f"Analyzing frames ({successful_count}/{processed_count})...",
                    processed_frames=successful_count,
                    progress_percent=round(10.0 + vlm_progress, 1)
                )

            except Exception as batch_exc:
                logger.exception(f"VLM analysis batch failure within range [{i} to {i + len(batch)}] for video: {video_id}")
                # Increment failed count by batch length
                failed_count += len(batch)
                continue

        # Adjust failed count by checking successfully generated frames count
        failed_count = processed_count - successful_count

        # 4. Save aggregated frames metadata list catalog: data/metadata/{video_id}_frames.json
        metadata_catalog_path = settings.METADATA_DIR / f"{video_id}_frames.json"
        try:
            with open(metadata_catalog_path, "w", encoding="utf-8") as cat_file:
                json.dump(rich_frames, cat_file, indent=4)
        except Exception as cat_exc:
            logger.exception(f"Failed writing frame catalog index JSON metadata for video: {video_id}")
            JobStatusService.update(video_id, status="failed", current_step="Failed saving frame metadata")
            raise FrameExtractionError(f"Failed saving frame catalog index: {str(cat_exc)}")

        # 5. Formulate final stats and log success
        reduction_percent = round((skipped_count / total_extracted) * 100.0, 2) if total_extracted > 0 else 0.0
        stats = {
            "video_id": video_id,
            "processed_frames": processed_count,
            "successful_frames": successful_count,
            "failed_frames": failed_count,
            "frames": rich_frames,
            "total_frames_extracted": total_extracted,
            "frames_sent_to_qwen": processed_count,
            "frames_skipped": skipped_count,
            "reduction_percent": reduction_percent,
            "frames_extracted": total_extracted,
            "frames_analyzed": processed_count,
        }

        # Trigger Event Aggregation service to group consecutive similar frames into events
        from app.services.event_aggregation import EventAggregationService
        from app.services.search_service import SearchService
        try:
            events = EventAggregationService.process_events(video_id, rich_frames)
            if events:
                try:
                    SearchService.index_events(video_id, events)
                except Exception as search_exc:
                    logger.exception(f"Failed to index events in vector store for video: {video_id}")
        except Exception as event_exc:
            logger.exception(f"Failed to execute event aggregation for video: {video_id}")

        # Stop and finalize pipeline timings
        tracker.end_pipeline()
        
        sampling_metrics.processing_duration_seconds = tracker.end_time - tracker.start_time
        
        if settings.ENABLE_ADAPTIVE_SAMPLING:
            tracker.set_sampling_stats(sampling_metrics)
        tracker.finalize()

        logger.info(
            f"Completed VLM rich metadata indexing for video: {video_id} | "
            f"Processed: {processed_count} | Success: {successful_count} | Failed: {failed_count}"
        )
        meta_logger.info(
            f"VLM Ingestion success | video_id={video_id} | "
            f"processed={processed_count} | success={successful_count} | failed={failed_count}"
        )

        return stats

    @classmethod
    def get_frames(cls, video_id: str) -> List[Dict[str, Any]]:
        """Retrieves previously extracted VLM rich frame metadata index catalogs.

        Args:
            video_id: Unique UUID4 string representing the video.

        Returns:
            list: Collection of VLM rich frame metadata dicts.

        Raises:
            VideoNotFoundError: If the video has not been uploaded or frames are not extracted.
        """
        # Validate video existence first
        VideoService.get_video(video_id)

        metadata_path = settings.METADATA_DIR / f"{video_id}_frames.json"
        if not metadata_path.exists():
            logger.warning(f"Lookup failed. Rich frames have not been extracted for video: {video_id}")
            raise VideoNotFoundError(f"Frames for video ID '{video_id}' have not been extracted yet.")

        try:
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                return json.load(meta_file)
        except Exception as exc:
            logger.exception(f"Failed to read rich frame catalog index: {metadata_path}")
            raise FrameExtractionError("Corrupted or unreadable frame metadata catalog.")
