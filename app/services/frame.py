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
from app.services.pipeline_contract import frame_catalog_path, frame_metadata_dir
from app.core.profiler import PerformanceTracker
from app.services.status_service import JobStatusService
from app.services.object_detection.detector import ObjectDetector
from app.services.object_tracker import ObjectTrackerService
from app.services.event_candidate_selector import EventCandidateSelector


class FrameExtractionService:
    """Service layer managing video frame extraction and coordinating VLM analysis runs."""

    @staticmethod
    def _timestamp_in_windows(
        timestamp_seconds: float,
        windows: List[Tuple[float, float]],
        context_seconds: float = 0.0,
    ) -> bool:
        """Return true when a timestamp falls inside any window, optionally padded by context."""
        pad = max(0.0, float(context_seconds or 0.0))
        return any((start - pad) <= timestamp_seconds <= (end + pad) for start, end in windows)

    @classmethod
    def _select_vlm_candidate_tuples(
        cls,
        extracted_tuples: List[Tuple[str, str, float, Path]],
        motion_windows: List[Tuple[float, float]],
    ) -> List[Tuple[str, str, float, Path, Dict[str, Any]]]:
        """Send only event-likely frames to the VLM, not empty background frames."""
        if not extracted_tuples:
            return []
        detector = ObjectDetector()
        frame_detections = [
            detector.detect_frame(path, frame_id, video_id, ts)
            for frame_id, video_id, ts, path in extracted_tuples
        ]
        tracking_map = ObjectTrackerService.track_frames(frame_detections)
        selection_map = EventCandidateSelector.select(
            extracted_tuples=extracted_tuples,
            frame_detections=frame_detections,
            tracking_map=tracking_map,
            motion_windows=motion_windows,
        )

        candidate_tuples: List[Tuple[str, str, float, Path, Dict[str, Any]]] = []
        for frame_id, video_id, ts, path in extracted_tuples:
            selection = selection_map.get(frame_id, {})
            if selection.get("selected"):
                candidate_tuples.append((frame_id, video_id, ts, path, selection))

        if not candidate_tuples and extracted_tuples:
            logger.info("No event-like frames selected for VLM; skipping empty/background-only coverage frames.")

        return candidate_tuples

    @classmethod
    def _build_temporal_context_strips(
        cls,
        extracted_tuples: List[Tuple[str, str, float, Path]],
        frame_dir: Path,
    ) -> List[Tuple[str, str, float, Path, Path]]:
        """Create prev/current/next analysis strips while preserving original frame paths."""
        if not settings.ENABLE_TEMPORAL_CONTEXT_STRIPS or len(extracted_tuples) < 2:
            return [(frame_id, video_id, ts, path, path) for frame_id, video_id, ts, path in extracted_tuples]

        import cv2
        import numpy as np

        context_dir = frame_dir / "context"
        context_dir.mkdir(parents=True, exist_ok=True)

        context_tuples: List[Tuple[str, str, float, Path, Path]] = []
        labels = ("PREVIOUS", "CURRENT", "NEXT")

        for idx, (frame_id, video_id, ts, original_path) in enumerate(extracted_tuples):
            neighbor_paths = [
                extracted_tuples[max(0, idx - 1)][3],
                original_path,
                extracted_tuples[min(len(extracted_tuples) - 1, idx + 1)][3],
            ]

            panels = []
            for label, path in zip(labels, neighbor_paths):
                image = cv2.imread(str(path))
                if image is None:
                    image = np.zeros((360, 640, 3), dtype=np.uint8)
                image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_AREA)
                cv2.rectangle(image, (0, 0), (512, 34), (0, 0, 0), thickness=-1)
                cv2.putText(
                    image,
                    label,
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(image)

            strip = cv2.hconcat(panels)
            out_path = context_dir / f"{original_path.stem}_context.jpg"
            if not cv2.imwrite(str(out_path), strip):
                logger.warning(f"Failed to write temporal context strip for {frame_id}; using original frame.")
                out_path = original_path

            context_tuples.append((frame_id, video_id, ts, out_path, original_path))

        return context_tuples

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
        source_video_duration_seconds = (total_frames / fps) if fps > 0 else 0.0
        tracker.set_video_duration(source_video_duration_seconds)
        
        estimated_analysis_frames = int(total_frames / fps) if fps > 0 else 0
        JobStatusService.update(video_id, total_frames=estimated_analysis_frames)

        logger.debug(f"Video specs: FPS={fps} | Total Frames={total_frames}")

        motion_windows: List[Tuple[float, float]] = []
        is_motion_windowing_enabled = settings.ENABLE_MOTION_WINDOWING
        window_stats = {}
        if is_motion_windowing_enabled:
            logger.info(f"Running Motion Window Service for video {video_id}...")
            motion_windows = MotionWindowService.detect_motion_windows(video_path)
            window_stats = {idx: {"start": w[0], "end": w[1], "frames_in": 0, "frames_selected": 0} for idx, w in enumerate(motion_windows)}
            logger.info(f"Detected {len(motion_windows)} motion windows.")

        # Temporary list of tuples to pass to VLM batching: (frame_id, video_id, timestamp_seconds, frame_absolute_path)
        extracted_tuples: List[Tuple[str, str, float, Path]] = []
        extract_timings: Dict[str, float] = {}
        frame_idx = 1
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_interval = max(1, int(round(fps)))
        current_raw_frame = 0

        saved_paths: List[Path] = []
        adaptive_telemetry: List[Dict[str, Any]] = []

        last_sent_frame = None
        last_retained_timestamp = 0.0
        last_out_of_window_timestamp = None
        
        retained_by_motion = 0
        retained_by_ssim = 0
        retained_by_temporal = 0
        retained_by_baseline = 0
        retained_by_gap_safeguard = 0
        
        total_extracted = 0
        skipped_count = 0

        try:
            while True:
                # Seek to exact frame position (more robust than MSEC seeking across codecs)
                if current_raw_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_raw_frame)

                # Read frame at current seek position
                extract_start = time.perf_counter()
                success, frame = cap.read()
                if not success:
                    break

                second = current_raw_frame / fps
                total_extracted += 1
                frame_id = f"{video_id}_f{frame_idx:04d}"

                should_send = True
                hist_diff, ssim_diff, motion_score = 0.0, 0.0, 0.0
                retained_by_outside_baseline = False
                retained_by_coverage_safeguard = False
                passed_temporal = False
                passed_ssim = False
                passed_histogram = False
                passed_motion = False
                decision_reasons: List[str] = []

                current_window_idx = None
                in_motion_window = False
                if is_motion_windowing_enabled and motion_windows:
                    for idx, (start_w, end_w) in enumerate(motion_windows):
                        if start_w <= second <= end_w:
                            in_motion_window = True
                            window_stats[idx]["frames_in"] += 1
                            current_window_idx = idx
                            break
                    if not in_motion_window:
                        baseline_interval = max(1.0, float(settings.OUT_OF_WINDOW_BASELINE_SECONDS))
                        if last_out_of_window_timestamp is None or (second - last_out_of_window_timestamp) >= baseline_interval:
                            should_send = True
                            retained_by_outside_baseline = True
                            last_out_of_window_timestamp = second
                            retained_by_baseline += 1
                        else:
                            should_send = False

                if (
                    should_send
                    and settings.ENABLE_ADAPTIVE_SAMPLING
                    and last_sent_frame is not None
                    and not in_motion_window
                    and not retained_by_outside_baseline
                ):
                    hist_diff, ssim_diff, motion_score = cls.compute_similarity_metrics(frame, last_sent_frame)

                    ssim_score = 1.0 - ssim_diff
                    histogram_score = 1.0 - hist_diff
                    
                    passed_ssim = ssim_score < settings.SSIM_THRESHOLD
                    passed_histogram = hist_diff > settings.HISTOGRAM_THRESHOLD
                    passed_motion = motion_score > settings.MOTION_THRESHOLD

                    is_scene_change = passed_histogram or passed_ssim
                    is_motion = passed_motion

                    passed_temporal = False
                    if not (is_scene_change or is_motion):
                        if (second - last_retained_timestamp) >= settings.TEMPORAL_INTERVAL_SECONDS:
                            should_send = True
                            passed_temporal = True
                        else:
                            should_send = False

                    reasons = []
                    if passed_temporal:
                        reasons.append("Temporal safeguard")
                    else:
                        if not passed_ssim: reasons.append("SSIM below threshold")
                        else: reasons.append("SSIM passed (scene change)")
                        if not passed_histogram: reasons.append("Histogram unchanged")
                        else: reasons.append("Histogram passed (scene change)")
                        if not passed_motion: reasons.append("Motion below threshold")
                        else: reasons.append("Motion passed")

                    if should_send:
                        if passed_temporal:
                            retained_by_temporal += 1
                        elif passed_motion:
                            retained_by_motion += 1
                        else:
                            retained_by_ssim += 1
                    decision_reasons = reasons[:]

                max_gap_seconds = max(1.0, float(settings.MAX_FRAME_GAP_SECONDS))
                if not should_send and (second - last_retained_timestamp) >= max_gap_seconds:
                    should_send = True
                    retained_by_coverage_safeguard = True
                    retained_by_gap_safeguard += 1
                    decision_reasons.append("Coverage safeguard")

                if should_send:
                    last_retained_timestamp = second
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
                    if not decision_reasons:
                        decision_reasons.append("Outside motion windows without baseline slot")
                    skipped_count += 1
                    logger.debug(
                        f"Skipped frame {frame_id} at {second}s (similarity: hist_diff={hist_diff:.3f}, "
                        f"ssim_diff={ssim_diff:.3f}, motion={motion_score:.3f})"
                    )

                if settings.ENABLE_ADAPTIVE_SAMPLING or is_motion_windowing_enabled:
                    adaptive_telemetry.append({
                        "frame_id": frame_id,
                        "timestamp": second,
                        "ssim_score": 1.0 - ssim_diff,
                        "histogram_score": 1.0 - hist_diff,
                        "motion_score": motion_score,
                        "thresholds_used": {
                            "ssim": settings.SSIM_THRESHOLD,
                            "histogram": settings.HISTOGRAM_THRESHOLD,
                            "motion": settings.MOTION_THRESHOLD,
                            "max_gap_seconds": settings.MAX_FRAME_GAP_SECONDS,
                        },
                        "passed_ssim": passed_ssim,
                        "passed_histogram": passed_histogram,
                        "passed_motion": passed_motion,
                        "passed_temporal": passed_temporal,
                        "passed_baseline": retained_by_outside_baseline,
                        "passed_coverage_safeguard": retained_by_coverage_safeguard,
                        "in_motion_window": in_motion_window,
                        "final_decision": "KEEP" if should_send else "DROP",
                        "decision_reason": " | ".join(decision_reasons) if decision_reasons else "Initial retention"
                    })

                # Increment states
                frame_idx += 1
                current_raw_frame += frame_interval

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

        retained_count = len(extracted_tuples)
        reduction_percentage = 0.0
        if total_extracted > 0:
            reduction_percentage = ((total_extracted - retained_count) / total_extracted) * 100.0

        if adaptive_telemetry:
            eval_count = len(adaptive_telemetry)
            retained = sum(1 for t in adaptive_telemetry if t["final_decision"] == "KEEP")
            dropped = eval_count - retained
            ret_rate = (retained / eval_count) * 100.0 if eval_count > 0 else 0.0
            
            avg_ssim = sum(t["ssim_score"] for t in adaptive_telemetry) / eval_count if eval_count > 0 else 0.0
            avg_hist = sum(t["histogram_score"] for t in adaptive_telemetry) / eval_count if eval_count > 0 else 0.0
            avg_motion = sum(t["motion_score"] for t in adaptive_telemetry) / eval_count if eval_count > 0 else 0.0
            
            drop_reasons = {}
            for t in adaptive_telemetry:
                if t["final_decision"] == "DROP":
                    if not t["passed_ssim"]: drop_reasons["SSIM below threshold"] = drop_reasons.get("SSIM below threshold", 0) + 1
                    if not t["passed_histogram"]: drop_reasons["Histogram unchanged"] = drop_reasons.get("Histogram unchanged", 0) + 1
                    if not t["passed_motion"]: drop_reasons["Motion below threshold"] = drop_reasons.get("Motion below threshold", 0) + 1
            
            top_drop_reasons = sorted(drop_reasons.items(), key=lambda x: x[1], reverse=True)
            
            metrics = {
                "total_frames": eval_count,
                "retained_frames": retained,
                "dropped_frames": dropped,
                "retention_rate": ret_rate,
                "frames_retained_by_motion": retained_by_motion,
                "frames_retained_by_ssim": retained_by_ssim,
                "frames_retained_by_temporal": retained_by_temporal,
                "frames_retained_by_baseline": retained_by_baseline,
                "frames_retained_by_gap_safeguard": retained_by_gap_safeguard,
                "avg_ssim": avg_ssim,
                "avg_histogram": avg_hist,
                "avg_motion": avg_motion,
                "top_drop_reasons": drop_reasons
            }
            
            debug_dir = PROJECT_ROOT / "data" / "reports" / "sampling_debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_path = debug_dir / f"{video_id}.json"
            try:
                with open(debug_path, "w", encoding="utf-8") as f:
                    json.dump({"metrics": metrics, "telemetry": adaptive_telemetry}, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to write sampling telemetry JSON: {e}")
                
            report_lines = [
                "# ADAPTIVE SAMPLING DIAGNOSTIC REPORT\n",
                f"**Video ID:** {video_id}\n",
                f"**Frames Evaluated:** {eval_count}",
                f"**Frames Retained:** {retained}",
                f"  - By Motion: {retained_by_motion}",
                f"  - By SSIM/Hist: {retained_by_ssim}",
                f"  - By Temporal Safeguard: {retained_by_temporal}",
                f"  - By Out-of-Window Baseline: {retained_by_baseline}",
                f"  - By Coverage Safeguard: {retained_by_gap_safeguard}",
                f"**Frames Dropped:** {dropped}",
                f"**Retention Rate:** {ret_rate:.2f}%\n",
                f"**Average SSIM:** {avg_ssim:.3f}",
                f"**Average Histogram Similarity:** {avg_hist:.3f}",
                f"**Average Motion Score:** {avg_motion:.3f}\n",
                "**Top Drop Reasons:**"
            ]
            for reason, count in top_drop_reasons:
                report_lines.append(f"* {reason}: {count} times")
                
            report_path = PROJECT_ROOT / "ADAPTIVE_SAMPLING_DIAGNOSTIC_REPORT.md"
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(report_lines))
            except Exception as e:
                logger.error(f"Failed to write sampling telemetry MD: {e}")
                
            logger.info(
                f"\n=============== SAMPLING DIAGNOSTICS ===============\n"
                f"Frames Evaluated: {eval_count}\n"
                f"Frames Retained: {retained}\n"
                f"Frames Dropped: {dropped}\n"
                f"Retention Rate: {ret_rate:.2f}%\n"
                f"Average SSIM: {avg_ssim:.3f}\n"
                f"Average Motion: {avg_motion:.3f}\n"
                f"====================================================\n"
            )

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
            f"Frames Retained For Coverage: {retained_count}\n"
            f"Estimated Workload Reduction: {reduction_percentage:.1f}%\n"
            f"-----------------------"
        )
        
        vlm_candidate_tuples = cls._select_vlm_candidate_tuples(extracted_tuples, motion_windows)
        candidate_frame_ids = {frame_id for frame_id, _video_id, _ts, _path, _context in vlm_candidate_tuples}
        candidate_context_map = {
            frame_id: context for frame_id, _video_id, _ts, _path, context in vlm_candidate_tuples
        }

        logger.info(
            f"Retained {retained_count} frames for coverage and selected {len(vlm_candidate_tuples)} "
            f"event-like frames for VLM analysis for video: {video_id}."
        )
        JobStatusService.update(
            video_id,
            current_step=f"Starting VLM Analysis (0/{len(vlm_candidate_tuples)})...",
            total_frames=len(vlm_candidate_tuples),
            progress_percent=10.0,
        )

        # 3. Process extracted frames in batches using the configured native VLM backend
        rich_frames: List[Dict[str, Any]] = []
        successful_count = 0
        failed_count = 0

        # Create video-specific metadata directory: data/metadata/{video_id}/
        video_metadata_dir = frame_metadata_dir(video_id)
        video_metadata_dir.mkdir(parents=True, exist_ok=True)

        analysis_pool = cls._build_temporal_context_strips(extracted_tuples, frame_dir)
        analysis_tuples = [
            (item[0], item[1], item[2], item[3], item[4], candidate_context_map.get(item[0], {}))
            for item in analysis_pool if item[0] in candidate_frame_ids
        ]
        processed_count = len(analysis_tuples)

        batch_size = settings.BATCH_SIZE
        for i in range(0, processed_count, batch_size):
            batch = analysis_tuples[i : i + batch_size]
            current_batch_num = i // batch_size + 1
            total_batches = (processed_count + batch_size - 1) // batch_size
            logger.info(f"Processing VLM batch {current_batch_num}/{total_batches} (frames {i} to {i + len(batch)} of {processed_count}) for video ID {video_id}...")

            try:
                logger.info(f"Active VLM Backend: {settings.VLM_ENGINE_TYPE}")
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
        metadata_catalog_path = frame_catalog_path(video_id)
        try:
            with open(metadata_catalog_path, "w", encoding="utf-8") as cat_file:
                json.dump(rich_frames, cat_file, indent=4)
        except Exception as cat_exc:
            logger.exception(f"Failed writing frame catalog index JSON metadata for video: {video_id}")
            JobStatusService.update(video_id, status="failed", current_step="Failed saving frame metadata")
            raise FrameExtractionError(f"Failed saving frame catalog index: {str(cat_exc)}")

        # 5. Formulate final stats and log success
        frames_filtered_before_vlm = max(0, retained_count - processed_count)
        reduction_percent = round((skipped_count / total_extracted) * 100.0, 2) if total_extracted > 0 else 0.0
        stats = {
            "video_id": video_id,
            "processed_frames": processed_count,
            "successful_frames": successful_count,
            "failed_frames": failed_count,
            "frames": rich_frames,
            "total_frames_extracted": total_extracted,
            "frames_retained_for_coverage": retained_count,
            "frames_sent_to_qwen": processed_count,
            "frames_filtered_before_vlm": frames_filtered_before_vlm,
            "frames_skipped": skipped_count,
            "reduction_percent": reduction_percent,
            "frames_extracted": total_extracted,
            "frames_analyzed": processed_count,
        }

        # Trigger Event Aggregation service to group consecutive similar frames into events
        from app.services.event_aggregation import EventAggregationService
        from app.services.search_service import SearchService
        try:
            logger.info("Running Event Aggregation...")
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
        if settings.ENABLE_ADAPTIVE_SAMPLING:
            tracker.set_sampling_stats(
                total=total_extracted,
                sent=processed_count,
                skipped=skipped_count,
                pct=reduction_percent,
            )
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

        metadata_path = frame_catalog_path(video_id)
        if not metadata_path.exists():
            logger.warning(f"Lookup failed. Rich frames have not been extracted for video: {video_id}")
            raise VideoNotFoundError(f"Frames for video ID '{video_id}' have not been extracted yet.")

        try:
            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                return json.load(meta_file)
        except Exception as exc:
            logger.exception(f"Failed to read rich frame catalog index: {metadata_path}")
            raise FrameExtractionError("Corrupted or unreadable frame metadata catalog.")
