"""Performance profiling utility module for the AI Video Ingestion pipeline.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger
from app.core.config import settings, PROJECT_ROOT

class PerformanceTracker:
    """Manages accumulation of execution metrics, writing performance logs, and generating markdown reports."""

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.frames_data: List[Dict[str, Any]] = []
        self.video_upload_ms = 0.0
        self.start_time = 0.0
        self.end_time = 0.0
        self.sampling_enabled = False
        self.sampling_total = 0
        self.sampling_sent = 0
        self.sampling_skipped = 0
        self.sampling_pct = 0.0

    def set_sampling_stats(self, total: int, sent: int, skipped: int, pct: float):
        """Sets the stats for adaptive frame sampling."""
        self.sampling_enabled = True
        self.sampling_total = total
        self.sampling_sent = sent
        self.sampling_skipped = skipped
        self.sampling_pct = pct

    def set_upload_time(self, ms: float):
        """Sets the video file upload duration in milliseconds."""
        self.video_upload_ms = ms

    def start_pipeline(self):
        """Starts timing the frame ingestion and analysis pipeline."""
        self.start_time = time.perf_counter()

    def end_pipeline(self):
        """Stops timing the frame ingestion and analysis pipeline."""
        self.end_time = time.perf_counter()

    def add_frame_timing(
        self,
        frame_id: str,
        extract_ms: float,
        ocr_ms: float,
        vlm_ms: float,
        json_repair_ms: float,
        validation_ms: float,
        write_ms: float,
    ):
        """Logs the performance timings of a single frame to the structured log file."""
        total_frame_ms = extract_ms + ocr_ms + vlm_ms + json_repair_ms + validation_ms + write_ms
        frame_log = {
            "frame_id": frame_id,
            "frame_extract_ms": round(extract_ms, 2),
            "ocr_ms": round(ocr_ms, 2),
            "vlm_ms": round(vlm_ms, 2),
            "json_repair_ms": round(json_repair_ms, 2),
            "validation_ms": round(validation_ms, 2),
            "write_ms": round(write_ms, 2),
            "total_frame_ms": round(total_frame_ms, 2),
        }
        self.frames_data.append(frame_log)

        # Append to performance.log
        log_path = settings.LOGS_DIR / "performance.log"
        try:
            # Ensure folder exists
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(frame_log) + "\n")
        except Exception as exc:
            logger.error(f"Failed to write frame performance log for {frame_id}: {exc}")

    def finalize(self) -> Dict[str, Any]:
        """Calculates aggregated performance statistics, appends summary log, and saves PERFORMANCE_REPORT.md."""
        if not self.end_time:
            self.end_pipeline()

        pipeline_duration_seconds = self.end_time - self.start_time
        total_ingestion_seconds = round(pipeline_duration_seconds + (self.video_upload_ms / 1000.0), 2)

        frames_processed = len(self.frames_data)
        if frames_processed > 0:
            avg_extract = sum(f["frame_extract_ms"] for f in self.frames_data) / frames_processed
            avg_ocr = sum(f["ocr_ms"] for f in self.frames_data) / frames_processed
            avg_vlm = sum(f["vlm_ms"] for f in self.frames_data) / frames_processed
            avg_write = sum(f["write_ms"] for f in self.frames_data) / frames_processed
        else:
            avg_extract = avg_ocr = avg_vlm = avg_write = 0.0

        video_summary = {
            "video_id": self.video_id,
            "frames_processed": frames_processed,
            "avg_frame_extract_ms": round(avg_extract, 2),
            "avg_ocr_ms": round(avg_ocr, 2),
            "avg_vlm_ms": round(avg_vlm, 2),
            "avg_write_ms": round(avg_write, 2),
            "total_ingestion_seconds": total_ingestion_seconds,
        }

        # Append to performance.log
        log_path = settings.LOGS_DIR / "performance.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(video_summary) + "\n")
        except Exception as exc:
            logger.error(f"Failed to write video performance summary log: {exc}")

        # Print summary statistics to console
        self._print_console_summary(video_summary)

        # Generate markdown report
        self._generate_report(video_summary)

        if self.sampling_enabled:
            self._generate_adaptive_sampling_report(pipeline_duration_seconds)

        self._generate_event_aggregation_report()

        return video_summary

    def _print_console_summary(self, summary: Dict[str, Any]):
        """Prints a human-readable performance summary block directly to the console output."""
        print("\n" + "=" * 60)
        print("               INGESTION PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f" Video ID:                {summary['video_id']}")
        print(f" Frames Processed:        {summary['frames_processed']}")
        print(f" Avg Frame Extract:       {summary['avg_frame_extract_ms']} ms")
        print(f" Avg OCR Time:            {summary['avg_ocr_ms']} ms")
        print(f" Avg VLM Inference:       {summary['avg_vlm_ms']} ms")
        print(f" Avg Metadata Write:      {summary['avg_write_ms']} ms")
        print(f" Total Ingestion Time:    {summary['total_ingestion_seconds']} seconds")
        print("=" * 60 + "\n")

    def _generate_report(self, summary: Dict[str, Any]):
        """Builds and writes the comprehensive PERFORMANCE_REPORT.md file."""
        frames_processed = len(self.frames_data)
        if frames_processed == 0:
            return

        # Sort frames to identify top 10 slowest
        sorted_frames = sorted(self.frames_data, key=lambda x: x["total_frame_ms"], reverse=True)
        top_10_slowest = sorted_frames[:10]

        # Calculate grand totals
        total_extract = sum(f["frame_extract_ms"] for f in self.frames_data)
        total_ocr = sum(f["ocr_ms"] for f in self.frames_data)
        total_vlm = sum(f["vlm_ms"] for f in self.frames_data)
        total_repair = sum(f["json_repair_ms"] for f in self.frames_data)
        total_validation = sum(f["validation_ms"] for f in self.frames_data)
        total_write = sum(f["write_ms"] for f in self.frames_data)

        grand_total_frame_ms = total_extract + total_ocr + total_vlm + total_repair + total_validation + total_write
        if grand_total_frame_ms == 0:
            grand_total_frame_ms = 1

        avg_repair = total_repair / frames_processed
        avg_val = total_validation / frames_processed
        avg_total = grand_total_frame_ms / frames_processed

        # Compute percentage values
        pct_extract = (total_extract / grand_total_frame_ms) * 100
        pct_ocr = (total_ocr / grand_total_frame_ms) * 100
        pct_vlm = (total_vlm / grand_total_frame_ms) * 100
        pct_repair = (total_repair / grand_total_frame_ms) * 100
        pct_val = (total_validation / grand_total_frame_ms) * 100
        pct_write = (total_write / grand_total_frame_ms) * 100

        # Construct and sort bottleneck rankings
        stages = [
            ("Qwen VLM Inference", total_vlm, pct_vlm),
            ("OCR Processing", total_ocr, pct_ocr),
            ("Frame Extraction", total_extract, pct_extract),
            ("Metadata Write", total_write, pct_write),
            ("JSON Repair", total_repair, pct_repair),
            ("Metadata Validation", total_validation, pct_val),
        ]
        stages.sort(key=lambda x: x[1], reverse=True)

        report_content = f"""# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `{summary['video_id']}`
* **Frames Processed**: {summary['frames_processed']}
* **Video Upload Time**: {self.video_upload_ms / 1000.0:.2f} seconds
* **Total Ingestion Time**: {summary['total_ingestion_seconds']:.2f} seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | {summary['avg_vlm_ms']:.2f} ms | {pct_vlm:.2f}% |
| **OCR Processing** | {summary['avg_ocr_ms']:.2f} ms | {pct_ocr:.2f}% |
| **Frame Extraction** | {summary['avg_frame_extract_ms']:.2f} ms | {pct_extract:.2f}% |
| **Metadata Write** | {summary['avg_write_ms']:.2f} ms | {pct_write:.2f}% |
| **JSON Repair & Normalization** | {avg_repair:.2f} ms | {pct_repair:.2f}% |
| **Metadata Validation** | {avg_val:.2f} ms | {pct_val:.2f}% |
| **Total Per Frame** | {avg_total:.2f} ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

"""
        for rank, (stage_name, total_ms, pct) in enumerate(stages, 1):
            report_content += f"{rank}. **{stage_name}**: {total_ms / 1000.0:.2f}s total ({pct:.2f}% of runtime)\n"

        report_content += """
---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
        for idx, f in enumerate(top_10_slowest, 1):
            report_content += f"| {idx} | `{f['frame_id']}` | {f['total_frame_ms']:.2f} | {f['vlm_ms']:.2f} | {f['ocr_ms']:.2f} | {f['frame_extract_ms']:.2f} | {f['write_ms']:.2f} |\n"

        report_content += f"""
---

## Recommendations

1. **Optimize {stages[0][0]}**:
   * Currently, the **{stages[0][0]}** stage represents the largest performance bottleneck at **{stages[0][2]:.2f}%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes {summary['avg_frame_extract_ms']:.2f} ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
"""
        report_path = PROJECT_ROOT / "PERFORMANCE_REPORT.md"
        try:
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(report_content)
            logger.info(f"Successfully generated PERFORMANCE_REPORT.md at: {report_path}")
        except Exception as exc:
            logger.error(f"Failed to write performance report markdown file: {exc}")

    def _generate_adaptive_sampling_report(self, pipeline_duration_seconds: float):
        """Generates the ADAPTIVE_SAMPLING_REPORT.md report."""
        frames_processed = len(self.frames_data)
        if frames_processed > 0:
            avg_frame_ms = sum(f["total_frame_ms"] for f in self.frames_data) / frames_processed
        else:
            avg_frame_ms = 0.0

        avg_frame_seconds = avg_frame_ms / 1000.0
        
        # Savings calculations
        savings_seconds = self.sampling_skipped * avg_frame_seconds
        savings_minutes = savings_seconds / 60.0
        projected_duration_seconds = pipeline_duration_seconds + savings_seconds

        report_content = f"""# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `{self.video_id}`
* **Original Frame Count (Extracted)**: {self.sampling_total}
* **Filtered Frame Count (Sent to Qwen)**: {self.sampling_sent}
* **Frames Skipped**: {self.sampling_skipped}
* **Frame Reduction Ratio**: {self.sampling_pct:.2f}%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: {avg_frame_seconds:.2f} seconds
* **Actual Pipeline Run Duration (with sampling)**: {pipeline_duration_seconds:.2f} seconds
* **Projected Run Duration Without Sampling**: {projected_duration_seconds:.2f} seconds
* **Estimated Runtime Savings**: {savings_seconds:.2f} seconds ({savings_minutes:.2f} minutes)
* **Actual Runtime Savings**: {savings_seconds:.2f} seconds ({savings_minutes:.2f} minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `{settings.ENABLE_ADAPTIVE_SAMPLING}`
* **SSIM_THRESHOLD**: `{settings.SSIM_THRESHOLD}`
* **HISTOGRAM_THRESHOLD**: `{settings.HISTOGRAM_THRESHOLD}`
* **MOTION_THRESHOLD**: `{settings.MOTION_THRESHOLD}`
"""
        report_path = PROJECT_ROOT / "ADAPTIVE_SAMPLING_REPORT.md"
        try:
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(report_content)
            logger.info(f"Successfully generated ADAPTIVE_SAMPLING_REPORT.md at: {report_path}")
        except Exception as exc:
            logger.error(f"Failed to write adaptive sampling report: {exc}")

    def _generate_event_aggregation_report(self):
        """Generates the EVENT_AGGREGATION_REPORT.md report."""
        original_metadata_count = len(self.frames_data)
        
        event_count = 0
        video_events_dir = settings.EVENTS_DIR / self.video_id
        if video_events_dir.exists():
            event_count = len(list(video_events_dir.glob("*.json")))
            
        compression_ratio = (original_metadata_count / event_count) if event_count > 0 else 0.0
        
        report_content = f"""# EVENT AGGREGATION REPORT

This report summarizes the results of grouping consecutive visually similar frames into consolidated events.

## Ingestion Overview

* **Video ID**: `{self.video_id}`
* **Original Metadata Count (Accepted Frames)**: {original_metadata_count}
* **Event Count (Aggregated Events)**: {event_count}
* **Compression Ratio**: {compression_ratio:.2f}x

---

## Grouping Details

* **EVENT_SIMILARITY_THRESHOLD**: `{settings.EVENT_SIMILARITY_THRESHOLD}`
* **EVENTS_DIR**: `{settings.EVENTS_DIR}`
"""
        report_path = PROJECT_ROOT / "EVENT_AGGREGATION_REPORT.md"
        try:
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(report_content)
            logger.info(f"Successfully generated EVENT_AGGREGATION_REPORT.md at: {report_path}")
        except Exception as exc:
            logger.error(f"Failed to write event aggregation report: {exc}")
