# Hybrid Tracking Test

This folder contains an isolated `td_case2` experiment that combines periodic YOLO detections with per-object OpenCV KCF propagation between detector refreshes, recently lost-track recovery, and offline fragment reconciliation.

## Purpose

The goal is to test whether we can reduce repeated YOLO inference while preserving usable object-track continuity for traffic and office CCTV footage.

## Hybrid Architecture

```text
Original video
  -> sequential timestamp-based frame selection
  -> YOLO refresh when required
  -> hybrid identity association
  -> recently lost-track recovery
  -> KCF tracker initialization / reinitialization
  -> KCF propagation between YOLO frames
  -> validation and adaptive re-triggering
  -> offline fragment reconciliation
  -> frame metrics, track summaries, comparison outputs
```

## Detector Association vs KCF Propagation

- YOLO detections are authoritative observations.
- KCF boxes are temporary propagated observations between detector frames.
- KCF predictions are never turned into fake YOLO detections.
- Track IDs are maintained by the hybrid track manager, not by KCF itself.

## Why KCF Boxes Must Not Be Treated As Fake YOLO Detections

KCF does not provide detector-class confidence or fresh object discovery. It only propagates an already observed box. Feeding KCF predictions into a detector-first multi-object tracker as if they were new detections would blur the distinction between observation and propagation and would make error analysis unreliable.

## OpenCV KCF Requirement

The experiment requires an OpenCV build that exposes one of:

- `cv2.TrackerKCF_create()`
- `cv2.legacy.TrackerKCF_create()`

Availability check:

```powershell
.\tests\td_case2\.venv\Scripts\python.exe -c "import cv2; print(hasattr(cv2,'TrackerKCF_create'), hasattr(getattr(cv2,'legacy',None),'TrackerKCF_create') if hasattr(cv2,'legacy') else False)"
```

## Environment Variables

- `TD_CASE2_RUN_DIR`
- `TD_CASE2_VIDEO_PATH`
- `TD_CASE2_HYBRID_PROCESSING_FPS`
- `TD_CASE2_HYBRID_YOLO_INTERVAL_FRAMES`
- `TD_CASE2_HYBRID_MAX_YOLO_GAP_SECONDS`
- `TD_CASE2_HYBRID_DETECTION_CONFIDENCE`
- `TD_CASE2_HYBRID_MIN_IOU_MATCH`
- `TD_CASE2_HYBRID_MAX_MISSED_REFRESHES`
- `TD_CASE2_HYBRID_MAX_TRACK_IDLE_SECONDS`
- `TD_CASE2_HYBRID_MIN_TRACK_HITS`
- `TD_CASE2_HYBRID_MOTION_MIN_AREA_RATIO`
- `TD_CASE2_HYBRID_MOTION_PERSISTENCE_FRAMES`
- `TD_CASE2_HYBRID_MOTION_TRACK_REGION_EXPANSION`
- `TD_CASE2_HYBRID_LOST_RECOVERY_SECONDS`
- `TD_CASE2_HYBRID_LOST_MAX_CENTER_DISTANCE_RATIO`
- `TD_CASE2_HYBRID_LOST_MIN_AREA_RATIO`
- `TD_CASE2_HYBRID_LOST_MAX_AREA_RATIO`
- `TD_CASE2_HYBRID_EMPTY_SCENE_YOLO_INTERVAL_SECONDS`
- `TD_CASE2_HYBRID_CLASS_CONFIDENCE_THRESHOLDS_JSON`
- `TD_CASE2_HYBRID_ENABLE_MOTION_TRIGGER`
- `TD_CASE2_HYBRID_ENABLE_ENTRY_TRIGGER`
- `TD_CASE2_HYBRID_ENABLE_OVERLAP_TRIGGER`
- `TD_CASE2_HYBRID_SAVE_VIDEO`
- `TD_CASE2_HYBRID_DEVICE`
- `TD_CASE2_HYBRID_ENTRY_ZONES_JSON`
- `TD_CASE2_HYBRID_ENTRY_ZONES_FILE`

## CLI Example

```powershell
python tests/td_case2/hybrid_tracking_test/run_hybrid_tracking_test.py `
  --video-path "C:\path\video.mp4" `
  --run-dir "C:\path\debug_run" `
  --processing-fps 10 `
  --yolo-interval-frames 3 `
  --max-yolo-gap-seconds 0.5 `
  --device cuda `
  --save-annotated-video
```

## Output Files

Outputs are written to:

```text
<TD_CASE2_RUN_DIR>\hybrid_tracking_test\
```

Files:

- `04c_hybrid_tracks.json`
- `04c_hybrid_tracking_report.json`
- `04c_hybrid_tracking_timing.json`
- `04c_hybrid_tracking_events.json`
- `04c_hybrid_track_summary.json`
- `04c_hybrid_annotated_video.mp4`
- `04c_hybrid_config.json`
- `04c_hybrid_failures.json`
- `04c_hybrid_frame_metrics.json`
- `04c_hybrid_comparison_report.json`
- `04c_hybrid_comparison_report.md`
- `04d_reconciled_tracks.json`
- `04d_track_merge_events.json`
- `04d_track_reconciliation_report.json`
- `04d_track_reconciliation_report.md`
- `04e_before_after_duplication_comparison.json`
- `04e_before_after_duplication_comparison.md`

## Annotated Video Inspection

The annotated MP4 overlays:

- track ID
- class name
- source: `YOLO` or `KCF`
- lifecycle state such as `confirmed`, `propagated`, or `reactivated`
- seconds since last detector confirmation
- frame-level trigger summary

Reactivated tracks are labeled as `REACTIVATED ID <n>` to make duplication inspection easier.

## Baseline Comparison

Run:

```powershell
python tests/td_case2/hybrid_tracking_test/compare_tracking_results.py `
  --run-dir "C:\path\debug_run"
```

It compares the baseline tracking output found in the run directory against `04c_hybrid_tracks.json`.

## Fragment Reconciliation

Run:

```powershell
python tests/td_case2/hybrid_tracking_test/track_fragment_reconciliation.py `
  --run-dir "C:\path\debug_run"
```

This stage preserves the raw `04c` track IDs and writes `04d_*` outputs containing a reconciled physical-object estimate.

## Before/After Duplication Comparison

Run:

```powershell
python tests/td_case2/hybrid_tracking_test/compare_duplication_runs.py `
  --before-run-dir "C:\path\previous_run" `
  --after-run-dir "C:\path\new_run"
```

It writes `04e_before_after_duplication_comparison.json` and `.md` in the newer run directory.

## Recommended July 17, 2026 PowerShell Rerun

```powershell
$env:TD_CASE2_RUN_DIR="C:\Users\PC\mk\Final_vedio_Ai_system\debug_runs\hybrid_test_run_v2"
$env:TD_CASE2_VIDEO_PATH="C:\Users\PC\Downloads\Untitled design.mp4"

$env:TD_CASE2_HYBRID_DETECTION_CONFIDENCE="0.35"
$env:TD_CASE2_HYBRID_MIN_IOU_MATCH="0.20"
$env:TD_CASE2_HYBRID_MAX_MISSED_REFRESHES="8"
$env:TD_CASE2_HYBRID_MAX_TRACK_IDLE_SECONDS="2.0"
$env:TD_CASE2_HYBRID_MIN_TRACK_HITS="3"

$env:TD_CASE2_HYBRID_MOTION_MIN_AREA_RATIO="0.006"
$env:TD_CASE2_HYBRID_MOTION_PERSISTENCE_FRAMES="3"
$env:TD_CASE2_HYBRID_MOTION_TRACK_REGION_EXPANSION="0.30"

$env:TD_CASE2_HYBRID_LOST_RECOVERY_SECONDS="2.0"
$env:TD_CASE2_HYBRID_EMPTY_SCENE_YOLO_INTERVAL_SECONDS="0.5"

python tests/td_case2/hybrid_tracking_test/run_hybrid_tracking_test.py
```

## Known Limitations

- No ground-truth metrics such as MOTA, HOTA, or IDF1 are computed.
- The current appearance cue is lightweight only.
- KCF propagation can still drift during long occlusions or abrupt scale changes.
- The comparison report uses heuristics for fragmentation and coverage when annotations are unavailable.

## Recommended Test Videos

- fixed-camera traffic CCTV with vehicles entering from frame borders
- office hallway footage with doorway entries
- scenes with moderate overlap or short occlusion

## Disabling Individual Adaptive Triggers

- motion trigger: `--disable-motion-trigger`
- entry trigger: `--disable-entry-zone-trigger`
- overlap trigger: `--disable-overlap-trigger`

## Scheduled Refresh Only

Disable motion, entry-zone, and overlap triggers while keeping validation-driven safety refreshes active:

```powershell
python tests/td_case2/hybrid_tracking_test/run_hybrid_tracking_test.py `
  --disable-motion-trigger `
  --disable-entry-zone-trigger `
  --disable-overlap-trigger
```

## Comparison Report Interpretation

- Fewer YOLO calls alone do not prove improvement.
- Review the annotated video alongside `04c_hybrid_comparison_report.md`.
- Treat fragmentation and identity fields as heuristics unless you also have hand-labeled annotations.

## Recommended Test Sequence

### Test 1: scheduled refresh only

- motion trigger off
- entry-zone trigger off
- overlap trigger off
- validation safety retained

Purpose:

- verify YOLO-to-KCF correction
- verify stable IDs
- verify KCF initialization and propagation

### Test 2: motion-trigger test

Enable uncovered-motion detection and measure:

- motion start timestamp
- emergency YOLO timestamp
- first object detection timestamp
- new-object detection delay

### Test 3: overlap test

Use footage with:

- vehicle overlap
- person crossing
- temporary occlusion

### Test 4: baseline comparison

Run the baseline tracker and this hybrid experiment on the same run/video, then compare:

- runtime
- YOLO calls
- fragmentation heuristics
- track duration
- visible object coverage
- annotated videos
