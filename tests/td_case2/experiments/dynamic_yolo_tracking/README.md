# Dynamic YOLO Tracking Experiment

This experiment is isolated from the active td_case2 pipeline.

Architecture under test:

```text
Video decode once
-> cheap OpenCV motion gate
-> dynamic frame-rate controller
-> YOLO only on selected frames
-> ByteTrack
-> conservative fragment merging
-> tracking report and preview
```

The real-video runner requires an explicit video path and will not auto-pick an old video:

```powershell
$env:TD_CASE2_DYNAMIC_VIDEO_PATH='C:\path\to\road_video.mp4'
tests\td_case2\.venv\Scripts\python.exe tests/td_case2/experiments/dynamic_yolo_tracking/run_dynamic_tracking_experiment.py
```

Optional legacy fixed-FPS comparison passes are disabled by default and can be enabled explicitly:

```powershell
tests\td_case2\.venv\Scripts\python.exe tests/td_case2/experiments/dynamic_yolo_tracking/run_dynamic_tracking_experiment.py --with-fixed-comparisons
```

Useful environment variables:

```text
TD_CASE2_DYNAMIC_VIDEO_PATH
TD_CASE2_DYNAMIC_OUTPUT_ROOT
TD_CASE2_DYNAMIC_TRACK_BUFFER_SECONDS
TD_CASE2_DYNAMIC_TRACK_HIGH_CONFIDENCE
TD_CASE2_DYNAMIC_TRACK_LOW_CONFIDENCE
TD_CASE2_DYNAMIC_TRACK_MATCH_THRESHOLD
TD_CASE2_DYNAMIC_TRACK_MIN_LENGTH
TD_CASE2_DYNAMIC_SAVE_ANNOTATED
TD_CASE2_DYNAMIC_SAVE_CROPS
TD_CASE2_EXP_EMPTY_HEARTBEAT_SECONDS
TD_CASE2_PERSON_YOLO_MODEL_PATH
TD_CASE2_OBJECT_YOLO_MODEL_PATH
TD_CASE2_YOLO_MODEL_PATH
TD_CASE2_YOLO_DEVICE
```

The runner creates a new run folder under `debug_runs/` and writes:

```text
<run_dir>/dynamic_yolo_tracking_experiment/
```

Default single-pass outputs:

```text
single_pass_dynamic_frames.json
single_pass_dynamic_yolo_detections.json
single_pass_dynamic_tracks_raw.json
single_pass_dynamic_tracks.json
single_pass_dynamic_state_transitions.json
single_pass_dynamic_report.json
single_pass_dynamic_preview.mp4
```

The experiment does not run OCR, Florence, event detection, or VLM.
