# Step 04B ByteTrack Experiment

This folder contains an isolated tracking experiment for `tests/td_case2`.

It keeps the active pipeline unchanged and writes all real-run outputs under:

`<run_dir>/tracking_experiment/`

The experiment does three things:

1. Builds a dense tracking-only frame branch from the original video.
2. Re-runs YOLO on those dense frames and feeds the detections into Ultralytics ByteTrack.
3. Applies a conservative post-merge stage, then converts the result back into the important `04B_tracks.json` schema expected by Step 05.

Run from the repository root with:

```powershell
tests\td_case2\.venv\Scripts\python.exe tests/td_case2/experiments/step04b_bytetrack/run_tracking_experiment.py
```

Optional environment overrides:

```text
TD_CASE2_EXP_TRACKING_FPS=5.0
TD_CASE2_EXP_TRACK_BUFFER_SECONDS=2.0
TD_CASE2_EXP_TRACK_HIGH_CONFIDENCE=0.25
TD_CASE2_EXP_TRACK_LOW_CONFIDENCE=0.10
TD_CASE2_EXP_TRACK_MATCH_THRESHOLD=0.80
TD_CASE2_EXP_TRACK_MIN_LENGTH=2
```
