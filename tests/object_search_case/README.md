# Object Search Case

This isolated testcase builds a searchable object index for a video using:

- YOLO for known-class detection
- BoT-SORT for local tracking across sampled frames
- lightweight appearance summaries from crops

It is intentionally separate from production code and the tender-demo pipeline.

## What It Finds

Known YOLO classes only:

- `person`
- `bicycle`
- `car`
- `motorcycle`
- `bus`
- `truck`
- `backpack`
- `handbag`
- `suitcase`

For each tracked object it stores:

- object id
- class name
- first seen / last seen timestamp
- duration
- matched frame timestamps
- best frame path
- best crop path
- simple appearance terms

## Limits

- It does not invent unknown classes.
- It does not do full person ReID across long gaps.
- Appearance is lightweight, based on crop colors and simple bag overlap.
- It is useful for searches like `person`, `white car`, `blue bike`, `person with bag`, `pink upper clothing`.
- Fine-grained pattern queries like `black stripes` may be unreliable in this first version.

## Build Index

Run from the repo root:

```powershell
.\\.venv\\Scripts\\python.exe video-search-engine\\tests\\object_search_case\\run_object_search_case.py `
  --video "C:\\path\\to\\traffic_video.mp4" `
  --sample-every-seconds 1.0 `
  --yolo-model "yolov8n.pt" `
  --yolo-conf 0.25 `
  --yolo-imgsz 640
```

## Split Person + Vehicle + OCR/Color Test

You can also combine all 3 parts together:

- person detection from `Person_detection.pt`
- vehicle detection from your custom vehicle model
- plate OCR + Florence vehicle color on vehicle crops

Example:

```powershell
.\\.venv\\Scripts\\python.exe video-search-engine\\tests\\object_search_case\\run_object_search_case.py `
  --video "C:\\path\\to\\traffic_video.mp4" `
  --sample-every-seconds 1.0 `
  --person-model "C:\\Mukul K\\vinfo1\\video-search-engine\\Person_detection\\Person_detection.pt" `
  --person-conf 0.25 `
  --person-imgsz 640 `
  --vehicle-model "C:\\path\\to\\your\\vehicle_model.pt" `
  --vehicle-conf 0.25 `
  --vehicle-imgsz 640
```

If you want to use the local models we already verified in this repo, you can use the short form:

```powershell
.\\.venv\\Scripts\\python.exe video-search-engine\\tests\\object_search_case\\run_object_search_case.py `
  --video "C:\\path\\to\\traffic_video.mp4" `
  --sample-every-seconds 1.0 `
  --use-local-split-models `
  --person-conf 0.25 `
  --person-imgsz 640 `
  --vehicle-conf 0.25 `
  --vehicle-imgsz 640
```

This uses:

- `video-search-engine\\Person_detection\\Person_detection.pt`
- `video-search-engine\\object_yolo\\best_old.pt`

Notes:

- the person model is automatically restricted to class `0` only
- the vehicle model is restricted to `bicycle`, `car`, `motorcycle`, `bus`, `truck`
- OCR and Florence color are applied only to vehicle detections

Outputs go under:

```text
video-search-engine/tests/object_search_case/debug_runs/<video_name>_<timestamp>/
```

## Search

```powershell
.\\.venv\\Scripts\\python.exe video-search-engine\\tests\\object_search_case\\search_object_index.py `
  --run-dir "C:\\...\\video-search-engine\\tests\\object_search_case\\debug_runs\\my_run" `
  --query "pink person bag"
```

Optional filters:

```powershell
--class-name person
--start-seconds 10
--end-seconds 40
--limit 10
```

## UI Viewer

You can also browse results with images in Streamlit:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run video-search-engine\\tests\\object_search_case\\object_search_ui.py
```

The UI shows:

- run selection
- query and class filters
- time filters
- best frame image
- best crop image
- frame hit table with timestamps
