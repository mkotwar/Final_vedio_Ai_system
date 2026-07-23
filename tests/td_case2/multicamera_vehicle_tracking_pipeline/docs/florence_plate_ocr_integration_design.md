## Florence Plate OCR Integration Design

### Files Inspected

- `OCR_MUKUL/anpr_frog_speed.py`
- `OCR_MUKUL/adaptor_florance_baseFT/README.md`
- `OCR_MUKUL/adaptor_florance_baseFT/processing_florence2.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/models/model_path_resolver.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/models/florence_runtime.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/models/florence_runtime_factory.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/evidence/evidence_models.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/tracking/tracking_models.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/persistence_worker.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/vehicle_colour_worker.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/workers/worker_supervisor.py`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/010_track_media.sql`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/011_vehicle_attribute.sql`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/012_plate_detection.sql`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/013_plate_reading.sql`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/database/migrations/014_plate_summary.sql`
- `tests/td_case2/multicamera_vehicle_tracking_pipeline/database/supabase/analytics_full_schema.sql`

### Exact Reusable Plate-Detector Code

Found in `OCR_MUKUL/anpr_frog_speed.py`:

- loader import: `from ultralytics import YOLO`
- model path variable: `PLATE_MODEL_PATH = r"license_plate_weights.pt"`
- threshold variable: `PLATE_CONFIDENCE_THRESHOLD = 0.5`
- inference pattern:
  - `plate_model = YOLO(PLATE_MODEL_PATH)` in the runtime section
  - per-vehicle crop inference through `plate_model(vehicle_roi, conf=PLATE_CONFIDENCE_THRESHOLD, verbose=False)`
- best-box selection pattern:
  - `best_plate = max(plate_results[0].boxes, key=lambda b: b.conf[0])`
- detection payload shape from Ultralytics:
  - `boxes.xyxy`
  - `boxes.conf`
  - optional class id through `boxes.cls`

Plate model type:

- Ultralytics YOLO model file
- expected current repository-relative filename: `license_plate_weights.pt`

Expected detector input:

- OpenCV BGR image array of a vehicle crop, not a full frame path

Expected detector output:

- one or more plate boxes in XYXY coordinates
- confidence per box
- no current custom class mapping beyond the detector class ids

Observed class assumptions:

- repository notes and prior validation logs indicate class `0` corresponds to a license plate
- current OCR_MUKUL code does not enforce a richer multi-class plate taxonomy

### Existing Plate-Crop Logic

Current OCR_MUKUL crop flow:

1. detect a vehicle on the frame
2. extract `vehicle_roi`
3. run the plate detector on `vehicle_roi`
4. choose the highest-confidence plate box
5. crop the plate from the vehicle ROI
6. optionally preprocess before OCR

Observed crop-related helpers:

- `resize_proportionally_if_needed(...)`
- preprocessing path before OCR:
  - YCrCb equalization
  - Gaussian blur

Current OCR_MUKUL code is optimized for one chosen crop, not for bounded candidate search across multiple evidence roles.

### Exact Florence OCR Behavior Found

Reusable OCR helper in `OCR_MUKUL/anpr_frog_speed.py`:

- `run_florence_inference(image_cv, task_prompt, text_input=None, use_adapter=True)`

Exact OCR task token currently used:

- `<OCR>`

Exact OCR invocation currently used:

- `run_florence_inference(plate_crop, "<OCR>", use_adapter=True)`
- result consumed with `res.get("<OCR>", "NOT_FOUND")`

Current helper behavior:

- converts BGR OpenCV image to PIL RGB
- builds prompt as `task_prompt + text_input` if text input exists
- calls:
  - `processor(text=prompt, images=image_pil, return_tensors="pt").to(device)`
  - `model.generate(..., max_new_tokens=1024, do_sample=False, num_beams=3, use_cache=False)`
  - `processor.batch_decode(...)`
  - `processor.post_process_generation(generated_text, task=task_prompt, image_size=(width, height))`
- returns parsed Florence output dictionary keyed by task token

Adapter-loading behavior in OCR_MUKUL:

- `BASE_MODEL_ID = "microsoft/Florence-2-base-ft"`
- `ADAPTER_PATH = r"adaptor_florance_baseFT"`
- `model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(device)`
- `processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)`
- `model = PeftModel.from_pretrained(model, ADAPTER_PATH)`
- `model.eval()`

Processor behavior from `processing_florence2.py`:

- `<OCR>` maps to `pure_text`
- `<OCR_WITH_REGION>` maps to structured OCR output
- `task_prompts_without_inputs["<OCR>"] = "What is the text in the image?"`
- post-processing returns:
  - `{ "<OCR>": "..." }` for pure text
  - no JSON contract is required

### Existing OCR Normalization and Validation

Found in `OCR_MUKUL/anpr_frog_speed.py`:

- `is_valid_indian_plate(text)`

Current normalization inside validation:

- uppercase
- remove non-alphanumeric characters with `re.sub(r"[^A-Z0-9]", "", text.upper())`

Current supported patterns:

- `^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$`
- `^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$`

Current verification rule:

- length over 10 is rejected
- first pattern requires the first two letters to be in a known India state-code list
- second Bharat-series pattern is accepted directly

Observed limitation:

- current validator is a useful base, but too narrow for final runtime policy if we want broader Indian format support and verified vs unverified separation

### Existing Fallback OCR Behavior

No robust multi-candidate fallback architecture exists in the active pipeline yet.

In OCR_MUKUL, fallback behavior is minimal:

- detect best plate once
- OCR that chosen crop
- optional preprocess path

There is no current bounded search across:

- `highest_confidence`
- `sharpest`
- `best_overall`
- `largest`
- `middle`
- `first`
- `last`

### Existing Hardcoded Paths Found

Hardcoded or machine-coupled values in `OCR_MUKUL/anpr_frog_speed.py`:

- `VIDEO_PATH = r"D:\\FrogCeLL\\Videos\\ANPR + speed\\RAW1.mp4"`
- `VEHICLE_MODEL_PATH = r"best_old.pt"`
- `PLATE_MODEL_PATH = r"license_plate_weights.pt"`
- `ADAPTER_PATH = r"adaptor_florance_baseFT"`
- `BASE_MODEL_ID = "microsoft/Florence-2-base-ft"`

These must not be copied into the active pipeline.

### Dependencies

Observed direct dependencies for reusable ANPR pieces:

- `ultralytics`
- `torch`
- `transformers`
- `peft`
- `Pillow`
- `opencv-python`
- `numpy`

### Code Coupled to Tracking or Speed

`OCR_MUKUL/anpr_frog_speed.py` is strongly coupled to unrelated concerns:

- video reading
- manual line GUI
- `supervision.ByteTrack`
- speed calculation
- thread queues
- CSV logging
- global mutable state

These must not be imported into the active pipeline runtime.

### Reusable Components

Safe to wrap or adapt:

- Florence load pattern
- Florence OCR task contract using `<OCR>`
- PEFT adapter loading pattern
- plate detector load pattern using Ultralytics YOLO
- Indian plate state-code list
- baseline normalization and format-validation logic
- basic plate image preprocessing ideas

Not safe to reuse directly:

- `anpr_frog_speed.py` thread orchestration
- GUI line setup
- speed-estimation logic
- video scanning logic
- global queue state
- CSV persistence

### Current Active Runtime Flow

Current live pipeline flow:

1. camera reader emits frames
2. shared detector produces vehicle detections
3. per-camera ByteTrack state builds `LocalVehicleTrack`
4. `TrackEvidenceCollector` stores bounded evidence candidates
5. completed track enters `PersistenceWorker`
6. persistence writes `vehicle_track`, `track_observation`, and selected `track_media`
7. `VehicleColourWorker` receives persisted completed tracks for bounded Florence colour enrichment

Relevant current runtime objects:

- completed track type: `tracking_models.LocalVehicleTrack`
- evidence container: `evidence_models.TrackEvidencePackage`
- saved vehicle evidence candidates:
  - `best_overall`
  - `highest_confidence`
  - `largest`
  - `sharpest`
  - `first`
  - `middle`
  - `last`

This means the best ANPR integration point is after completed-track persistence, reusing the saved local vehicle evidence files and the same per-track isolation model used by vehicle colour.

### Analytics Schema Inspection Result

Important correction: the analytics schema already has dedicated ANPR tables.

Relevant tables:

- `analytics.track_media`
- `analytics.plate_detection`
- `analytics.plate_reading`
- `analytics.plate_summary`

#### `analytics.track_media`

Relevant columns:

- `id`
- `vehicle_track_id`
- `media_type`
- `storage_uri`
- `frame_number`
- `captured_at`
- `video_time_seconds`
- `bbox`
- `width`
- `height`
- `quality_score`
- `sharpness_score`
- `selection_rank`
- `is_primary`
- `metadata`

Allowed media types include:

- `BEST_VEHICLE_CROP`
- `PLATE_CROP`

Conclusion:

- plate crop references already fit `analytics.track_media`
- no migration is needed for plate-media storage

#### `analytics.plate_detection`

Relevant columns:

- `vehicle_track_id`
- `track_observation_id`
- `track_media_id`
- `detected_at`
- `frame_number`
- `bbox_x1`
- `bbox_y1`
- `bbox_x2`
- `bbox_y2`
- `confidence`
- `detector_name`
- `detector_version`
- `metadata`

Foreign keys:

- `vehicle_track_id -> analytics.vehicle_track(id)`
- `track_observation_id -> analytics.track_observation(id)`
- `track_media_id -> analytics.track_media(id)`

Conclusion:

- selected plate detections can be linked both to the vehicle track and the chosen `PLATE_CROP` media row

#### `analytics.plate_reading`

Relevant columns:

- `plate_detection_id`
- `ocr_engine`
- `ocr_version`
- `raw_text`
- `normalized_text`
- `plate_pattern`
- `confidence`
- `status`
- `is_selected`
- `metadata`

Allowed statuses:

- `VERIFIED`
- `PROBABLE`
- `PARTIAL`
- `UNKNOWN`

Conclusion:

- raw OCR and normalized OCR are already represented explicitly
- verified vs weak/unverified output belongs here, not in `vehicle_attribute`

#### `analytics.plate_summary`

Relevant columns:

- `vehicle_track_id`
- `selected_plate_reading_id`
- `canonical_plate`
- `plate_pattern`
- `status`
- `confidence`
- `reading_count`

Important constraint:

- unique `(vehicle_track_id)`

Conclusion:

- final per-track selected registration value belongs in `plate_summary`
- this is the right searchable canonical result

### Schema Fit Decision

Best representation with the existing schema:

1. persist selected best plate crop in `analytics.track_media` as `PLATE_CROP`
2. persist detector result in `analytics.plate_detection`
3. persist one or more OCR attempts in `analytics.plate_reading`
4. persist final chosen plate in `analytics.plate_summary`

`analytics.vehicle_attribute` should remain for colour and other non-plate attributes.

No ANPR schema migration is currently required.

### `model_registry` Name Check

The prompt referenced `analytics.model_registry`, but the actual current schema in this pipeline uses:

- `analytics.ai_model`
- `analytics.run_model`

ANPR model audit should therefore reuse the existing `ai_model` and `run_model` pattern rather than inventing a new `model_registry` table name.

### Minimum Refactor Required

Minimum safe ANPR additions to the active pipeline:

- portable ANPR config
- shared plate-detector runtime wrapper
- shared Florence OCR extractor using the existing runtime
- bounded vehicle-evidence selector
- bounded plate-candidate collector
- deterministic best-plate selector
- plate-text normalizer
- Indian registration validator
- ANPR enrichment service
- one bounded ANPR worker after persistence
- repositories for:
  - `track_media` reuse for `PLATE_CROP`
  - `plate_detection`
  - `plate_reading`
  - `plate_summary`

### Recommended Integration Point

Recommended runtime order:

1. completed track persisted
2. vehicle colour enrichment and ANPR enrichment consume the same completed track context
3. ANPR reuses saved evidence images only
4. no video rescan
5. no per-frame OCR
6. no model load per track

Preferred model-sharing strategy:

- one Florence runtime instance shared across colour and OCR tasks
- one plate-detector runtime instance shared across ANPR jobs

### Portable Path Resolution Decision

Use the existing `models/model_path_resolver.py` for:

- `PLATE_DETECTOR_MODEL_PATH`
- `FLORENCE_MODEL_PATH`
- `FLORENCE_ADAPTER_PATH`
- `FLORENCE_PROCESSOR_PATH`

Required precedence:

1. CLI
2. environment variable
3. YAML
4. project-relative path
5. clear error

No resolved model path may be written into database payloads.

### Final Design Conclusion

The current active pipeline already has the right completed-track and bounded-evidence architecture for ANPR.

The clean implementation path is:

- reuse OCR_MUKUL detector and Florence OCR behavior
- do not reuse OCR_MUKUL tracking/thread/video code
- search a bounded set of saved vehicle evidence crops
- detect bounded plate candidates
- OCR only plate crops
- persist plate media and plate text through the dedicated analytics plate tables
- keep image files under `artifacts`
- keep dry-run inference real but database writes disabled
