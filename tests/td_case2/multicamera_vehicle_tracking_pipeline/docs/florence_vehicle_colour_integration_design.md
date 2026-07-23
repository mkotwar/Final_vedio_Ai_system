## Florence Vehicle Colour Integration Design

### OCR_MUKUL Files Inspected

- `OCR_MUKUL/anpr_frog_speed.py`
- `OCR_MUKUL/adaptor_florance_baseFT/README.md`
- `OCR_MUKUL/adaptor_florance_baseFT/processing_florence2.py`
- `OCR_MUKUL/adaptor_florance_baseFT/adapter_config.json`
- `OCR_MUKUL/adaptor_florance_baseFT/adapter_model.safetensors`

### Reusable Florence Modules Found

- `OCR_MUKUL/anpr_frog_speed.py`
  Contains the current Florence runtime loading and inference helper.
- `OCR_MUKUL/adaptor_florance_baseFT/processing_florence2.py`
  Local Florence processor implementation used through `trust_remote_code=True`.
- `OCR_MUKUL/adaptor_florance_baseFT`
  PEFT adapter directory with tokenizer and processor assets.

### Reusable Florence Functions Found

- `run_florence_inference(image_cv, task_prompt, text_input=None, use_adapter=True)`
  Current signature in `anpr_frog_speed.py`.
  Behavior:
  - converts OpenCV BGR image to PIL RGB
  - builds prompt as `task_prompt + text_input` when text input is present
  - calls `processor(text=..., images=..., return_tensors="pt").to(device)`
  - uses `torch.no_grad()` when `use_adapter=True`
  - uses `model.disable_adapter()` context when `use_adapter=False`
  - calls `model.generate(...)`
  - decodes with `processor.batch_decode(...)`
  - parses with `processor.post_process_generation(...)`
  - returns parsed Florence output dictionary

### Exact Model-Loading Behavior

Current loading in `anpr_frog_speed.py`:

- `BASE_MODEL_ID = "microsoft/Florence-2-base-ft"`
- `device = "cuda" if torch.cuda.is_available() else "cpu"`
- `model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True, attn_implementation="eager").to(device)`
- `processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)`
- `model = PeftModel.from_pretrained(model, ADAPTER_PATH)`
- `model.eval()`

Important consequences:

- model path is currently not configurable
- processor path is currently not configurable
- adapter path is currently relative and hardcoded
- CUDA/CPU choice is automatic only
- no explicit dtype selection is implemented
- no local-files-only flag is implemented
- no timeout or retry policy is implemented
- model is loaded once in `__main__`, which is good for reuse

### Exact Adapter-Loading Behavior

- `ADAPTER_PATH = r"adaptor_florance_baseFT"`
- adapter is loaded through `PeftModel.from_pretrained(model, ADAPTER_PATH)`
- OCR path uses adapter-enabled inference
- colour path currently disables the adapter via `model.disable_adapter()`

This means the current colour extraction uses the base Florence model, not the LoRA adapter.

### Current Colour Prompt and Output Format

Current colour prompt in `anpr_frog_speed.py`:

- task prompt: `"<VQA>"`
- text input: `"What is the primary color of the vehicle?"`

Current extraction call:

- `run_florence_inference(vehicle_roi_crop, "<VQA>", "What is the primary color of the vehicle?", use_adapter=False)`

Current expected output:

- dictionary access through `res.get('<VQA>', 'N/A')`
- effectively plain text color output such as `"white"` or `"red"`

Current code does not:

- request JSON
- normalize colour labels
- validate confidence
- preserve raw output separately from normalized output

### Current OCR-Related Florence Functions

OCR use in `anpr_frog_speed.py`:

- `run_florence_inference(plate_crop, "<OCR>", use_adapter=True)`
- OCR result accessed with `res.get('<OCR>', 'NOT_FOUND')`

This OCR logic is tightly coupled to plate processing and should not be reused for this task beyond the shared Florence runtime behavior.

### Image Preprocessing Found

For vehicle colour:

- vehicle ROI crop extracted from full frame
- `resize_proportionally_if_needed(vehicle_roi.copy())`

For plate OCR only:

- resize if needed
- YCrCb equalization
- Gaussian blur

Colour extraction currently uses the resized vehicle crop only. No special colour-specific normalization is applied.

### Device Selection and CPU Fallback

Found:

- `device = "cuda" if torch.cuda.is_available() else "cpu"`

Not found:

- explicit CPU fallback after CUDA failure
- retry on OOM
- configurable device override
- configurable dtype

### trust_remote_code Usage

Found in both base model and processor loading:

- `trust_remote_code=True`

This is required by the current OCR_MUKUL Florence loading behavior and should remain configurable rather than hardcoded in new integration code.

### Dependency Requirements Observed

From imports and runtime code:

- `torch`
- `transformers`
- `peft`
- `PIL`
- `cv2`
- `numpy`

The reusable Florence subset for colour extraction specifically depends on:

- `torch`
- `transformers`
- `peft`
- `Pillow`

### Existing Hardcoded Paths Found

Machine-specific or hardcoded runtime paths in `OCR_MUKUL/anpr_frog_speed.py`:

- `VIDEO_PATH = r"D:\FrogCeLL\Videos\ANPR + speed\RAW1.mp4"`
- `VEHICLE_MODEL_PATH = r"best_old.pt"`
- `PLATE_MODEL_PATH = r"license_plate_weights.pt"`
- `ADAPTER_PATH = r"adaptor_florance_baseFT"`

Hardcoded model identifier:

- `BASE_MODEL_ID = "microsoft/Florence-2-base-ft"`

Hardcoded output folders:

- `detection/...`

These are not portable enough for the active pipeline and must not be copied as-is.

### Coupling With OCR

Strong coupling found:

- Florence runtime lives in the same file as OCR, speed, tracking, video reading, GUI, CSV logging, and threading
- OCR and colour share the same helper and globals
- OCR path uses adapter, colour path disables adapter

### Coupling With Tracking / Video / GUI

Strong coupling found in `anpr_frog_speed.py`:

- manual video path setup
- line drawing GUI
- YOLO vehicle detection
- YOLO plate detection
- ByteTrack usage through `supervision`
- multi-thread queue orchestration
- CSV persistence

These concerns must not be imported into the new pipeline runtime.

### Reusable Parts Recommended

Safe reusable parts:

- Florence base model loading pattern
- Florence processor loading pattern
- PEFT adapter loading pattern
- Florence inference pattern using PIL image input
- `disable_adapter()` use for colour VQA if we preserve current behavior

Not safe to reuse directly:

- global variables
- OpenCV video orchestration
- queue/thread logic from OCR_MUKUL
- OCR-specific plate crop preprocessing
- manual GUI setup
- hardcoded paths
- CSV logging

### Recommended Wrapper Location

Recommended new modules inside the active pipeline:

- `models/model_path_resolver.py`
- `config/florence.yaml`
- `config/vehicle_colour.yaml`
- `models/florence_runtime.py`
- `models/florence_runtime_factory.py`
- `enrichment/florence_colour_response_parser.py`
- `enrichment/vehicle_colour_mapping.py`
- `enrichment/vehicle_colour_models.py`
- `enrichment/media_resolver.py`
- `enrichment/florence_vehicle_colour_extractor.py`
- `enrichment/vehicle_colour_enrichment_service.py`
- `persistence/vehicle_colour_repository.py`
- `workers/vehicle_colour_worker.py`

Minimal OCR_MUKUL refactor is preferred only if importing the current Florence helper cleanly is not possible.

### Analytics Schema Inspection Result

Existing table already suitable:

- `analytics.vehicle_attribute`

Actual columns relevant to colour from `011_vehicle_attribute.sql`:

- `id uuid primary key default gen_random_uuid()`
- `vehicle_track_id uuid not null`
- `attribute_scope varchar(30) not null default 'TRACK'`
- `primary_color varchar(50)`
- `secondary_color varchar(50)`
- `color_confidence numeric`
- `vehicle_class varchar(40)`
- `class_confidence numeric`
- `attribute_source varchar(50)`
- `attribute_status varchar(30) not null default 'CURRENT'`
- `observation_count integer not null default 1`
- `metadata jsonb not null default '{}'::jsonb`
- `created_at timestamptz not null default now()`
- `updated_at timestamptz not null default now()`

Relevant constraints:

- foreign key:
  `vehicle_track_id -> analytics.vehicle_track(id)`
- `attribute_scope in ('TRACK', 'GLOBAL')`
- `attribute_status in ('CURRENT', 'HISTORICAL', 'REJECTED')`
- `color_confidence` must be within `[0, 1]` when present
- no direct `track_media_id` column exists
- no unique constraint exists for one colour row per track

Relevant indexes:

- `idx_vehicle_attribute_vehicle_track on (vehicle_track_id, attribute_status)`
- `idx_vehicle_attribute_primary_color on (primary_color)`

Implications:

- no schema migration is required for first-pass per-track colour persistence
- linkage to media must go into `metadata`, because no `track_media_id` column exists
- application-level idempotency will be required

### track_media Linkage Support

`analytics.track_media` contains:

- `id`
- `vehicle_track_id`
- `media_type`
- `storage_uri`
- metadata and quality fields

No foreign key from `vehicle_attribute` to `track_media` exists.

So the new integration should:

- resolve the `BEST_VEHICLE_CROP` row for the track
- use its `storage_uri` as canonical source
- persist `source_media_type`, `source_storage_uri`, and optionally `track_media_id` in `metadata`

### Gaps To Address During Implementation

- portable Florence model-path resolution
- reusable runtime separated from OCR_MUKUL globals
- robust response parsing and colour normalization
- application-level idempotency for `analytics.vehicle_attribute`
- bounded worker integration after persistence stage
- dry-run mode with real inference and zero database writes
