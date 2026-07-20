# Vision Backend Modes

The isolated Step 7 pipeline now supports `auto`, `florence`, `gemini`, and `disabled`.

PowerShell example:

```powershell
$env:TD_CASE2_VISION_BACKEND = "auto"
$env:TD_CASE2_FLORENCE_MODEL_PATH = "C:\Users\PC\mk\models\Florence-2-base-ft"
$env:GEMINI_API_KEY = "your_api_key_here"
$env:TD_CASE2_GEMINI_MODEL = "gemini-2.5-flash"
$env:TD_CASE2_GEMINI_TIMEOUT_SECONDS = "60"
$env:TD_CASE2_GEMINI_MAX_RETRIES = "2"
$env:TD_CASE2_GEMINI_MIN_CONFIDENCE = "0.75"
```

Behavior:

- `auto`: tries Florence first and falls back to Gemini on load/inference failure.
- `florence`: uses only Florence and fails clearly if the local model is unavailable.
- `gemini`: uses only Gemini and does not initialize Florence.
- `disabled`: returns safe empty/model-disabled outputs without crashing the pipeline.

Validation:

```powershell
& .\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.validate_vision_backend
```

Real API validation is opt-in:

```powershell
& .\.venv\Scripts\python.exe -m tests.td_case2.streaming_tracking_pipeline.validate_vision_backend --allow-real-api
```
