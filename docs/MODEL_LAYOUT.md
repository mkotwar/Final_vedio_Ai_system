# Model Layout

This repository stores local model assets under the repository-root `models/` directory.

That asset folder is different from the multicamera pipeline runtime Python package:

- Asset folder: `models/`
- Runtime package: `tests/td_case2/multicamera_vehicle_tracking_pipeline/models/`

Do not commit model weights.

## Required Structure

```text
models/
  vehicle_detection/
    best_old.pt
  plate_detection/
    license_plate_weights.pt
  florence/
    Florence-2-base-ft/
      config.json
      configuration_florence2.py
      modeling_florence2.py
      processing_florence2.py
      preprocessor_config.json
      tokenizer.json
      tokenizer_config.json
      vocab.json
      model.safetensors
      pytorch_model.bin
  florence_adapters/
    adaptor_florance_baseFT/
      adapter_config.json
      adapter_model.safetensors
      added_tokens.json
      merges.txt
      preprocessor_config.json
      processor_config.json
      special_tokens_map.json
      tokenizer.json
      tokenizer_config.json
      vocab.json
```

## Resolution Precedence

1. CLI override
2. Environment variable
3. YAML config value
4. Project-relative default under `models/`
5. Clear missing-path error

## Environment Variables

- `VEHICLE_DETECTOR_MODEL_PATH`
- `PLATE_DETECTOR_MODEL_PATH`
- `FLORENCE_MODEL_PATH`
- `FLORENCE_PROCESSOR_PATH`
- `FLORENCE_ADAPTER_PATH`

## Project Defaults

- `models/vehicle_detection/best_old.pt`
- `models/plate_detection/license_plate_weights.pt`
- `models/florence/Florence-2-base-ft`
- `models/florence_adapters/adaptor_florance_baseFT`

## Git Ignore Rules

The repository ignores local model weights and Hugging Face cache content under the root `models/` folder. The runtime Python package remains trackable.

## Verification

Filesystem-only check:

```powershell
& ".\tests\td_case2\.venv\Scripts\python.exe" -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.check_model_readiness
```

Minimal load check:

```powershell
& ".\tests\td_case2\.venv\Scripts\python.exe" -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.check_model_readiness --load-check
```

## Another PC Setup

1. Copy the required local model assets into the root `models/` layout.
2. Keep weights out of Git.
3. Optionally set the environment variables if you need non-default locations.
4. Run the readiness checker before any smoke test.

## Warning

Do not switch runtime paths back to these legacy locations:

- `object/vehical_detection`
- `ocr_colour`
- `C:\Mukul K\models`
- `OCR_MUKUL`
- `tests/td_case2/local_models`

Those may remain as duplicate copies until cleanup is explicitly approved.
