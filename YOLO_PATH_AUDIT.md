# YOLO Path Audit

Date: 2026-06-25

## Summary

Root cause: **A) Current Working Directory differs from Project Root. Relative path therefore fails.**

`YOLO("yolo11m.pt")` resolves relative to the process current working directory, not relative to `detector.py` or the repository root. In the audited run, the process cwd was `C:\Mukul K\vinfo1`, so Ultralytics looked for `C:\Mukul K\vinfo1\yolo11m.pt`. That file does not exist. The real local weights file is `C:\Mukul K\vinfo1\video-search-engine\yolo11m.pt`.

Because the relative file was missing from cwd, Ultralytics treated `yolo11m.pt` as a known model asset name and attempted to download it from GitHub.

## Step 1: Current Working Directory

Diagnostic output before `YOLO("yolo11m.pt")`:

```text
============================================================
Current Working Directory: C:\Mukul K\vinfo1
Resolved Path: C:\Mukul K\vinfo1\yolo11m.pt
Exists: False
============================================================
```

## Step 2: Project Root

`Path(__file__).resolve()` for the detector module:

```text
C:\Mukul K\vinfo1\video-search-engine\app\services\object_detection\detector.py
```

Parent audit:

```text
parents[1]: C:\Mukul K\vinfo1\video-search-engine\app\services
parents[2]: C:\Mukul K\vinfo1\video-search-engine\app
parents[3]: C:\Mukul K\vinfo1\video-search-engine
parents[4]: C:\Mukul K\vinfo1
```

The directory that contains `yolo11m.pt` is:

```text
C:\Mukul K\vinfo1\video-search-engine
```

Correct project root:

```text
C:\Mukul K\vinfo1\video-search-engine
```

This also matches `app.core.config.PROJECT_ROOT`.

## Step 3: Path Comparison

Current Working Directory:

```text
C:\Mukul K\vinfo1
```

Project Root:

```text
C:\Mukul K\vinfo1\video-search-engine
```

They are **not identical**.

Yes, `YOLO("yolo11m.pt")` is searching the wrong directory in this execution context because it resolves against cwd.

## Step 4: Absolute Path Verification

Absolute path expression from `detector.py`:

```python
MODEL_PATH = (
    Path(__file__)
    .resolve()
    .parents[3]
    / "yolo11m.pt"
)
```

Observed values:

```text
MODEL_PATH: C:\Mukul K\vinfo1\video-search-engine\yolo11m.pt
MODEL_PATH.exists(): True
MODEL_PATH.size: 40684120
```

## Step 5: Direct Load Test

Test:

```python
YOLO(str(MODEL_PATH))
```

Result: **A) Load immediately**

Observed:

```json
{
  "outcome": "loaded",
  "elapsed_seconds": 0.1374,
  "exception_type": null,
  "exception": null
}
```

## Step 6: Relative Path Test

Test:

```python
YOLO("yolo11m.pt")
```

Result: failed and attempted GitHub download.

Observed:

```json
{
  "outcome": "failed",
  "exception_type": "ConnectionError",
  "exception": "Download failure for https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt. Retry limit reached. Curl return value 7"
}
```

Difference:

- `YOLO(str(MODEL_PATH))` loads the existing local file.
- `YOLO("yolo11m.pt")` searches `C:\Mukul K\vinfo1\yolo11m.pt`, does not find it, and then Ultralytics attempts to download the named asset.

## Step 7: Root Cause

**A) Current Working Directory differs from Project Root. Relative path therefore fails.**

Evidence:

- Cwd is `C:\Mukul K\vinfo1`.
- Project root is `C:\Mukul K\vinfo1\video-search-engine`.
- `Path("yolo11m.pt").resolve()` is `C:\Mukul K\vinfo1\yolo11m.pt`.
- `Path("yolo11m.pt").exists()` is `False`.
- Absolute `MODEL_PATH` is `C:\Mukul K\vinfo1\video-search-engine\yolo11m.pt`.
- `MODEL_PATH.exists()` is `True`.
- `YOLO(str(MODEL_PATH))` loads successfully.
- `YOLO("yolo11m.pt")` attempts GitHub download.

This rules out corrupted weights and rules out Ultralytics ignoring a valid local file.

## Step 8: Production Recommendation

If changing production code after this audit, replace:

```python
self._model = YOLO("yolo11m.pt")
```

with a portable absolute path derived from the project root:

```python
MODEL_PATH = (
    Path(__file__)
    .resolve()
    .parents[3]
    / "yolo11m.pt"
)

self._model = YOLO(str(MODEL_PATH))
```

Do not hardcode `C:\...`.

An equivalent repo-convention option is to import `PROJECT_ROOT` from `app.core.config` and use:

```python
MODEL_PATH = PROJECT_ROOT / "yolo11m.pt"
self._model = YOLO(str(MODEL_PATH))
```

The audited parent level for `detector.py` is `parents[3]`.
