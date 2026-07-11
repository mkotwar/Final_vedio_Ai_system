# Qwen API Demo Commands

Search-ready first:

```powershell
cd "C:\Mukul K\vinfo1\video-search-engine"

$env:TD_CASE2_INPUT_VIDEO="C:\Mukul K\delhi_test_vedios\test_anpr_day_10min.mp4"

.\.venv\Scripts\python.exe tests\td_case2\run_td_case2_search_ready_pipeline.py
```

Open UI:

```powershell
$env:TD_CASE2_RUN_DIR="<run_dir printed by search-ready runner>"
.\.venv\Scripts\python.exe -m streamlit run tests\td_case2\td_case2_traffic_search_ui.py
```

Run VLM with Qwen API:

```powershell
$env:TD_CASE2_RUN_DIR="<same run_dir>"
$env:TD_CASE2_VLM_BACKEND="api_qwen"
$env:TD_CASE2_QWEN_API_PROVIDER="openrouter"
$env:TD_CASE2_QWEN_API_KEY="<your_api_key>"
$env:TD_CASE2_QWEN_API_MODEL="qwen/qwen3-vl-8b-instruct"

.\.venv\Scripts\python.exe tests\td_case2\run_td_case2_vlm_event_pipeline.py
```

Run VLM locally:

```powershell
$env:TD_CASE2_RUN_DIR="<same run_dir>"
$env:TD_CASE2_VLM_BACKEND="local_qwen"

.\.venv\Scripts\python.exe tests\td_case2\run_td_case2_vlm_event_pipeline.py
```

Skip VLM:

```powershell
$env:TD_CASE2_RUN_DIR="<same run_dir>"
$env:TD_CASE2_VLM_BACKEND="disabled"

.\.venv\Scripts\python.exe tests\td_case2\run_td_case2_vlm_event_pipeline.py
```
