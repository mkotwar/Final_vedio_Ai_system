# Full Surveillance Intelligence Benchmark

This benchmark runs the complete experimental surveillance intelligence stack in one pass without modifying production code.

Flow:

1. Stage the input video into the existing production pipeline.
2. Run `FrameExtractionService.extract_frames(video_id)`.
3. Reuse the produced frame metadata and aggregated events.
4. Build actor timelines.
5. Build the evidence graph.
6. Run baseline investigation reasoning.
7. Run retrieval-augmented investigation reasoning.
8. Write a combined final report.

Run with:

```powershell
$env:BENCHMARK_INPUT_VIDEO="C:\path\to\your_video.mp4"
$env:BENCHMARK_VIDEO_ID="manual-test-" + [guid]::NewGuid().ToString()
.\.venv\Scripts\python.exe tests\full_surveillance_intelligence\run_full_surveillance_intelligence_benchmark.py
```

Outputs are written to:

`tests/full_surveillance_intelligence/data/output/`

