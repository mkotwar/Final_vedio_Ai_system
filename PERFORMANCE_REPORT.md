# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `d798ad0e-b854-4b39-97d0-5d53d4463f03`
* **Frames Processed**: 5
* **Video Upload Time**: 0.01 seconds
* **Total Ingestion Time**: 52.93 seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | 5524.03 ms | 94.30% |
| **OCR Processing** | 272.93 ms | 4.66% |
| **Frame Extraction** | 58.43 ms | 1.00% |
| **Metadata Write** | 0.32 ms | 0.01% |
| **JSON Repair & Normalization** | 2.11 ms | 0.04% |
| **Metadata Validation** | 0.02 ms | 0.00% |
| **Total Per Frame** | 5857.85 ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

1. **Qwen VLM Inference**: 27.62s total (94.30% of runtime)
2. **OCR Processing**: 1.36s total (4.66% of runtime)
3. **Frame Extraction**: 0.29s total (1.00% of runtime)
4. **JSON Repair**: 0.01s total (0.04% of runtime)
5. **Metadata Write**: 0.00s total (0.01% of runtime)
6. **Metadata Validation**: 0.00s total (0.00% of runtime)

---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `d798ad0e-b854-4b39-97d0-5d53d4463f03_f0018` | 10165.44 | 10044.73 | 52.03 | 68.08 | 0.49 |
| 2 | `d798ad0e-b854-4b39-97d0-5d53d4463f03_f0004` | 4794.97 | 4393.86 | 328.16 | 72.49 | 0.38 |
| 3 | `d798ad0e-b854-4b39-97d0-5d53d4463f03_f0014` | 4789.82 | 4393.86 | 328.16 | 67.52 | 0.22 |
| 4 | `d798ad0e-b854-4b39-97d0-5d53d4463f03_f0010` | 4789.27 | 4393.86 | 328.16 | 66.94 | 0.24 |
| 5 | `d798ad0e-b854-4b39-97d0-5d53d4463f03_f0001` | 4749.74 | 4393.86 | 328.16 | 17.12 | 0.26 |

---

## Recommendations

1. **Optimize Qwen VLM Inference**:
   * Currently, the **Qwen VLM Inference** stage represents the largest performance bottleneck at **94.30%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes 58.43 ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
