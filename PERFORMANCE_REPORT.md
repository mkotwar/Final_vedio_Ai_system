# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `457f33c8-0fed-4fdd-badb-a6dfed341981`
* **Frames Processed**: 5
* **Video Upload Time**: 0.00 seconds
* **Total Ingestion Time**: 275.85 seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | 52748.28 ms | 99.32% |
| **OCR Processing** | 339.08 ms | 0.64% |
| **Frame Extraction** | 21.44 ms | 0.04% |
| **Metadata Write** | 0.37 ms | 0.00% |
| **JSON Repair & Normalization** | 2.69 ms | 0.01% |
| **Metadata Validation** | 0.02 ms | 0.00% |
| **Total Per Frame** | 53111.88 ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

1. **Qwen VLM Inference**: 263.74s total (99.32% of runtime)
2. **OCR Processing**: 1.70s total (0.64% of runtime)
3. **Frame Extraction**: 0.11s total (0.04% of runtime)
4. **JSON Repair**: 0.01s total (0.01% of runtime)
5. **Metadata Write**: 0.00s total (0.00% of runtime)
6. **Metadata Validation**: 0.00s total (0.00% of runtime)

---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `457f33c8-0fed-4fdd-badb-a6dfed341981_f0030` | 166297.95 | 166201.06 | 71.78 | 23.98 | 0.71 |
| 2 | `457f33c8-0fed-4fdd-badb-a6dfed341981_f0013` | 24816.42 | 24385.09 | 405.90 | 24.81 | 0.33 |
| 3 | `457f33c8-0fed-4fdd-badb-a6dfed341981_f0021` | 24816.26 | 24385.09 | 405.90 | 24.75 | 0.25 |
| 4 | `457f33c8-0fed-4fdd-badb-a6dfed341981_f0026` | 24814.98 | 24385.09 | 405.90 | 23.53 | 0.24 |
| 5 | `457f33c8-0fed-4fdd-badb-a6dfed341981_f0001` | 24813.80 | 24385.09 | 405.90 | 10.13 | 0.34 |

---

## Recommendations

1. **Optimize Qwen VLM Inference**:
   * Currently, the **Qwen VLM Inference** stage represents the largest performance bottleneck at **99.32%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes 21.44 ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
