# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `a8c6b594-42a0-485a-bcf3-ccb8fda0f844`
* **Frames Processed**: 5
* **Video Upload Time**: 0.00 seconds
* **Total Ingestion Time**: 15.80 seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | 2521.84 ms | 95.34% |
| **OCR Processing** | 60.62 ms | 2.29% |
| **Frame Extraction** | 62.17 ms | 2.35% |
| **Metadata Write** | 0.29 ms | 0.01% |
| **JSON Repair & Normalization** | 0.13 ms | 0.01% |
| **Metadata Validation** | 0.01 ms | 0.00% |
| **Total Per Frame** | 2645.07 ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

1. **Qwen VLM Inference**: 12.61s total (95.34% of runtime)
2. **Frame Extraction**: 0.31s total (2.35% of runtime)
3. **OCR Processing**: 0.30s total (2.29% of runtime)
4. **Metadata Write**: 0.00s total (0.01% of runtime)
5. **JSON Repair**: 0.00s total (0.01% of runtime)
6. **Metadata Validation**: 0.00s total (0.00% of runtime)

---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `a8c6b594-42a0-485a-bcf3-ccb8fda0f844_f0017` | 4582.02 | 4439.98 | 69.68 | 71.86 | 0.33 |
| 2 | `a8c6b594-42a0-485a-bcf3-ccb8fda0f844_f0007` | 2175.65 | 2042.31 | 58.35 | 74.66 | 0.20 |
| 3 | `a8c6b594-42a0-485a-bcf3-ccb8fda0f844_f0012` | 2173.51 | 2042.31 | 58.35 | 72.49 | 0.25 |
| 4 | `a8c6b594-42a0-485a-bcf3-ccb8fda0f844_f0004` | 2171.78 | 2042.31 | 58.35 | 70.65 | 0.30 |
| 5 | `a8c6b594-42a0-485a-bcf3-ccb8fda0f844_f0001` | 2122.40 | 2042.31 | 58.35 | 21.20 | 0.37 |

---

## Recommendations

1. **Optimize Qwen VLM Inference**:
   * Currently, the **Qwen VLM Inference** stage represents the largest performance bottleneck at **95.34%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes 62.17 ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
