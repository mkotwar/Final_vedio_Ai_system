# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `521e4881-b614-43c9-a3b4-ffc4fbaa2de6`
* **Frames Processed**: 25
* **Video Upload Time**: 0.12 seconds
* **Total Ingestion Time**: 183.98 seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | 5010.17 ms | 59.89% |
| **OCR Processing** | 2895.56 ms | 34.62% |
| **Frame Extraction** | 458.27 ms | 5.48% |
| **Metadata Write** | 0.66 ms | 0.01% |
| **JSON Repair & Normalization** | 0.30 ms | 0.00% |
| **Metadata Validation** | 0.02 ms | 0.00% |
| **Total Per Frame** | 8364.98 ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

1. **Qwen VLM Inference**: 125.25s total (59.89% of runtime)
2. **OCR Processing**: 72.39s total (34.62% of runtime)
3. **Frame Extraction**: 11.46s total (5.48% of runtime)
4. **Metadata Write**: 0.02s total (0.01% of runtime)
5. **JSON Repair**: 0.01s total (0.00% of runtime)
6. **Metadata Validation**: 0.00s total (0.00% of runtime)

---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0002` | 11865.36 | 7889.87 | 3375.21 | 599.51 | 0.45 |
| 2 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0022` | 10270.71 | 6507.51 | 3331.97 | 430.00 | 0.86 |
| 3 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0017` | 10129.49 | 7061.93 | 2563.16 | 502.92 | 1.09 |
| 4 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0014` | 10040.26 | 6178.55 | 3417.22 | 443.45 | 0.78 |
| 5 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0009` | 9371.71 | 6274.60 | 2591.58 | 504.20 | 0.97 |
| 6 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0001` | 9262.66 | 5712.19 | 3375.21 | 173.85 | 0.92 |
| 7 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0006` | 8971.50 | 6149.39 | 2344.69 | 476.58 | 0.58 |
| 8 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0023` | 8757.57 | 5006.60 | 3331.97 | 417.95 | 0.69 |
| 9 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0016` | 8593.06 | 4683.87 | 3417.22 | 490.94 | 0.77 |
| 10 | `521e4881-b614-43c9-a3b4-ffc4fbaa2de6_f0004` | 8588.81 | 4716.12 | 3375.21 | 496.89 | 0.24 |

---

## Recommendations

1. **Optimize Qwen VLM Inference**:
   * Currently, the **Qwen VLM Inference** stage represents the largest performance bottleneck at **59.89%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes 458.27 ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
