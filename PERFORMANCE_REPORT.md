# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a`
* **Frames Processed**: 10
* **Video Upload Time**: 0.00 seconds
* **Total Ingestion Time**: 23.84 seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | 3520.08 ms | 96.91% |
| **OCR Processing** | 67.52 ms | 1.86% |
| **Frame Extraction** | 43.82 ms | 1.21% |
| **Metadata Write** | 0.68 ms | 0.02% |
| **JSON Repair & Normalization** | 0.31 ms | 0.01% |
| **Metadata Validation** | 0.03 ms | 0.00% |
| **Total Per Frame** | 3632.43 ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

1. **Qwen VLM Inference**: 35.20s total (96.91% of runtime)
2. **OCR Processing**: 0.68s total (1.86% of runtime)
3. **Frame Extraction**: 0.44s total (1.21% of runtime)
4. **Metadata Write**: 0.01s total (0.02% of runtime)
5. **JSON Repair**: 0.00s total (0.01% of runtime)
6. **Metadata Validation**: 0.00s total (0.00% of runtime)

---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0002` | 4601.49 | 4479.28 | 71.29 | 49.19 | 1.40 |
| 2 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0010` | 4332.17 | 4216.62 | 70.72 | 44.02 | 0.51 |
| 3 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0006` | 4331.96 | 4224.55 | 62.14 | 44.42 | 0.51 |
| 4 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0007` | 3836.60 | 3728.56 | 62.14 | 45.16 | 0.46 |
| 5 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0008` | 3832.60 | 3721.89 | 62.14 | 47.82 | 0.45 |
| 6 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0003` | 3831.42 | 3711.52 | 71.29 | 47.72 | 0.50 |
| 7 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0004` | 3811.79 | 3693.23 | 71.29 | 46.51 | 0.46 |
| 8 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0001` | 2747.91 | 2650.85 | 71.29 | 24.59 | 0.73 |
| 9 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0005` | 2502.35 | 2393.49 | 62.14 | 45.53 | 0.81 |
| 10 | `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a_f0009` | 2496.00 | 2380.78 | 70.72 | 43.20 | 0.94 |

---

## Recommendations

1. **Optimize Qwen VLM Inference**:
   * Currently, the **Qwen VLM Inference** stage represents the largest performance bottleneck at **96.91%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes 43.82 ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
