# PERFORMANCE REPORT

This report summarizes the performance metrics collected during the video ingestion pipeline.

## Ingestion Overview

* **Video ID**: `1a5b3d35-40ca-4f0f-ac42-c108d2c31161`
* **Frames Processed**: 66
* **Video Upload Time**: 0.05 seconds
* **Total Ingestion Time**: 511.12 seconds

---

## Average Stage Timings

| Stage | Average Time (ms) | Percentage of Runtime |
| :--- | :--- | :--- |
| **Qwen VLM Inference** | 5110.97 ms | 56.73% |
| **OCR Processing** | 3645.08 ms | 40.46% |
| **Frame Extraction** | 251.77 ms | 2.79% |
| **Metadata Write** | 0.70 ms | 0.01% |
| **JSON Repair & Normalization** | 0.43 ms | 0.00% |
| **Metadata Validation** | 0.02 ms | 0.00% |
| **Total Per Frame** | 9008.98 ms | 100.00% |

---

## Bottleneck Ranking

Based on total runtime spent in each stage:

1. **Qwen VLM Inference**: 337.32s total (56.73% of runtime)
2. **OCR Processing**: 240.58s total (40.46% of runtime)
3. **Frame Extraction**: 16.62s total (2.79% of runtime)
4. **Metadata Write**: 0.05s total (0.01% of runtime)
5. **JSON Repair**: 0.03s total (0.00% of runtime)
6. **Metadata Validation**: 0.00s total (0.00% of runtime)

---

## Top 10 Slowest Frames

| Rank | Frame ID | Total Frame Time (ms) | VLM Inference (ms) | OCR Time (ms) | Extract Time (ms) | Write Time (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0001` | 16289.93 | 10769.43 | 5413.52 | 93.21 | 0.50 |
| 2 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0002` | 14113.49 | 8405.15 | 5413.52 | 294.13 | 0.34 |
| 3 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0062` | 11774.26 | 6842.48 | 4681.34 | 249.65 | 0.49 |
| 4 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0094` | 11321.59 | 6222.88 | 4888.94 | 208.83 | 0.88 |
| 5 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0029` | 11071.79 | 7026.50 | 3790.74 | 253.49 | 0.78 |
| 6 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0032` | 10920.87 | 6925.67 | 3760.74 | 233.12 | 0.96 |
| 7 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0072` | 10881.42 | 6431.21 | 4214.05 | 235.31 | 0.51 |
| 8 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0023` | 10593.48 | 5993.98 | 4343.83 | 255.34 | 0.27 |
| 9 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0058` | 10452.13 | 6336.96 | 3400.67 | 713.44 | 0.73 |
| 10 | `1a5b3d35-40ca-4f0f-ac42-c108d2c31161_f0050` | 10434.22 | 7162.27 | 3038.76 | 232.33 | 0.58 |

---

## Recommendations

1. **Optimize Qwen VLM Inference**:
   * Currently, the **Qwen VLM Inference** stage represents the largest performance bottleneck at **56.73%** of the total frame processing time.
   * If VLM is the bottleneck: Consider flash attention, quantizing the model (INT4/INT8), or enabling frame-skipping based on motion thresholding.
   * If OCR is the bottleneck: Optimize easyocr reader parameters, switch to GPU if VRAM allows, or run OCR asynchronously in parallel processes.
2. **Optimize Frame Extraction**:
   * Frame extraction takes 251.77 ms per frame on average. If this is high, utilize faster decoder libraries (like `decord`) or write frames directly to an in-memory byte stream instead of saving JPEGs to disk.
3. **Motion Detection / Pixel-Movement Filtering**:
   * Implement a frame-to-frame pixel difference threshold. For frames below a threshold (no motion), skip OCR, VLM, and JSON writing completely, and clone the previous frame's metadata to save significant compute.
