# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `1a5b3d35-40ca-4f0f-ac42-c108d2c31161`
* **Original Frame Count (Extracted)**: 121
* **Filtered Frame Count (Sent to Qwen)**: 66
* **Frames Skipped**: 55
* **Frame Reduction Ratio**: 45.45%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: 9.01 seconds
* **Actual Pipeline Run Duration (with sampling)**: 511.07 seconds
* **Projected Run Duration Without Sampling**: 1006.56 seconds
* **Estimated Runtime Savings**: 495.49 seconds (8.26 minutes)
* **Actual Runtime Savings**: 495.49 seconds (8.26 minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `True`
* **SSIM_THRESHOLD**: `0.92`
* **HISTOGRAM_THRESHOLD**: `0.25`
* **MOTION_THRESHOLD**: `0.15`
