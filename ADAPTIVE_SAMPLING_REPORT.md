# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `cba6a2be-d66b-4650-a8c0-2cf64aed9e6a`
* **Original Frame Count (Extracted)**: 10
* **Filtered Frame Count (Sent to Qwen)**: 10
* **Frames Skipped**: 0
* **Frame Reduction Ratio**: 0.00%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: 3.63 seconds
* **Actual Pipeline Run Duration (with sampling)**: 23.84 seconds
* **Projected Run Duration Without Sampling**: 23.84 seconds
* **Estimated Runtime Savings**: 0.00 seconds (0.00 minutes)
* **Actual Runtime Savings**: 0.00 seconds (0.00 minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `True`
* **SSIM_THRESHOLD**: `0.92`
* **HISTOGRAM_THRESHOLD**: `0.25`
* **MOTION_THRESHOLD**: `0.15`
