# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `521e4881-b614-43c9-a3b4-ffc4fbaa2de6`
* **Original Frame Count (Extracted)**: 25
* **Filtered Frame Count (Sent to Qwen)**: 25
* **Frames Skipped**: 0
* **Frame Reduction Ratio**: 0.00%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: 8.36 seconds
* **Actual Pipeline Run Duration (with sampling)**: 183.86 seconds
* **Projected Run Duration Without Sampling**: 183.86 seconds
* **Estimated Runtime Savings**: 0.00 seconds (0.00 minutes)
* **Actual Runtime Savings**: 0.00 seconds (0.00 minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `True`
* **SSIM_THRESHOLD**: `0.92`
* **HISTOGRAM_THRESHOLD**: `0.25`
* **MOTION_THRESHOLD**: `0.15`
