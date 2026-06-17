# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `a8c6b594-42a0-485a-bcf3-ccb8fda0f844`
* **Original Frame Count (Extracted)**: 20
* **Filtered Frame Count (Sent to Qwen)**: 5
* **Frames Skipped**: 15
* **Frame Reduction Ratio**: 75.00%

## Filter Breakdown

* **Dropped by Motion Windowing**: 0
* **Dropped by SSIM Threshold**: 0
* **Dropped by Histogram Correlation**: 15
* **Dropped by Motion Score Threshold**: 0

## Dynamic FPS Telemetry

* **Average Extraction Rate**: 2.00 FPS
* **State Transitions**: 0
* **Burst Activations**: 0

### Time in FPS Modes

* **IDLE (0.1 FPS)**: 0.0s
* **LOW_ACTIVITY (0.5 FPS)**: 0.0s
* **NORMAL_ACTIVITY (1.0 FPS)**: 10.0s
* **HIGH_ACTIVITY (2.0 FPS)**: 0.0s
* **BURST_CAPTURE (5.0 FPS)**: 0.0s

## Event Candidate Layer

* **Candidates Evaluated**: 19
* **Candidates Forwarded to VLM**: 5
* **Redundant Frames Dropped**: 15
* **Candidate Reduction**: 78.95%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: 2.65 seconds
* **Actual Pipeline Run Duration (with sampling)**: 15.80 seconds
* **Projected Run Duration Without Sampling**: 55.48 seconds
* **Estimated Runtime Savings**: 39.68 seconds (0.66 minutes)
* **Actual Runtime Savings**: 39.68 seconds (0.66 minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `True`
* **SSIM_THRESHOLD**: `0.92`
* **HISTOGRAM_THRESHOLD**: `0.25`
* **MOTION_THRESHOLD**: `0.15`
