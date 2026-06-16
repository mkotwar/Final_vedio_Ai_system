# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `457f33c8-0fed-4fdd-badb-a6dfed341981`
* **Original Frame Count (Extracted)**: 31
* **Filtered Frame Count (Sent to Qwen)**: 5
* **Frames Skipped**: 26
* **Frame Reduction Ratio**: 83.87%

## Filter Breakdown

* **Dropped by Motion Windowing**: 12
* **Dropped by SSIM Threshold**: 6
* **Dropped by Histogram Correlation**: 8
* **Dropped by Motion Score Threshold**: 0

## Dynamic FPS Telemetry

* **Average Extraction Rate**: 1.03 FPS
* **State Transitions**: 7
* **Burst Activations**: 1

### Time in FPS Modes

* **IDLE (0.1 FPS)**: 10.0s
* **LOW_ACTIVITY (0.5 FPS)**: 4.0s
* **NORMAL_ACTIVITY (1.0 FPS)**: 11.0s
* **HIGH_ACTIVITY (2.0 FPS)**: 0.0s
* **BURST_CAPTURE (5.0 FPS)**: 3.0s

## Event Candidate Layer

* **Candidates Evaluated**: 18
* **Candidates Forwarded to VLM**: 5
* **Redundant Frames Dropped**: 14
* **Candidate Reduction**: 77.78%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: 53.11 seconds
* **Actual Pipeline Run Duration (with sampling)**: 275.85 seconds
* **Projected Run Duration Without Sampling**: 1656.76 seconds
* **Estimated Runtime Savings**: 1380.91 seconds (23.02 minutes)
* **Actual Runtime Savings**: 1380.91 seconds (23.02 minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `True`
* **SSIM_THRESHOLD**: `0.92`
* **HISTOGRAM_THRESHOLD**: `0.25`
* **MOTION_THRESHOLD**: `0.15`
