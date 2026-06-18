# ADAPTIVE SAMPLING REPORT

This report summarizes the performance metrics and compute savings achieved by enabling Adaptive Frame Sampling.

## Ingestion Overview

* **Video ID**: `d798ad0e-b854-4b39-97d0-5d53d4463f03`
* **Original Frame Count (Extracted)**: 18
* **Filtered Frame Count (Sent to Qwen)**: 5
* **Frames Skipped**: 13
* **Frame Reduction Ratio**: 72.22%

## Filter Breakdown

* **Dropped by Motion Windowing**: 0
* **Dropped by SSIM Threshold**: 1
* **Dropped by Histogram Correlation**: 10
* **Dropped by Motion Score Threshold**: 2

## Dynamic FPS Telemetry

* **Average Extraction Rate**: 2.00 FPS
* **State Transitions**: 5
* **Burst Activations**: 0

### Time in FPS Modes

* **IDLE (0.1 FPS)**: 0.0s
* **LOW_ACTIVITY (0.5 FPS)**: 1.0s
* **NORMAL_ACTIVITY (1.0 FPS)**: 7.0s
* **HIGH_ACTIVITY (2.0 FPS)**: 1.0s
* **BURST_CAPTURE (5.0 FPS)**: 0.0s

## Event Candidate Layer

* **Candidates Evaluated**: 17
* **Candidates Forwarded to VLM**: 5
* **Redundant Frames Dropped**: 13
* **Candidate Reduction**: 76.47%

---

## Runtime & Savings Analysis

* **Average Processing Time Per Sent Frame**: 5.86 seconds
* **Actual Pipeline Run Duration (with sampling)**: 52.92 seconds
* **Projected Run Duration Without Sampling**: 129.08 seconds
* **Estimated Runtime Savings**: 76.15 seconds (1.27 minutes)
* **Actual Runtime Savings**: 76.15 seconds (1.27 minutes)

---

## Threshold Configurations

* **ENABLE_ADAPTIVE_SAMPLING**: `True`
* **SSIM_THRESHOLD**: `0.92`
* **HISTOGRAM_THRESHOLD**: `0.25`
* **MOTION_THRESHOLD**: `0.15`
