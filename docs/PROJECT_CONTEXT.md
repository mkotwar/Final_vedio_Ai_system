# AI Video Review & Investigation System

## Project Mission

This project is an enterprise-grade AI Video Review & Investigation System inspired by modern Safe City, CCTV Intelligence, and Video Investigation platforms.

The objective is NOT to generate video captions or create summarized videos.

The objective is to transform raw video footage into structured, searchable investigation intelligence.

The system must allow operators, investigators, and analysts to:

* Upload videos from CCTV, VMS, drones, mobile devices, and recorded footage.
* Automatically analyze video content.
* Extract people, vehicles, objects, activities, and events.
* Build a searchable investigation database.
* Generate investigation-grade text summaries.
* Search using natural language.
* Search by appearance similarity.
* Investigate activity across multiple cameras.
* Perform dwell, path, speed, and area analytics.
* Review events through a web-based dashboard.

The final deliverable is a Video Intelligence Platform, not a video summarization application.

---

# Core Architectural Principle

The system is EVENT-CENTRIC.

All downstream capabilities must operate on events, not raw frames.

Incorrect:

Video
→ Frames
→ Search

Correct:

Video
→ Frames
→ Metadata
→ Events
→ Search
→ Investigation

Events are the primary source of truth.

Investigator-facing evidence exports must remain event-driven. If the system
produces a condensed evidence video or clip package, it should be assembled
from searchable events and already-generated artifacts rather than by
re-running scene understanding directly over the source video.

---

# Current System Architecture

Video Upload
↓
Frame Extraction
↓
Adaptive Sampling
↓
OCR
↓
Vision-Language Model Analysis
↓
Frame Metadata
↓
Event Aggregation
↓
Event Catalog
↓
Investigation Services

---

# Current Technology Stack

Backend

* FastAPI
* Pydantic
* Uvicorn

AI / ML

* Qwen2.5-VL-7B-Instruct
* Transformers
* Accelerate
* PyTorch
* BitsAndBytes

Computer Vision

* OpenCV
* EasyOCR

Storage

* JSON Metadata
* JSON Events

Infrastructure

* Docker
* Docker Compose

Hardware

* RTX 5070 Ti 16GB
* 32GB RAM

---

# Current Processing Pipeline

1. User uploads video.
2. Video is stored.
3. Frames are extracted.
4. Adaptive sampling reduces redundant frames.
5. OCR extracts visible text.
6. Qwen2.5-VL generates rich frame metadata.
7. Metadata is validated and repaired.
8. Event Aggregator merges related frames.
9. Event Catalog is created.
10. Investigation services consume events.

---

# Frame Metadata Purpose

Frame metadata is an intermediate artifact.

It is NOT the final product.

Frame metadata exists only to enable event construction.

Example:

Frame Metadata:

* person wearing backpack
* crossing road
* red motorcycle
* taxi near gate

These should ultimately become events.

---

# Event Architecture

Events are the core system entity.

Every event should contain:

```json
{
  "event_id": "evt_001",
  "event_type": "pedestrian_crossing",
  "description": "Pedestrian crossed the road carrying a white bag",
  "start_time": "00:00:00",
  "end_time": "00:00:05",
  "duration_seconds": 5
}
```

All future systems must consume events rather than frame metadata.

Examples:

* Search indexes events.
* Summaries are generated from events.
* Dashboards display events.
* Analytics use events.
* ReID links events.
* Multi-camera tracking links events.

---

# What Has Been Completed

## Video Ingestion

Status: Production Beta

Completed:

* Video Upload
* Frame Extraction
* Timestamp Generation
* Adaptive Sampling
* OCR
* VLM Metadata Generation
* Metadata Validation
* Event Aggregation
* Event Catalog Creation
* Performance Profiling

---

## Performance Status

Profiling Results:

Qwen Inference:
80–94% of runtime

OCR:
5–19% of runtime

Everything Else:
<1%

Current Performance:

Approximately 7–10 seconds per analyzed frame.

The primary bottleneck is VLM inference.

Further optimization of OCR, JSON processing, or storage is currently low priority.

---

# Current Development Phase

Phase 1

Event-Centric Investigation Summaries

Goal:

Generate investigation reports from event catalogs.

Input:

Events

Output:

Human-readable investigation summary

Example:

* 32 vehicles entered
* 28 vehicles exited
* 14 pedestrians observed
* Peak activity between 18:05 and 18:25
* One vehicle remained stationary for 12 minutes

Important:

Summaries must be generated from events.

Never summarize raw frames.

---

# Upcoming Roadmap

## Phase 1

Event-Centric Summaries

Deliverables:

* Summary Service
* Summary API
* Timeline Generation
* Statistics Engine
* Peak Activity Detection
* Notable Event Detection

---

## Phase 2

Semantic Search

Goal:

Search videos using natural language.

Examples:

* Find red motorcycle.
* Find person carrying backpack.
* Find vehicle entering gate.

Architecture:

Events
→ Embeddings
→ Qdrant
→ Search API

Recommended Embeddings:

* BGE-M3
* BGE-Large

Deliverables:

* Embedding Pipeline
* Qdrant Integration
* Search Service
* Search API

---

## Phase 3

Object Tracking

Goal:

Create persistent object identities.

Recommended Stack:

YOLO
+
ByteTrack

Output:

Track IDs

Required For:

* Path Analytics
* Dwell Analytics
* Speed Analytics
* Area Analytics

---

## Phase 4

Investigation Analytics

Capabilities:

* Area Analytics
* Polygon Zones
* Dwell Detection
* Path Analysis
* Speed Analysis

---

## Phase 5

Person ReID & Vehicle ReID

Goal:

Appearance similarity search.

Examples:

* Find this person.
* Find this vehicle.
* Search across videos.
* Search across cameras.

---

## Phase 6

Multi-Camera Investigation

Goal:

Track entities across cameras.

Examples:

Camera A
→ Camera B
→ Camera C

Deliverables:

* Cross-camera correlation
* Investigation timelines

---

## Phase 7

Investigation Dashboard

Recommended Stack:

React
+
FastAPI

Features:

* Upload Videos
* Search
* Filters
* Event Review
* Timeline
* Analytics
* Case Management

---

# Architectural Rules

Rule 1

Events are the source of truth.

Rule 2

Do not build features directly on frame metadata.

Rule 3

Every feature should consume event catalogs whenever possible.

Rule 4

Search is more important than further ingestion optimization.

Rule 5

Tracking must exist before path, dwell, area, and speed analytics.

Rule 6

ReID should be implemented after search and tracking.

Rule 7

All services should be production-grade and API-first.

Rule 8

Every phase must be validated before moving to the next phase.

---

# End Goal

The final system should function as a complete AI Video Review & Investigation Platform capable of:

* Video Understanding
* Event Detection
* Investigation Search
* Semantic Retrieval
* Appearance Search
* Multi-Camera Tracking
* Analytics
* Investigation Reporting
* Safe City Operations

This project is not a video captioning system.

This project is an enterprise video intelligence platform.
