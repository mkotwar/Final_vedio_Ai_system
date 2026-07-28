# Multicamera Tracking Reference

Last updated: 2026-07-28

This document explains the current tracking implementation in `tests/td_case2/multicamera_vehicle_tracking_pipeline` as it exists in code today. It is intentionally code-specific. It does not describe generic ByteTrack behavior unless that behavior is actually exercised by this project.

## 1. Architecture

Simple view:

```text
camera video file
-> CameraReaderWorker
-> FramePacket
-> DetectionWorker
-> SharedVehicleDetector
-> DetectionPacket
-> TrackingWorker
-> CameraDetectionRouter
-> one CameraTracker per camera
-> one native ByteTrack instance per camera
-> LocalTrackLifecycle
-> class stabilization
-> identity continuity split checks
-> fragment-link recovery
-> TrackEvidenceCollector
-> PersistenceWorker
-> analytics persistence / dry run
-> FastAPI read API
-> React UI
```

Important code entry points:

- `ingestion/camera_source.py`
- `workers/camera_reader_worker.py`
- `detection/vehicle_detector.py`
- `workers/detection_worker.py`
- `tracking/camera_detection_router.py`
- `tracking/camera_tracker.py`
- `tracking/tracker_factory.py`
- `tracking/track_lifecycle.py`
- `tracking/class_stabilization.py`
- `tracking/class_recalculation.py`
- `evidence/track_evidence_collector.py`
- `workers/tracking_worker.py`
- `persistence/analytics_persistence_service.py`
- `api/`
- `frontend/`

## 2. Actual flow

### 2.1 Video source -> frame packet

- Input: file-backed camera entries from `config/cameras*.yaml`.
- Reader: `CameraReaderWorker` opens one `CameraSource` per camera.
- Output: `FramePacket(camera_code, camera_name, source_path, frame_number, source_fps, source_frame_count, video_time_seconds, camera_timestamp, frame)`.
- Ordering rule: frame numbers are generated locally and must be strictly increasing for each camera.
- Failure points: missing file, unreadable file, invalid FPS, empty stream, invalid frame dimensions.

### 2.2 Frame packet -> detection packet

- Worker: one shared `DetectionWorker`.
- Detector: `SharedVehicleDetector`.
- YOLO call: `model.predict(source=frame, conf, iou, imgsz, device, verbose=False)`.
- Output: `DetectionPacket(..., detections=[VehicleDetection(...)], source_fps, frame)`.
- Detection filtering that actually happens:
  - class normalization through `normalize_runtime_vehicle_class`
  - allowed-class filtering
  - finite-bbox validation
  - bbox clamping to frame
  - invalid box rejection
- Detection filtering that is not implemented separately in project code:
  - explicit min-box-area threshold
  - explicit max-detections cap
  - duplicate suppression beyond YOLO/NMS

### 2.3 Detection packet -> per-camera tracking

- Worker: one `TrackingWorker`.
- Router: `CameraDetectionRouter`.
- Isolation rule: the router creates one `CameraTracker` per `camera_code`, and each `CameraTracker` asks `TrackerFactory` for one native tracker instance for that camera.
- Shared state across cameras: none in tracker state; lifecycle state is partitioned by camera.
- Failure points:
  - packet for unexpected camera
  - out-of-order packet
  - downstream tracking exception

### 2.4 YOLO detections -> ByteTrack input

- Supervision backend path:
  - `camera_tracker.py` calls `to_supervision_detections(packet)`.
  - Format passed to Supervision: `sv.Detections(xyxy=float32[N,4], confidence=float32[N], class_id=int32[N])`.
- Ultralytics backend path:
  - `camera_tracker.py` wraps detections in `_ResultsLike`.
  - `xyxy` is preserved.
  - `xywh` is synthesized only for tracker compatibility.
- Coordinates:
  - input from YOLO: `xyxy`
  - tracking input: still `xyxy` for Supervision; row-compatible structures for Ultralytics
  - stored observations: float `bbox_xyxy`
- Metadata preserved into tracking observations:
  - `confidence`
  - `class_id` and normalized `class_name`
  - `frame_number`
  - `camera_code`
  - `video_time_seconds`
  - `camera_timestamp`
- Conversion safeguards:
  - non-finite values rejected
  - `x2 > x1` and `y2 > y1` enforced
  - confidence must be finite and in `[0,1]`

### 2.5 ByteTrack output -> lifecycle

- `CameraTracker` turns tracker rows into `TrackObservation`.
- `LocalTrackLifecycle.update()` owns the logical state machine.
- Important distinction:
  - native tracker ID: the per-backend tracker's current numeric ID
  - logical track ID: the lifecycle-managed per-camera ID that is persisted
  - track UUID: `RUN_ID:CAMERA_CODE:TRACK_<logical_id>` when a run ID is present
- Runtime effect:
  - if ByteTrack reuses one numeric ID for the wrong object, lifecycle can split it
  - if ByteTrack creates a new numeric ID for the same object after a short gap, lifecycle can relink it

### 2.6 Class stabilization

- Every accepted observation contributes to:
  - `class_scores`
  - `class_observation_counts`
  - `class_max_confidences`
  - `raw_class_history`
- `build_class_diagnostics()` computes:
  - provisional winner
  - stable class
  - lock state
  - confidence and winner margin
- The latest YOLO label does not directly overwrite the final track class.
- Why:
  - one noisy frame should not flip a settled `car` track to `bus`
  - persistence and UI should show the stabilized class, while diagnostics still preserve raw history
- Important runtime caveat:
  - `TrackingConfig()` has built-in alias and family defaults
  - `load_tracking_config()` currently supplies empty mappings when YAML omits them
  - the effective worker runtime from `config/tracking.yaml` therefore has `class_aliases = {}` and `class_families = {}`
  - in real worker runs, class compatibility is exact-match only unless YAML explicitly adds families

### 2.7 Identity continuity safeguards

- Entry point: `evaluate_identity_continuity()` in `class_recalculation.py`.
- Checks that actually exist:
  - spatial score
  - class compatibility
  - area ratio
- If continuity fails:
  - lifecycle finalizes the existing logical track
  - that finalized track keeps the old logical ID and UUID
  - a fresh logical track is allocated for the new observation
- This is the current implementation of "ByteTrack ID remains the same internally but lifecycle splits the logical track".

### 2.8 Completion, evidence, persistence, API, UI

- When a logical track times out or camera input ends:
  - `completed` if it met confirmation rules
  - `discarded` if it stayed too short or tentative
- `TrackEvidenceCollector.finalize_track()` selects saved crops and full frames.
- `AnalyticsPersistenceService.save_completed_track()` persists:
  - `vehicle_track`
  - `track_observation`
  - optional `track_media`
  - later enrichment data
- The read-only FastAPI layer serves runs, tracks, media, matches, global vehicles, and search.
- The React frontend displays stabilized class, evidence, memberships, and search results.

## 3. Tracker isolation

- Number of native ByteTrack instances: one per camera code seen by `TrackerFactory`.
- Creation site: `TrackerFactory.get_or_create(camera_code, frame_rate=...)`.
- Routing site: `CameraDetectionRouter.route(packet)`.
- State sharing:
  - native tracker state: not shared
  - lifecycle state: one dictionary per camera inside `LocalTrackLifecycle`
  - evidence state: one dictionary per `track_uuid`
- Reset handling:
  - tracker cache reset: `TrackerFactory.reset()`
  - lifecycle flush: `flush_camera()` and `flush_all()`
- Camera FPS path:
  - file reader reads source FPS from OpenCV
  - `FramePacket.source_fps` is copied into `DetectionPacket.source_fps`
  - `CameraTracker` resolves tracker FPS as `packet.source_fps or tracking_config.frame_rate`
  - this value is passed only when a tracker instance is first created for that camera

## 4. Effective values and precedence

Current environment inspection on Tuesday, July 28, 2026 found no active `TD_CASE2_MULTICAM_*` or `VEHICLE_DETECTOR_MODEL_PATH` overrides in the shell used for this analysis.

Current precedence confirmed in code:

- Detection: environment override -> YAML -> code default
- Tracking: environment override -> YAML -> code default
- Worker: CLI/orchestrator override -> YAML -> code default
- Persistence: CLI/orchestrator override -> environment override -> YAML -> code default
- Evidence: YAML -> code default

Important runtime nuance:

- `tracking.frame_rate` is configured as `20.0` in `config/tracking.yaml`.
- In the worker pipeline, native tracker creation prefers per-camera source FPS from the video file.
- The YAML `20.0` is therefore a fallback, not the final effective FPS when `source_fps` is available.

## 5. Parameter reference

### 5.1 Detection and tracker parameters

| Parameter | Current value | Config file | Code field | Passed to | Meaning | Effect if increased | Effect if decreased | Main risk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tracking backend | `supervision_bytetrack` | `config/tracking.yaml` | `TrackingConfig.backend` | `TrackerFactory` | Chooses native tracker implementation | may unlock different thresholds | stays on current backend | backend mismatch with assumptions |
| track activation threshold | `0.15` | `config/tracking.yaml` | `track_activation_threshold` | `sv.ByteTrack` | min confidence to activate track in Supervision backend | fewer weak activations | more weak activations | fragmentation vs false tracks |
| high-confidence threshold | `0.15` | `config/tracking.yaml` | `track_high_thresh` | Ultralytics backend only | high score gate for Ultralytics ByteTrack | stricter primary association | looser primary association | more misses or more noise |
| low-confidence threshold | `0.10` | `config/tracking.yaml` | `track_low_thresh` | Ultralytics backend only | lower score gate for second-pass matching | more weak boxes eligible | fewer weak recoveries | false continuation vs fragmentation |
| new-track threshold | `0.30` | `config/tracking.yaml` | `new_track_thresh` | Ultralytics backend only | minimum score to spawn new track | fewer new IDs | more new IDs | missed starts vs track spam |
| minimum matching threshold | `0.80` | `config/tracking.yaml` | `minimum_matching_threshold` | `sv.ByteTrack` | minimum association confidence in Supervision backend | stricter matching | looser matching | fragmentation vs ID switches |
| match threshold | `0.80` | `config/tracking.yaml` | `match_thresh` | Ultralytics backend only | association threshold | stricter matching | looser matching | fragmentation vs ID switches |
| lost track buffer | `30` | `config/tracking.yaml` | `lost_track_buffer` | `sv.ByteTrack` | frames ByteTrack can retain lost tracks | longer recovery window | shorter recovery window | stale reactivation vs early fragmentation |
| track buffer | `30` | `config/tracking.yaml` | `track_buffer` | Ultralytics backend only | lost-track retention | longer recovery window | shorter recovery window | stale reactivation vs fragmentation |
| frame rate | `20.0` fallback | `config/tracking.yaml` | `frame_rate` | tracker constructor | fallback FPS only when packet FPS missing | longer time-based retention inside backend assumptions | shorter | wrong runtime FPS if source missing |
| minimum consecutive frames | `1` | `config/tracking.yaml` | `minimum_consecutive_frames` | `sv.ByteTrack` | min consecutive frames for native activation | fewer short activations | more short activations | false tracks vs fragmentation |
| lifecycle confirmation observations | `3` | `config/tracking.yaml` | `min_confirmed_observations` | `LocalTrackLifecycle` | observations needed for logical `active` state | more stable completed tracks | quicker activation | discarded short tracks vs noise |
| maximum missed frames | `30` | `config/tracking.yaml` | `max_lost_frames` | `LocalTrackLifecycle` | logical lost timeout | longer logical recovery window | shorter logical recovery window | stale tracks vs fragmentation |
| detection confidence threshold | `0.25` | `config/detection.yaml` | `confidence_threshold` | YOLO predict | detector confidence gate | fewer detections | more detections | misses vs false tracks |
| YOLO NMS IoU threshold | `0.45` | `config/detection.yaml` | `iou_threshold` | YOLO predict | detector NMS overlap threshold | more overlapping boxes survive | more aggressive suppression | duplicates vs misses |
| YOLO image size | `640` | `config/detection.yaml` | `image_size` | YOLO predict | inference resize | may improve small-object recall | faster inference | throughput vs recall |
| allowed classes | `3wheeler, car, bus, truck, motorcycle` | `config/detection.yaml` | `allowed_classes` | detector post-filter | classes allowed into tracking | broader tracked set | narrower tracked set | irrelevant objects or missing targets |
| preserve state per camera | `true` | `config/tracking.yaml` | `preserve_state_per_camera` | persisted config/report only | intended isolation flag | no runtime change today | no runtime change today | false sense of tunability |

### 5.2 Class stabilization

These values currently come from code defaults because `tracking.yaml` does not override them.

| Parameter | Current value | Config source | Meaning |
| --- | --- | --- | --- |
| enabled | `true` | `TrackingConfig.ClassStabilizationConfig` | turn stabilization on |
| strategy | `confidence_weighted_vote` | code default | documented scoring approach |
| observation_count_weight | `0.10` | code default | reward repeated class observations |
| max_confidence_weight | `0.25` | code default | reward strong peak confidence |
| minimum_observations | `2` | code default | minimum history before stable class can be set |
| minimum_winner_margin | `0.20` | code default | margin needed to lock class |
| lock_after_observations | `5` | code default | history size needed before lock |
| allow_unlock_on_strong_conflict | `true` | code default | lets repeated contrary evidence replace locked class |
| strong_conflict_min_observations | `3` | code default | minimum repeated contrary observations |
| strong_conflict_margin | `0.75` | code default | margin needed to replace locked class |

Related effective runtime values:

| Parameter | Current effective value | Source |
| --- | --- | --- |
| class aliases | `{}` | YAML omission plus `load_tracking_config()` empty override |
| class families | `{}` | YAML omission plus `load_tracking_config()` empty override |

### 5.3 Identity continuity and fragment linking

| Parameter | Current value | Source | Meaning | Notes |
| --- | --- | --- | --- | --- |
| identity continuity enabled | `true` | `config/tracking.yaml` | allow same-camera logical split checks | active today |
| identity minimum spatial score | `0.20` | `config/tracking.yaml` | spatial continuity floor | used |
| identity minimum class compatibility | `0.50` | `config/tracking.yaml` | class-family continuity floor | used, but families are currently empty so non-exact classes score `0.0` |
| identity maximum area ratio | `3.50` | `config/tracking.yaml` | scale discontinuity limit | used |
| identity hard split spatial score | `0.08` | `config/tracking.yaml` | immediate split threshold | used |
| fragment linking enabled | `true` | `config/tracking.yaml` | allow logical relink after new tracker ID | active today |
| fragment maximum gap seconds | `1.5` | `config/tracking.yaml` | max allowed temporal gap | used |
| fragment minimum spatial score | `0.55` | `config/tracking.yaml` | relink spatial continuity floor | used |
| fragment minimum class compatibility | `0.50` | `config/tracking.yaml` | relink class-family floor | used, but families are currently empty so non-exact classes score `0.0` |
| fragment require no time overlap | `true` | `config/tracking.yaml` | reject overlapping fragments | used |
| reject verified plate conflict | `true` | `config/tracking.yaml` | reject contradictory verified plates | not implemented in `evaluate_fragment_link()` yet |

### 5.4 Worker and evidence parameters that affect perceived tracking quality

| Parameter | Current value | Source | Meaning |
| --- | --- | --- | --- |
| frame queue size | `20` | `config/workers.yaml` | frame backpressure capacity |
| detection queue size | `20` | `config/workers.yaml` | detector-to-tracker buffer |
| completed track queue size | `20` | `config/workers.yaml` | tracker-to-persistence buffer |
| queue put timeout seconds | `2.0` | `config/workers.yaml` | bounded queue wait |
| queue get timeout seconds | `1.0` | `config/workers.yaml` | polling wait |
| shutdown timeout seconds | `30.0` | `config/workers.yaml` | thread join limit |
| vehicle colour worker count | `1` | `config/workers.yaml` | enrichment concurrency |
| ANPR worker count | `1` | `config/workers.yaml` | enrichment concurrency |
| evidence minimum detection confidence | `0.20` | `config/evidence.yaml` | observation must exceed this for evidence selection |
| evidence minimum crop width | `40` | `config/evidence.yaml` | smallest saved crop width |
| evidence minimum crop height | `40` | `config/evidence.yaml` | smallest saved crop height |
| evidence padding ratios | `0.05`, `0.08`, `0.08`, `8px` | `config/evidence.yaml` | crop expansion |
| visibility weight | `0.30` | `config/evidence.yaml` | best-overall score weight |
| detection confidence weight | `0.20` | `config/evidence.yaml` | best-overall score weight |
| sharpness weight | `0.15` | `config/evidence.yaml` | best-overall score weight |
| centeredness weight | `0.15` | `config/evidence.yaml` | best-overall score weight |
| bbox area weight | `0.10` | `config/evidence.yaml` | best-overall score weight |
| edge penalty weight | `0.10` | `config/evidence.yaml` | penalty for clipped/edge crops |

### 5.5 Requested items that are not implemented

- explicit minimum box area threshold in detection
- explicit maximum detections cap
- class-agnostic tracking toggle
- center-distance threshold as a separately configurable parameter
- bbox area-ratio threshold separate from `maximum_area_ratio`
- width-ratio threshold
- height-ratio threshold
- minimum IoU for conflicting classes
- class conflict confidence threshold
- colour compatibility inside fragment linking
- verified-plate conflict enforcement inside fragment-link evaluation

## 6. ByteTrack behavior in this project

- New native track creation:
  - handled by the native backend
  - Supervision backend uses `track_activation_threshold`, `lost_track_buffer`, `minimum_matching_threshold`, `minimum_consecutive_frames`
- New logical track creation:
  - handled by `LocalTrackLifecycle`
  - if no active/lost logical state can be reused, it allocates a new logical ID
- Active track:
  - lifecycle state becomes `active` once `observation_count >= min_confirmed_observations`
- Confirmed track:
  - this project's logical confirmation is lifecycle-based, not directly the native backend's own state machine
- Missing detections:
  - native tracker may still keep internal state
  - lifecycle increments `lost_frame_count` when a logical track is not observed in a frame
- Temporarily lost:
  - lifecycle changes `active` or `temporarily_lost` tracks into `temporarily_lost`
- Removed:
  - lifecycle finalizes track once `lost_frame_count > max_lost_frames`
- Reactivation:
  - same logical track can resume from `temporarily_lost` when a matching observation arrives
  - same-object relink after a new native ID is handled by fragment linking
- IDs:
  - logical IDs are allocated per camera
  - they do not reset across all cameras in one run
  - `build_track_uuid()` prefixes the run code when one exists

Difference between the thresholds:

- YOLO detection confidence: detector score gate before tracking
- ByteTrack activation threshold: backend score gate for native track activation
- new-track threshold: Ultralytics backend only; score gate for new native tracks
- matching threshold / minimum matching threshold: backend association strictness
- lifecycle confirmation threshold: number of accepted observations before logical `active`

## 7. Common failure modes

### Same vehicle receives two IDs

Likely causes in current code:

- YOLO misses or low detector confidence
- strict matching threshold
- short lost-track retention
- wrong or unstable source FPS
- large bbox jump lowering spatial continuity
- occlusion
- tracker reset between runs
- out-of-order packet rejection

Safe first checks:

- detector recall on the missed frames
- `minimum_matching_threshold` or `match_thresh`
- `lost_track_buffer` / `track_buffer`
- `max_lost_frames`
- actual `source_fps` vs fallback `tracking.frame_rate`

### One track contains two vehicles

Likely causes:

- permissive native association
- nearby objects with similar geometry
- long lost retention
- weak class separation in native tracking
- identity continuity thresholds too permissive

Current safeguard:

- lifecycle can split one reused native ID when continuity fails strongly

### Duplicate simultaneous tracks

Likely causes:

- duplicate YOLO boxes surviving NMS
- loose YOLO IoU threshold
- backend creates another native track for overlapping detections
- short confirmation threshold

### Short noisy tracks

Likely causes:

- low activation threshold
- low detection confidence threshold
- `minimum_consecutive_frames=1`
- low lifecycle confirmation requirement relative to scene noise

## 8. Worker ordering and queue behavior

- Cameras are read in parallel by one `CameraReaderWorker` per enabled camera.
- Detection is shared through one `DetectionWorker`.
- Tracking is shared through one `TrackingWorker`.
- Native tracker isolation is still per camera because `CameraDetectionRouter` partitions by `camera_code`.
- Per-camera ordering is enforced twice:
  - `TrackingWorker._validate_order()`
  - `LocalTrackLifecycle._validate_packet_order()`
- If frame 104 arrives after frame 103 for the same camera: processing continues.
- If frame 103 arrives after frame 104 for the same camera: worker raises an out-of-order error.
- Frames can effectively be dropped if upstream shutdown occurs or a fatal worker error stops the pipeline.
- High source FPS does not guarantee good tracking because:
  - detection quality can still be poor
  - association can still be too strict or too loose
  - occlusion and duplicated detections still dominate
  - queue backpressure and model latency can still affect end-to-end throughput

## 9. Persistence and data model

Main persisted entities used by tracking:

- processing run
- camera run
- vehicle track
- track observation
- track media
- vehicle attribute
- plate detection
- plate reading
- plate summary
- cross-camera match
- global vehicle
- global vehicle track

Logical meanings:

- raw observation: one accepted per-frame tracked bbox
- active tracker state: current native tracker memory
- lifecycle state: tentative/active/temporarily_lost/completed/discarded
- completed track: finalized logical local track
- persisted track: completed logical track translated into analytics schema
- global vehicle object: post-run cross-camera grouping, not same-camera tracking state

Persistence note:

- the pipeline is still track-centric at this stage, not event-centric
- this remains an isolated TD case pipeline and does not change the root system rule that events should remain the long-term source of truth

## 10. Evidence selection after tracking

Evidence roles that can be selected:

- `first`
- `middle`
- `last`
- `highest_confidence`
- `largest`
- `sharpest`
- `best_overall`

Saved artifacts per chosen role can include:

- vehicle crop
- original full frame
- annotated full frame
- manifest entry with bbox and scores

Evidence affects perception, not IDs:

- a poor crop can hide that tracking was actually correct
- a clipped sharp crop can make a mixed-ID problem look worse than it is
- the current scorer explicitly rewards visibility and centeredness so `best_overall` is less likely to choose an edge-clipped crop

## 11. Practical tuning guide

| Symptom | First metrics to inspect | Parameters to check | Safe direction | Risk |
| --- | --- | --- | --- | --- |
| same vehicle gets new ID after short miss | missed detections, temporarily_lost count, source FPS | detector confidence, matching threshold, lost buffers, max_lost_frames | slightly lower detector threshold or slightly longer lost retention | more false continuations |
| tracker switches from truck to car | raw class history, continuity split findings | class stabilization defaults, identity continuity thresholds | tighten continuity or keep stronger stabilization | more fragmentation |
| too many tiny tracks | discarded track count, short observation counts | detector threshold, activation threshold, confirmation observations | raise detector/activation or confirmation slightly | missed short real objects |
| false tracks | empty-scene detections, duplicate boxes | detector threshold, NMS IoU, activation threshold | raise detector threshold first | recall loss |
| duplicate IDs for one vehicle | duplicate detections, tracker output rows | NMS IoU, matching thresholds | reduce duplicate boxes before loosening matcher | over-suppression |
| tracks survive too long | long temporarily_lost spans | lost buffers, max_lost_frames | reduce retention slightly | fragmentation |
| tracks disappear too quickly | completed after brief miss | lost buffers, max_lost_frames, detector recall | increase retention slightly | stale relinks |
| objects at frame edge fragment | edge-clipped detections, visibility diagnostics | detector recall, continuity thresholds | review edge cases before tightening matcher | hidden true detections |
| nearby vehicles merge | continuity split report, bbox overlap | matching threshold, continuity spatial threshold | tighten association and continuity | fragmentation |
| class changes frequently | raw class history and winner margin | class stabilization defaults | preserve stabilization, do not trust latest label | slower class correction |

Tune together:

- lower detector threshold + higher confirmation observations
- longer lost retention + stricter matching
- looser matching + stricter identity continuity

## 12. Conservative experimental ranges

These are intentionally small test ranges around current values.

| Parameter | Current | Small test range | Monitor | Stop condition |
| --- | --- | --- | --- | --- |
| detection confidence | `0.25` | `0.20-0.30` | recall, false tracks | false-track growth becomes obvious |
| NMS IoU | `0.45` | `0.40-0.50` | duplicate tracks, missed nearby vehicles | duplicates rise or neighbors disappear |
| activation threshold | `0.15` | `0.12-0.20` | short noisy tracks, missed starts | obvious noise or missed entries |
| minimum matching threshold | `0.80` | `0.75-0.85` | fragmentation vs switches | either symptom worsens sharply |
| lost track buffer | `30` | `20-40` | short-gap recovery | stale re-links or more fragments |
| max lost frames | `30` | `20-40` | logical completion timing | same vehicle splits or stale tracks linger |
| min confirmed observations | `3` | `2-4` | completed/discarded ratio | too many false completes or too many discards |
| identity minimum spatial score | `0.20` | `0.15-0.30` | split frequency | merge returns or fragmentation spikes |
| identity hard split score | `0.08` | `0.06-0.12` | severe ID-switch prevention | same object starts over-splitting |
| fragment max gap seconds | `1.5` | `1.0-2.0` | same-object recovery after short miss | unsafe relinks or excess fragments |

Priority order for experiments:

1. prevent ID switches
2. reduce fragmentation
3. maintain detection recall
4. maintain throughput

## 13. Debugging commands

Print the effective tracking runtime report:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.report_tracking_configuration `
  --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml `
  --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml `
  --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml `
  --worker-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\workers.yaml `
  --persistence-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\persistence.yaml `
  --evidence-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\evidence.yaml `
  --json-output debug_runs\multicamera_vehicle_tracking_pipeline\tracking_config_report.json
```

Run one camera:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_worker_multicamera_tracking `
  --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml `
  --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml `
  --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml `
  --worker-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\workers.yaml `
  --camera-code CAM_001 `
  --max-frames-per-camera 200 `
  --output-report debug_runs\multicamera_vehicle_tracking_pipeline\cam1_report.json
```

Run three cameras:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_worker_multicamera_tracking `
  --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras_user_three_videos.yaml `
  --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml `
  --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml `
  --worker-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\workers.yaml `
  --max-frames-per-camera 256 `
  --save-sample-frames `
  --sample-frame-limit-per-camera 3 `
  --output-report debug_runs\multicamera_vehicle_tracking_pipeline\three_camera_report.json
```

Run full video:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.validate_worker_multicamera_tracking `
  --camera-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\cameras.yaml `
  --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml `
  --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml `
  --worker-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\workers.yaml `
  --output-report debug_runs\multicamera_vehicle_tracking_pipeline\full_run_report.json
```

Generate identity-switch diagnostics:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.diagnose_track_identity_switch `
  --run-code RUN_20260727_170538 `
  --camera-code CAM_002 `
  --video-path tests\td_case2\multicamera_vehicle_tracking_pipeline\data\testv\2test_20.mp4 `
  --start-frame 0 `
  --end-frame 300 `
  --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml `
  --detection-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\detection.yaml `
  --output-dir debug_runs\multicamera_vehicle_tracking_pipeline\identity_diag `
  --output-report debug_runs\multicamera_vehicle_tracking_pipeline\identity_diag.json
```

Inspect one persisted track's enrichment/evidence state:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.verify_enrichment_run `
  --run-code RUN_20260724_151402 `
  --track-uuid RUN_20260724_151402:CAM_001:TRACK_4 `
  --include-observations `
  --json-output debug_runs\multicamera_vehicle_tracking_pipeline\track4_audit.json
```

Recalculate or audit stabilized classes for a specific track:

```powershell
.\.venv\Scripts\python.exe -m tests.td_case2.multicamera_vehicle_tracking_pipeline.scripts.recalculate_track_classes `
  --run-code RUN_20260727_131724 `
  --track-uuid RUN_20260727_131724:CAM_001:TRACK_2 `
  --tracking-config tests\td_case2\multicamera_vehicle_tracking_pipeline\config\tracking.yaml `
  --dry-run `
  --output-report debug_runs\multicamera_vehicle_tracking_pipeline\track2_class_recalc.json
```

Compare two saved reports:

```powershell
git diff --no-index debug_runs\multicamera_vehicle_tracking_pipeline\tracking_config_report_a.json debug_runs\multicamera_vehicle_tracking_pipeline\tracking_config_report_b.json
```

## 14. Known limitations

- The current Supervision and Ultralytics ByteTrack integrations do not expose per-match IoU or association score through this pipeline.
- `reject_verified_plate_conflict` exists in config but is not enforced by fragment-link evaluation yet.
- `preserve_state_per_camera` is currently descriptive only; runtime behavior is already per-camera regardless of that flag.
- The worker pipeline is track-centric and isolated; it is not the final event-centric investigation pipeline.
- Several requested thresholds in the original analysis prompt are not separate implemented knobs today.

## 15. Files inspected

- `config/tracking.yaml`
- `config/detection.yaml`
- `config/workers.yaml`
- `config/evidence.yaml`
- `tracking/tracking_config.py`
- `tracking/tracker_factory.py`
- `tracking/camera_tracker.py`
- `tracking/camera_detection_router.py`
- `tracking/track_lifecycle.py`
- `tracking/class_stabilization.py`
- `tracking/class_recalculation.py`
- `tracking/supervision_conversion.py`
- `tracking/annotation.py`
- `workers/camera_reader_worker.py`
- `workers/detection_worker.py`
- `workers/tracking_worker.py`
- `workers/worker_config.py`
- `workers/worker_supervisor.py`
- `evidence/track_evidence_collector.py`
- `evidence/evidence_config.py`
- `evidence/evidence_models.py`
- `persistence/analytics_persistence_service.py`
- `persistence/evidence_to_track_media_mapper.py`
- `persistence/persistence_models.py`
- `scripts/validate_worker_multicamera_tracking.py`
- `scripts/diagnose_track_identity_switch.py`
- `scripts/recalculate_track_classes.py`
- `tests/test_tracker_factory.py`
- `tests/test_camera_tracker.py`
- `tests/test_track_lifecycle.py`
- `tests/test_camera_detection_router.py`
- `tests/test_tracking_worker.py`
- `tests/test_track_evidence_collector.py`
- `tests/test_recalculate_track_classes.py`

READY: The current multicamera tracking implementation, lifecycle, runtime configuration, safeguards, failure modes and every effective parameter are fully documented and traceable.
