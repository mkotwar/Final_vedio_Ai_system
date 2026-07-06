# Perception Audit

- Input video: `C:\Mukul K\test_video\V_ai_test_2min.mp4`
- Video duration: `138.57s`
- Frames analyzed: `139`
- Total detections: `354`
- Total tracks: `23`
- Overlay video: `C:\Mukul K\vinfo1\video-search-engine\tests\manual_benchmark_case\data\output\perception_frame_overlay.mp4`

## Detected Classes

- chair: detections=133, frames=132, avg_conf=0.458, tracks=1, avg_lifetime=138.00s, avg_speed=0.41px/s
- dining table: detections=20, frames=20, avg_conf=0.281, tracks=1, avg_lifetime=126.00s, avg_speed=0.55px/s
- person: detections=67, frames=55, avg_conf=0.771, tracks=20, avg_lifetime=34.35s, avg_speed=7.27px/s
- tv: detections=134, frames=134, avg_conf=0.462, tracks=1, avg_lifetime=138.00s, avg_speed=0.52px/s

## Tender Object Audit

- handbag: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- backpack: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- suitcase: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- laptop bag: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- parcel: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- bottle: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- phone: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- box: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- helmet: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- fire extinguisher: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000
- any movable object: detected=False, never_detected=True, tracked=False, detections=0, frames=0, avg_conf=0.000

## Track Quality

- Interrupted tracks: `17`
- Merge candidates: `7`
- Split candidates: `6`

## Evidence-Based Answers

- Dominant detected classes: tv (134), chair (133), person (67), dining table (20).
- Reliably visible classes by frequency/confidence: person.
- Tender objects completely missing: handbag, backpack, suitcase, laptop bag, parcel, bottle, phone, box, helmet, fire extinguisher, any movable object.
- Unstable tracks observed: 22 of 23 tracks.
- Classes needing perception review: chair, dining table, person, tv.