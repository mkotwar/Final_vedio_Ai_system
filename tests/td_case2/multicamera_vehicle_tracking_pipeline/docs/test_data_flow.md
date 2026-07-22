# Test Data Flow

This folder is meant for basic testing of multi-camera data flow.

## Simple flow

1. Configure a test camera in `cameras`
2. Save local tracker output in `vehicle_tracks`
3. Save plate and colour enrichment in `vehicle_attributes`
4. Save optional trajectory boxes in `vehicle_observations`
5. Save cross-camera comparison output in `vehicle_matches`
6. Query `searchable_vehicles` for search results

## Example

- Camera `CAM-001` sees a white car
- The local tracker writes one `vehicle_tracks` row
- OCR stores `DL01AB1234` and `DL01AB12?4` in `plate_readings`
- A second camera later sees a similar car
- Matching code compares plate, colour, class, and time gap
- One `vehicle_matches` row stores the result as `probable`

## Why this is enough for now

The current testing goal is to prove:

- multiple cameras can coexist
- local tracks stay independent per camera
- plate and colour data can be searched
- simple cross-camera candidate matching can be saved

That makes it a good stepping stone before adding heavier pipeline infrastructure.
