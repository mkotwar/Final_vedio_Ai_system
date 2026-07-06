# Entity Timeline Summary

- Input video: `C:\Mukul K\test_video\V_ai_test_2min.mp4`
- Video duration: `138.57s`
- Entities tracked: `23`
- Entities by type: `{'object': 3, 'person': 20}`
- Average track length: `47.35s`
- Average track frame count: `15.39`
- Broken tracks: `17`
- Lost tracks: `21`
- Wall-clock runtime: `13.44s`

## Broken Tracks

- Track #1 (object/chair): 138.00s, frames=133, distance=78.36px, gaps=5
- Track #2 (object/tv): 138.00s, frames=134, distance=73.65px, gaps=5
- Track #3 (object/dining table): 126.00s, frames=20, distance=19.46px, gaps=11
- Track #5 (person/person): 41.00s, frames=3, distance=176.09px, gaps=2
- Track #6 (person/person): 44.00s, frames=6, distance=268.92px, gaps=1
- Track #7 (person/person): 72.00s, frames=3, distance=20.90px, gaps=2
- Track #8 (person/person): 70.00s, frames=6, distance=40.38px, gaps=5
- Track #9 (person/person): 36.00s, frames=3, distance=111.84px, gaps=1
- Track #10 (person/person): 96.00s, frames=11, distance=155.31px, gaps=5
- Track #11 (person/person): 62.00s, frames=5, distance=47.48px, gaps=2
- Track #12 (person/person): 88.00s, frames=5, distance=18.43px, gaps=4
- Track #13 (person/person): 35.00s, frames=3, distance=29.82px, gaps=2
- Track #14 (person/person): 44.00s, frames=3, distance=130.69px, gaps=2
- Track #16 (person/person): 34.00s, frames=3, distance=78.77px, gaps=2
- Track #17 (person/person): 34.00s, frames=3, distance=47.83px, gaps=2
- Track #18 (person/person): 23.00s, frames=2, distance=36.92px, gaps=1
- Track #22 (person/person): 5.00s, frames=2, distance=23.71px, gaps=1

## Lost Tracks

- Track #3 (object/dining table): 126.00s, frames=20, distance=19.46px, gaps=11
- Track #4 (person/person): 0.00s, frames=1, distance=0.00px, gaps=0
- Track #5 (person/person): 41.00s, frames=3, distance=176.09px, gaps=2
- Track #6 (person/person): 44.00s, frames=6, distance=268.92px, gaps=1
- Track #7 (person/person): 72.00s, frames=3, distance=20.90px, gaps=2
- Track #8 (person/person): 70.00s, frames=6, distance=40.38px, gaps=5
- Track #9 (person/person): 36.00s, frames=3, distance=111.84px, gaps=1
- Track #10 (person/person): 96.00s, frames=11, distance=155.31px, gaps=5
- Track #11 (person/person): 62.00s, frames=5, distance=47.48px, gaps=2
- Track #12 (person/person): 88.00s, frames=5, distance=18.43px, gaps=4
- Track #13 (person/person): 35.00s, frames=3, distance=29.82px, gaps=2
- Track #14 (person/person): 44.00s, frames=3, distance=130.69px, gaps=2
- Track #15 (person/person): 0.00s, frames=1, distance=0.00px, gaps=0
- Track #16 (person/person): 34.00s, frames=3, distance=78.77px, gaps=2
- Track #17 (person/person): 34.00s, frames=3, distance=47.83px, gaps=2
- Track #18 (person/person): 23.00s, frames=2, distance=36.92px, gaps=1
- Track #19 (person/person): 0.00s, frames=1, distance=0.00px, gaps=0
- Track #20 (person/person): 0.00s, frames=1, distance=0.00px, gaps=0
- Track #21 (person/person): 0.00s, frames=1, distance=0.00px, gaps=0
- Track #22 (person/person): 5.00s, frames=2, distance=23.71px, gaps=1
- Track #23 (person/person): 3.00s, frames=4, distance=76.77px, gaps=0

## Longest Tracks

- Track #1 (object/chair): 138.00s, frames=133, distance=78.36px, gaps=5
- Track #2 (object/tv): 138.00s, frames=134, distance=73.65px, gaps=5
- Track #3 (object/dining table): 126.00s, frames=20, distance=19.46px, gaps=11
- Track #10 (person/person): 96.00s, frames=11, distance=155.31px, gaps=5
- Track #12 (person/person): 88.00s, frames=5, distance=18.43px, gaps=4
- Track #7 (person/person): 72.00s, frames=3, distance=20.90px, gaps=2
- Track #8 (person/person): 70.00s, frames=6, distance=40.38px, gaps=5
- Track #11 (person/person): 62.00s, frames=5, distance=47.48px, gaps=2
- Track #6 (person/person): 44.00s, frames=6, distance=268.92px, gaps=1
- Track #14 (person/person): 44.00s, frames=3, distance=130.69px, gaps=2

## Most Active Tracks

- Track #10 (person/person): 96.00s, frames=11, distance=155.31px, gaps=5
- Track #6 (person/person): 44.00s, frames=6, distance=268.92px, gaps=1
- Track #23 (person/person): 3.00s, frames=4, distance=76.77px, gaps=0
- Track #5 (person/person): 41.00s, frames=3, distance=176.09px, gaps=2
- Track #9 (person/person): 36.00s, frames=3, distance=111.84px, gaps=1
- Track #1 (object/chair): 138.00s, frames=133, distance=78.36px, gaps=5
- Track #11 (person/person): 62.00s, frames=5, distance=47.48px, gaps=2
- Track #14 (person/person): 44.00s, frames=3, distance=130.69px, gaps=2
- Track #16 (person/person): 34.00s, frames=3, distance=78.77px, gaps=2
- Track #2 (object/tv): 138.00s, frames=134, distance=73.65px, gaps=5

## Notes

- This benchmark performs no event reasoning and no VLM reasoning.
- Detection, tracking, and OCR are invoked through production services without changing thresholds or inference settings.
- OCR proximity uses `same_frame_text_only_no_ocr_bounding_boxes_available` because OCRService.extract_text does not expose text bounding boxes.
- Zone history uses deterministic 3x3 frame-position zones because no production polygon zone service is present.