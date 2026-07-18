from __future__ import annotations

import unittest

from tests.td_case2.streaming_tracking_pipeline.sources import SyntheticFrameSource, build_processing_frame_indices


class SourceTests(unittest.TestCase):
    def test_reading_before_open_fails(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=1, source_fps=30, frame_width=10, frame_height=10)
        with self.assertRaises(RuntimeError):
            source.read()

    def test_normal_open_read_end_close_flow_and_idempotent_close(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=2, source_fps=2, frame_width=10, frame_height=10)
        source.open()
        self.assertEqual(source.read().frame_index, 0)
        self.assertEqual(source.read().frame_index, 1)
        self.assertIsNone(source.read())
        self.assertIsNone(source.read())
        source.close()
        source.close()
        self.assertTrue(source.closed)
        with self.assertRaises(RuntimeError):
            source.read()

    def test_reset_behavior(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=2, source_fps=2, frame_width=10, frame_height=10)
        source.open()
        self.assertEqual(source.read().frame_index, 0)
        source.close()
        source.reset()
        source.open()
        self.assertEqual(source.read().frame_index, 0)

    def test_all_frames_selection_and_exact_divisor(self) -> None:
        self.assertEqual(build_processing_frame_indices(4, 4.0, None), (0, 1, 2, 3))
        self.assertEqual(build_processing_frame_indices(10, 10.0, 5.0), (0, 2, 4, 6, 8))

    def test_fractional_fps_selection_timing(self) -> None:
        indices = build_processing_frame_indices(30, 30.0, 7.0)
        self.assertEqual(indices[0], 0)
        self.assertEqual(len(indices), 7)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(tuple(sorted(indices)), indices)
        times = [index / 30.0 for index in indices]
        gaps = [right - left for left, right in zip(times, times[1:])]
        self.assertLess(max(gaps) - min(gaps), 1.0 / 30.0 + 1e-9)

    def test_2997_source_fps_handling_zero_frames_and_first_frame(self) -> None:
        self.assertEqual(build_processing_frame_indices(0, 29.97, 7.0), ())
        indices = build_processing_frame_indices(300, 29.97, 7.0)
        self.assertEqual(indices[0], 0)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertTrue(all(0 <= item < 300 for item in indices))

    def test_timestamps_use_source_frame_index(self) -> None:
        source = SyntheticFrameSource(source_id="cam", total_frames=10, source_fps=10, frame_width=10, frame_height=10, target_processing_fps=5)
        source.open()
        packets = []
        while True:
            packet = source.read()
            if packet is None:
                break
            packets.append(packet)
        self.assertEqual([item.frame_index for item in packets], [0, 2, 4, 6, 8])
        self.assertEqual([item.timestamp_sec for item in packets], [0.0, 0.2, 0.4, 0.6, 0.8])

    def test_target_fps_greater_than_source_policy_and_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            build_processing_frame_indices(10, 10.0, 11.0)
        with self.assertRaises(ValueError):
            SyntheticFrameSource(source_id="cam", total_frames=-1, source_fps=10, frame_width=10, frame_height=10)
        with self.assertRaises(ValueError):
            SyntheticFrameSource(source_id="cam", total_frames=1, source_fps=0, frame_width=10, frame_height=10)
        with self.assertRaises(ValueError):
            SyntheticFrameSource(source_id="cam", total_frames=1, source_fps=1, frame_width=0, frame_height=10)

    def test_custom_frame_factory(self) -> None:
        source = SyntheticFrameSource(
            source_id="cam",
            total_frames=1,
            source_fps=1,
            frame_width=10,
            frame_height=10,
            frame_factory=lambda index: {"custom": index},
        )
        source.open()
        self.assertEqual(source.read().frame, {"custom": 0})


if __name__ == "__main__":
    unittest.main()

