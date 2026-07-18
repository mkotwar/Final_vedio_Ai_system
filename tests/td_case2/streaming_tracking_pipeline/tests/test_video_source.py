import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.video_source import OpenCvVideoSource


def _make_video(path: Path, *, fps=10.0, frame_count=12):
    try:
        import cv2
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"OpenCV/numpy unavailable: {exc}") from exc

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        writer.release()
        raise unittest.SkipTest("OpenCV VideoWriter could not create mp4 test video.")
    for index in range(frame_count):
        frame = np.full((24, 32, 3), index * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class OpenCvVideoSourceTest(unittest.TestCase):
    def test_reads_selected_frames_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            _make_video(video_path, fps=10.0, frame_count=12)
            source = OpenCvVideoSource(video_path, source_id="cam_a", target_processing_fps=5.0, max_processed_frames=4)

            source.open()
            packets = []
            while True:
                packet = source.read()
                if packet is None:
                    break
                packets.append(packet)
            source.close()

            self.assertEqual([packet.frame_index for packet in packets], [0, 2, 4, 6])
            self.assertEqual(packets[1].timestamp_sec, 0.2)
            self.assertEqual(source.metadata_report()["expected_selected_frames"], 4)
            self.assertTrue(source.closed)

    def test_reset_allows_reopen_after_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "sample.mp4"
            _make_video(video_path, fps=10.0, frame_count=4)
            source = OpenCvVideoSource(video_path, target_processing_fps=5.0)

            source.open()
            source.close()
            source.reset()
            source.open()
            self.assertEqual(source.read().frame_index, 0)
            source.close()

    def test_missing_video_path_fails_fast(self):
        with self.assertRaises(FileNotFoundError):
            OpenCvVideoSource("missing-video.mp4")


if __name__ == "__main__":
    unittest.main()
