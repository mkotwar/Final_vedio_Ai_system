from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.multicamera_vehicle_tracking_pipeline.enrichment.media_resolver import MediaResolutionError, resolve_local_media_path


class MediaResolverTests(unittest.TestCase):
    def test_relative_uri_resolves_inside_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_root = Path(tmpdir)
            target = artifact_root / "RUN_1" / "CAM_001" / "best_overall.jpg"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"jpg")
            resolved = resolve_local_media_path(
                storage_uri="RUN_1/CAM_001/best_overall.jpg",
                artifact_root=artifact_root,
            )
            self.assertEqual(resolved, target.resolve())

    def test_absolute_uri_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(MediaResolutionError):
                resolve_local_media_path(storage_uri=r"F:\bad\path.jpg", artifact_root=Path(tmpdir))


if __name__ == "__main__":
    unittest.main()
