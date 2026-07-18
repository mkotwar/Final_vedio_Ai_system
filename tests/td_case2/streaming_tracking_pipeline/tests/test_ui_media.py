from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.ui_media import (
    image_display_spec,
    image_status,
    object_crop_caption,
    plate_crop_caption,
    render_evidence_image,
)


class UIMediaTests(unittest.TestCase):
    def test_image_type_specs_return_uniform_dimensions(self) -> None:
        self.assertEqual({"width": 640, "height": 360, "aspect_ratio": "16 / 9", "fit": "contain"}, image_display_spec("full_frame"))
        self.assertEqual({"width": 320, "height": 240, "aspect_ratio": "4 / 3", "fit": "contain"}, image_display_spec("object_crop"))
        self.assertEqual({"width": 300, "height": 100, "aspect_ratio": "3 / 1", "fit": "contain"}, image_display_spec("plate_crop"))

    def test_missing_image_status_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = image_status("missing.jpg", run_dir=tmp, repo_root=tmp)
            self.assertFalse(status["exists"])
            self.assertEqual("Image missing", status["placeholder"])

    def test_render_evidence_image_handles_missing_path_with_fixed_container(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = _FakeStreamlitTarget()
            status = render_evidence_image(
                target,
                "missing.jpg",
                "full_frame",
                run_dir=tmp,
                repo_root=tmp,
                caption="Full frame",
                missing_message="Full frame unavailable",
            )

            self.assertFalse(status["exists"])
            self.assertIn("max-width:640px", target.markdown_calls[0])
            self.assertIn("aspect-ratio:16 / 9", target.markdown_calls[0])
            self.assertIn("Full frame unavailable", target.markdown_calls[0])

    def test_render_evidence_image_embeds_existing_image_without_stretching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "crop.jpg"
            image.write_bytes(b"not-a-real-jpeg-but-readable")
            target = _FakeStreamlitTarget()
            status = render_evidence_image(target, image, "object_crop", run_dir=tmp, repo_root=tmp, caption="Vehicle crop")

            self.assertTrue(status["exists"])
            self.assertIn("object-fit:contain", target.markdown_calls[0])
            self.assertIn("max-width:320px", target.markdown_calls[0])

    def test_captions_include_only_available_values(self) -> None:
        evidence = {"track_id": 18, "object_class": "car", "plate_text": "DL01AB1234", "plate_status": "verified"}

        self.assertEqual("Vehicle crop · Track 18 · car", object_crop_caption(evidence))
        self.assertEqual("Plate crop · DL01AB1234 · verified", plate_crop_caption(evidence))


class _FakeStreamlitTarget:
    def __init__(self) -> None:
        self.markdown_calls: list[str] = []

    def markdown(self, payload: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdown_calls.append(payload)
        if not unsafe_allow_html:
            raise AssertionError("Evidence renderer must allow central HTML/CSS container rendering.")


if __name__ == "__main__":
    unittest.main()
