from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.td_case2.streaming_tracking_pipeline.dominant_colour_analysis import estimate_dominant_colour


class DominantColourAnalysisTests(unittest.TestCase):
    def test_red_tail_light_on_white_vehicle_is_not_red(self) -> None:
        cv2 = __import__("cv2")
        import numpy as np

        with tempfile.TemporaryDirectory() as temp_dir:
            image = np.full((80, 120, 3), (235, 235, 235), dtype=np.uint8)
            image[58:70, 88:108] = (0, 0, 255)
            path = Path(temp_dir) / "white_with_tail_light.jpg"
            cv2.imwrite(str(path), image)
            result = estimate_dominant_colour(path, object_class="car", raw_colour="red")
            self.assertNotEqual(result.dominant_colour, "red")
            self.assertIn(result.dominant_colour, {"white", "gray"})

    def test_person_central_clothing_colour(self) -> None:
        cv2 = __import__("cv2")
        import numpy as np

        with tempfile.TemporaryDirectory() as temp_dir:
            image = np.full((100, 50, 3), (20, 20, 20), dtype=np.uint8)
            image[20:65, 12:38] = (240, 240, 240)
            path = Path(temp_dir) / "person.jpg"
            cv2.imwrite(str(path), image)
            result = estimate_dominant_colour(path, object_class="person", raw_colour=None)
            self.assertEqual(result.dominant_colour, "white")
            self.assertEqual(result.colour_method, "person_clothing_colour")


if __name__ == "__main__":
    unittest.main()
