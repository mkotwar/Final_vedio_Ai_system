from __future__ import annotations

import inspect
import unittest

from tests.td_case2.streaming_tracking_pipeline import run_anpr_video_10fps_validation as runner


class AnprVideoValidationRunnerTests(unittest.TestCase):
    def test_parser_defaults_to_bounded_10fps_auto_devices(self) -> None:
        args = runner.build_parser().parse_args([])
        self.assertEqual(args.target_fps, 10.0)
        self.assertEqual(args.max_processed_frames, 600)
        self.assertFalse(args.full_video)
        self.assertEqual(args.tracking_backend, "ultralytics_bytetrack")
        self.assertEqual(args.vehicle_device, "auto")
        self.assertEqual(args.plate_device, "auto")
        self.assertEqual(args.florence_device, "auto")
        self.assertEqual(args.florence_dtype, "auto")
        self.assertFalse(args.florence_load_in_4bit)
        self.assertFalse(args.florence_load_in_8bit)

    def test_auto_device_plan_uses_cuda_float16_when_available(self) -> None:
        args = runner.build_parser().parse_args([])
        plan = runner._build_device_plan(args, cuda_available=True, gpu_name="Test GPU")
        self.assertEqual(plan.actual_vehicle_device, "cuda")
        self.assertEqual(plan.actual_plate_device, "cuda")
        self.assertEqual(plan.actual_florence_device, "cuda")
        self.assertEqual(plan.actual_florence_dtype, "float16")
        self.assertEqual(plan.gpu_name, "Test GPU")

    def test_auto_device_plan_uses_cpu_float32_without_cuda(self) -> None:
        args = runner.build_parser().parse_args([])
        plan = runner._build_device_plan(args, cuda_available=False, gpu_name=None)
        self.assertEqual(plan.actual_vehicle_device, "cpu")
        self.assertEqual(plan.actual_plate_device, "cpu")
        self.assertEqual(plan.actual_florence_device, "cpu")
        self.assertEqual(plan.actual_florence_dtype, "float32")

    def test_full_video_flag_keeps_default_limit_available(self) -> None:
        args = runner.build_parser().parse_args(["--full-video"])
        self.assertTrue(args.full_video)
        self.assertEqual(args.max_processed_frames, 600)

    def test_runner_source_stays_sequential(self) -> None:
        source = inspect.getsource(runner)
        forbidden_tokens = ("queue.Queue", "import threading", "import multiprocessing", "ProcessPoolExecutor", "ThreadPoolExecutor")
        for token in forbidden_tokens:
            self.assertNotIn(token, source)

    def test_direct_answers_keep_step8_blocked_without_ocr(self) -> None:
        summary = {
            "anpr_eligible_crop_jobs": 1,
            "eligibility_exclusion_counts": {},
            "plate_metrics": {
                "raw_detector_boxes": 0,
                "accepted_plate_candidates": 0,
                "ocr_non_empty_outputs": 0,
                "by_crop_role": {},
            },
        }
        answers = runner._direct_answer_summary(summary)
        self.assertEqual(answers["ready_for_step8_verification"], "no, improve crop selection before Step 8")
        self.assertEqual(answers["likely_failure_area"], "plate_visibility_or_plate_detection_on_selected_vehicle_crops")


if __name__ == "__main__":
    unittest.main()
