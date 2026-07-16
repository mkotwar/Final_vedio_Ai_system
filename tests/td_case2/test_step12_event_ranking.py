from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_checks import write_json
from step_12_event_candidate_ranking import run_event_candidate_ranking


class Step12CriticalEventRankingTests(unittest.TestCase):
    def test_step11_5_yes_accident_is_preserved_even_with_low_candidate_score(self) -> None:
        run_dir = Path(tempfile.mkdtemp(prefix="td_case2_step12_"))
        write_json(
            run_dir / "11_5_vlm_filtered_event_candidates.json",
            {
                "status": "success",
                "candidate_events": [
                    {
                        "candidate_event_id": "scene_evt_critical",
                        "event_type": "sudden_stop",
                        "best_timestamp_seconds": 7.0,
                        "best_timestamp_text": "00:07",
                        "context_start_seconds": 6.0,
                        "context_end_seconds": 8.0,
                        "context_duration_seconds": 2.0,
                        "candidate_score": 0.22,
                        "confidence_label": "low",
                        "severity_label": "medium",
                        "trigger_reasons": ["vehicle_close_interaction"],
                        "involved_track_ids": ["vehicle_track_0001"],
                        "involved_classes": ["car"],
                        "scene_evidence": {"vehicle_count_max": 2},
                        "representative_frame": {"image_path": "frame.jpg"},
                        "full_frame_paths": ["frame.jpg"],
                        "needs_vlm_review": True,
                        "final_event_truth": "unknown_candidate_only",
                        "vlm_filter": {
                            "decision": "yes",
                            "visible_event_type": "accident",
                            "short_reason": "A visible accident is present.",
                        },
                    }
                ],
            },
        )
        write_json(run_dir / "11_full_scene_event_candidates.json", {"candidate_events": []})

        ranked_payload, selected_payload, report_payload, _flat = run_event_candidate_ranking(
            run_dir=run_dir,
            ranking_config={
                "top_k": 1,
                "min_ranking_score": 0.5,
                "min_temporal_gap_seconds": 1.0,
                "max_per_event_type": 1,
                "max_per_time_cluster": 1,
                "prefer_traffic_safety": True,
                "include_low_confidence": True,
                "save_flat": True,
                "require_full_frame_path": True,
            },
        )

        self.assertEqual(ranked_payload["summary"]["critical_event_count"], 1)
        self.assertEqual(selected_payload["selected_count"], 1)
        self.assertEqual(selected_payload["selected_candidates"][0]["candidate_event_id"], "scene_evt_critical")
        self.assertEqual(report_payload["critical_event_count"], 1)


if __name__ == "__main__":
    unittest.main()
