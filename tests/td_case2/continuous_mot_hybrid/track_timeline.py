from __future__ import annotations

from collections import Counter
from typing import Any


def build_tracking_timeline_report(track_rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [float(item["duration_seconds"]) for item in track_rows]
    return {
        "status": "success",
        "raw_track_ids": len(track_rows),
        "confirmed_tracks": len([item for item in track_rows if bool(item["confirmed"])]),
        "tentative_tracks": len([item for item in track_rows if not bool(item["confirmed"])]),
        "tracks_under_0_5_seconds": len([item for item in track_rows if float(item["duration_seconds"]) < 0.5]),
        "tracks_under_1_second": len([item for item in track_rows if float(item["duration_seconds"]) < 1.0]),
        "duration_seconds": {
            "average": round(sum(durations) / len(durations), 6) if durations else 0.0,
            "median": round(sorted(durations)[len(durations) // 2], 6) if durations else 0.0,
            "maximum": round(max(durations), 6) if durations else 0.0,
        },
        "class_counts": dict(sorted(Counter(str(item["class_name"]) for item in track_rows).items())),
        "family_counts": dict(sorted(Counter(str(item["object_family"]) for item in track_rows).items())),
    }

