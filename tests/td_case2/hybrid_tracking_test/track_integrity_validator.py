from __future__ import annotations

from collections import Counter
from typing import Any


def build_track_integrity_report(
    rebuilt_tracks: list[dict[str, Any]],
    drift_reports: list[dict[str, Any]],
    sanitized_tracks: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    drift_by_track_id = {int(item.get("track_id", 0)): item for item in drift_reports}
    sanitized_by_track_id = {int(item.get("track_id", 0)): item for item in sanitized_tracks}
    flag_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    tracks_payload: list[dict[str, Any]] = []
    for rebuilt in rebuilt_tracks:
        track_id = int(rebuilt.get("track_id", 0))
        drift = drift_by_track_id.get(track_id, {})
        sanitized = sanitized_by_track_id.get(track_id, {})
        combined_flags = sorted(
            set(list(rebuilt.get("integrity_flags", [])) + list(drift.get("flags", [])) + list(sanitized.get("drift_flags", [])))
        )
        for flag in combined_flags:
            flag_counter[flag] += 1
        status_counter[str(sanitized.get("track_integrity_status", "unknown"))] += 1
        tracks_payload.append(
            {
                "track_id": track_id,
                "original_summary": dict(rebuilt.get("original_summary", {})),
                "rebuilt_timeline": dict(rebuilt.get("rebuilt_metadata", {})),
                "integrity_flags": combined_flags,
                "track_integrity_status": str(sanitized.get("track_integrity_status", "unknown")),
                "valid_observation_count": int(sanitized.get("valid_observation_count", 0) or 0),
                "invalid_observation_count": int(sanitized.get("invalid_observation_count", 0) or 0),
                "trimmed_kcf_duration_seconds": round(float(sanitized.get("trimmed_kcf_duration_seconds", 0.0) or 0.0), 6),
            }
        )
    report = {
        "status": "success",
        "track_count": len(rebuilt_tracks),
        "integrity_flag_counts": dict(sorted(flag_counter.items())),
        "track_integrity_status_counts": dict(sorted(status_counter.items())),
        "tracks": tracks_payload,
    }
    markdown_lines = [
        "# Track Timeline Integrity Report",
        "",
        f"- Tracks audited: {report['track_count']}",
    ]
    for key, value in sorted(report["integrity_flag_counts"].items()):
        markdown_lines.append(f"- {key}: {value}")
    for key, value in sorted(report["track_integrity_status_counts"].items()):
        markdown_lines.append(f"- {key}: {value}")
    return report, "\n".join(markdown_lines)


__all__ = ["build_track_integrity_report"]
