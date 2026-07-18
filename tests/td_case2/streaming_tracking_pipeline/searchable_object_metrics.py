from __future__ import annotations

from collections import Counter
from typing import Any

from .searchable_object_schemas import SearchableVehicleRecord


def build_searchable_object_metrics(
    records: list[SearchableVehicleRecord],
    *,
    join_failures: list[str] | None = None,
    missing_input_artifacts: list[str] | None = None,
) -> dict[str, Any]:
    record_ids = Counter(record.record_id for record in records)
    status_counts = Counter(record.plate_status for record in records)
    group_counts = Counter(record.object_group for record in records)
    warnings = Counter(warning for record in records for warning in record.warnings)
    return {
        "vehicle_records_created": group_counts["vehicle"] or len([record for record in records if record.object_group != "person"]),
        "person_records_created": group_counts["person"],
        "object_records_created": len(records),
        "verified_plate_records": status_counts["verified"],
        "weak_plate_records": status_counts["weak"],
        "invalid_plate_records": status_counts["invalid"],
        "no_plate_records": status_counts["no_plate_detected"],
        "records_with_colour": sum(1 for record in records if record.normalized_colour),
        "records_without_colour": sum(1 for record in records if not record.normalized_colour),
        "records_with_vehicle_crop": sum(1 for record in records if record.representative_vehicle_crop_path),
        "records_with_plate_crop": sum(1 for record in records if record.representative_plate_crop_path),
        "records_missing_track_times": warnings["missing_track_times"],
        "records_by_class": dict(Counter(record.object_class or "unknown" for record in records)),
        "records_by_object_group": dict(group_counts),
        "records_by_colour": dict(Counter(record.normalized_colour or "unknown" for record in records)),
        "records_by_plate_status": dict(status_counts),
        "duplicate_record_ids": sorted(record_id for record_id, count in record_ids.items() if count > 1),
        "join_failures": list(join_failures or []),
        "missing_input_artifacts": list(missing_input_artifacts or []),
        "warning_counts": dict(warnings),
    }
