from __future__ import annotations

SUPPORTED_TRACK_MEDIA_ROLES = (
    "BEST_OVERALL",
    "FIRST",
    "MIDDLE",
    "LAST",
    "HIGHEST_CONFIDENCE",
    "LARGEST",
    "SHARPEST",
)

ROLE_TO_MEDIA_TYPE = {
    "BEST_OVERALL": "BEST_VEHICLE_CROP",
    "FIRST": "VEHICLE_CROP",
    "MIDDLE": "VEHICLE_CROP",
    "LAST": "VEHICLE_CROP",
    "HIGHEST_CONFIDENCE": "VEHICLE_CROP",
    "LARGEST": "VEHICLE_CROP",
    "SHARPEST": "VEHICLE_CROP",
}

ROLE_TO_SELECTION_RANK = {
    "BEST_OVERALL": 1,
    "FIRST": 10,
    "MIDDLE": 20,
    "LAST": 30,
    "HIGHEST_CONFIDENCE": 40,
    "LARGEST": 50,
    "SHARPEST": 60,
}


def normalize_track_media_role(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in SUPPORTED_TRACK_MEDIA_ROLES:
        raise ValueError(f"Unsupported track media role: {value}")
    return normalized


def normalize_track_media_type(value: str) -> str:
    role = normalize_track_media_role(value)
    return ROLE_TO_MEDIA_TYPE[role]
