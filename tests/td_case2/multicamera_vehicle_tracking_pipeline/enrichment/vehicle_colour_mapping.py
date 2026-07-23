from __future__ import annotations


SUPPORTED_VEHICLE_COLOURS = (
    "BLACK",
    "WHITE",
    "SILVER",
    "GREY",
    "RED",
    "BLUE",
    "GREEN",
    "YELLOW",
    "ORANGE",
    "BROWN",
    "BEIGE",
    "PURPLE",
    "UNKNOWN",
)

_ALIASES = {
    "gray": "GREY",
    "grey": "GREY",
    "dark grey": "GREY",
    "dark gray": "GREY",
    "light grey": "SILVER",
    "light gray": "SILVER",
    "silver": "SILVER",
    "navy": "BLUE",
    "maroon": "RED",
    "cream": "BEIGE",
    "gold": "YELLOW",
    "violet": "PURPLE",
    "black": "BLACK",
    "white": "WHITE",
    "red": "RED",
    "blue": "BLUE",
    "green": "GREEN",
    "yellow": "YELLOW",
    "orange": "ORANGE",
    "brown": "BROWN",
    "beige": "BEIGE",
    "purple": "PURPLE",
    "unknown": "UNKNOWN",
}


def normalize_vehicle_colour(value: str) -> str:
    normalized = " ".join(str(value).strip().lower().replace("_", " ").replace("-", " ").split())
    if not normalized:
        return "UNKNOWN"
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    upper = normalized.upper()
    if upper in SUPPORTED_VEHICLE_COLOURS:
        return upper
    return "UNKNOWN"

