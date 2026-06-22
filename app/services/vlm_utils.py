import json
import re
from typing import Dict, Any, List
from loguru import logger

from app.core.utils import format_timestamp_human


def format_timestamp_human_vlm(seconds: float) -> str:
    """Converts float seconds to playback timestamp formatted as HH:MM:SS."""
    return format_timestamp_human(seconds)


def clean_json_response(raw_response: str) -> str:
    """Strips markdown block wraps (e.g. ```json ... ```) from VLM answers robustly."""
    cleaned = raw_response.strip()
    # Find json/markdown code blocks if present
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if json_match:
        cleaned = json_match.group(1).strip()

    # Strip any extra outer backticks just in case
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    # Remove trailing commentaries often generated outside markdown block
    bracket_match = re.search(r"([\[\{].*[\]\}])", cleaned, re.DOTALL)
    if bracket_match:
        cleaned = bracket_match.group(1).strip()

    try:
        import json_repair
        parsed = json_repair.repair_json(cleaned, return_objects=True)
        if not isinstance(parsed, (dict, list)):
            parsed = json.loads(cleaned)
    except Exception:
        try:
            parsed = json.loads(cleaned)
        except Exception:
            # Fallback: extract the first curly brace block
            dict_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
            if dict_match:
                try:
                    import json_repair
                    parsed = json_repair.repair_json(dict_match.group(1), return_objects=True)
                    if not isinstance(parsed, (dict, list)):
                        parsed = json.loads(dict_match.group(1))
                except Exception:
                    parsed = json.loads(dict_match.group(1).strip())
            else:
                raise

    # Handle list-wrapped responses
    if isinstance(parsed, list) and len(parsed) > 0:
        reconstructed = {}
        is_field_list = False
        field_keys = {
            "scene_type", "scene_description", "objects", "people_count",
            "activities", "keywords", "caption", "events",
        }

        for item in parsed:
            if isinstance(item, dict) and "type" in item:
                t = str(item["type"]).lower()
                if t in field_keys or t == "location" or t == "time" or t == "environment":
                    is_field_list = True
                    key_map = {
                        "location": "scene_type",
                        "time": "keywords",
                        "environment": "scene_description",
                    }
                    schema_key = key_map.get(t, t)

                    val = None
                    for field in [
                        "description", "value", "count", "summary", "summary_text",
                        "summary_caption", "text_summary", "tags", "search_tags", "counts",
                    ]:
                        if field in item:
                            val = item[field]
                            break
                    if val is None:
                        val = item.get("attributes")

                    if val is not None:
                        reconstructed[schema_key] = val

        if is_field_list:
            return json.dumps(reconstructed)

        # Check if it's a simple list of detected objects
        first_item = parsed[0]
        if isinstance(first_item, dict) and (
            "type" in first_item or "subtype" in first_item or "color" in first_item
        ):
            reconstructed = {
                "objects": parsed,
                "events": [],
                "scene_type": "unknown",
                "scene_description": "Detected objects in the frame.",
                "caption": "A frame containing several objects.",
                "people_count": sum(
                    1 for x in parsed
                    if isinstance(x, dict) and "person" in str(x.get("type", "")).lower()
                ),
                "activities": [],
                "keywords": list(set(
                    str(x.get("type", "")) for x in parsed
                    if isinstance(x, dict) and x.get("type")
                )),
            }
            return json.dumps(reconstructed)

        if isinstance(first_item, dict):
            return json.dumps(first_item)

    return json.dumps(parsed)


def normalize_metadata_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Normalizes and repairs the parsed JSON dict to strictly match FrameRichMetadata schema."""
    # Ensure base string fields exist
    if "scene_type" not in parsed or not parsed["scene_type"]:
        parsed["scene_type"] = "unknown"
    if "scene_description" not in parsed or not parsed["scene_description"]:
        parsed["scene_description"] = ""
    if "caption" not in parsed or not parsed["caption"]:
        parsed["caption"] = "No description available."

    # Cross-populate scene_description and caption if one is empty
    if not parsed["scene_description"] and parsed["caption"]:
        parsed["scene_description"] = parsed["caption"]
    elif parsed["scene_description"] and not parsed["caption"]:
        parsed["caption"] = parsed["scene_description"]

    # Ensure lists are lists of strings
    for list_field in ["activities", "keywords"]:
        val = parsed.get(list_field)
        if val is None:
            parsed[list_field] = []
        elif isinstance(val, str):
            parsed[list_field] = [s.strip() for s in val.split(",") if s.strip()]
        elif not isinstance(val, list):
            parsed[list_field] = [str(val)]
        else:
            cleaned_list = []
            for item in val:
                if isinstance(item, dict):
                    label = (
                        item.get("type")
                        or item.get("relation")
                        or item.get("activity")
                        or item.get("description")
                        or ""
                    )
                    label_str = str(label).strip()
                    if label_str:
                        cleaned_list.append(label_str)
                else:
                    cleaned_list.append(str(item))
            parsed[list_field] = cleaned_list

    # Ensure people_count is int
    pc = parsed.get("people_count")
    if pc is None:
        parsed["people_count"] = 0
    else:
        try:
            parsed["people_count"] = int(pc)
        except (ValueError, TypeError):
            parsed["people_count"] = 0

    # Ensure objects list exists and is normalized
    objs = parsed.get("objects")
    if not isinstance(objs, list):
        objs = []

    normalized_objs = []
    for obj in objs:
        if not isinstance(obj, dict):
            continue

        # Map camelCase subType to lowercase subtype if needed
        sub_type = obj.get("subtype")
        if sub_type is None:
            sub_type = obj.get("subType", "")

        obj_id = str(obj.get("id", "")).strip()
        obj_type = str(obj.get("type", "unknown"))
        obj_subtype = str(sub_type).lower().strip()

        # Normalize subtypes to prevent oscillation
        if obj_subtype in ["adult male", "male", "individual", "pedestrian", "visitor", "man", "woman", "female", "guard", "security"]:
            obj_subtype = "person"
        elif obj_subtype in ["shopper"]:
            obj_subtype = "customer"
        elif obj_subtype in ["staff", "worker"]:
            obj_subtype = "employee"

        obj_color = str(obj.get("color", ""))
        obj_condition = str(obj.get("condition", "normal")).lower().strip()

        # Normalize attributes list
        attrs = obj.get("attributes")
        if attrs is None:
            attrs_list = []
        elif isinstance(attrs, str):
            attrs_list = [s.strip() for s in attrs.split(",") if s.strip()]
        elif isinstance(attrs, list):
            attrs_list = []
            for attr in attrs:
                if isinstance(attr, dict):
                    dict_parts = [f"{k}: {v}" if v else k for k, v in attr.items()]
                    attrs_list.append(", ".join(dict_parts))
                else:
                    attrs_list.append(str(attr))
        else:
            attrs_list = [str(attrs)]

        normalized_objs.append({
            "id": obj_id,
            "type": obj_type,
            "subtype": obj_subtype,
            "color": obj_color,
            "condition": obj_condition,
            "attributes": attrs_list,
        })

    parsed["objects"] = normalized_objs

    # ── Normalize events list ──────────────────────────────────────────────
    events = parsed.get("events")
    if not isinstance(events, list):
        events = []

    valid_severities = {"low", "medium", "high", "critical"}
    normalized_events = []

    for evt in events:
        if not isinstance(evt, dict):
            continue

        event_type = str(evt.get("event_type", "unknown")).lower().strip()
        description = str(evt.get("description", "")).strip()
        severity = str(evt.get("severity", "medium")).lower().strip()

        # Normalize actors to list of strings
        actors = evt.get("actors", [])
        if isinstance(actors, str):
            actors = [a.strip() for a in actors.split(",") if a.strip()]
        elif not isinstance(actors, list):
            actors = []
        else:
            actors = [str(a) for a in actors]

        # Fallback severity if unrecognized value
        if severity not in valid_severities:
            severity = "medium"

        # Skip placeholder "none" events
        if event_type and event_type != "none":
            normalized_events.append({
                "event_type": event_type,
                "description": description,
                "actors": actors,
                "severity": severity,
            })

    parsed["events"] = normalized_events

    # Pre-merge event types into activities so ActivityRecoveryService
    # doesn't overwrite them with generic fallbacks
    if normalized_events:
        existing_activities = parsed.get("activities", [])
        event_activity_labels = [
            e["event_type"].replace("_", " ") for e in normalized_events
        ]
        # Merge preserving order, no duplicates
        merged = list(dict.fromkeys(existing_activities + event_activity_labels))
        parsed["activities"] = merged
    # ── End events normalization ───────────────────────────────────────────

    return parsed


def generate_search_text(meta: Dict[str, Any]) -> str:
    """Autogenerates search indexing block by joining structural text descriptors."""
    parts = [
        meta.get("scene_type", ""),
        meta.get("scene_description", ""),
        meta.get("caption", ""),
        ", ".join(meta.get("activities", [])),
        ", ".join(meta.get("keywords", [])),
    ]

    # Index event descriptions and types for searchability
    for evt in meta.get("events", []):
        parts.append(evt.get("event_type", ""))
        parts.append(evt.get("description", ""))

    # Append OCR detected text if present
    ocr_data = meta.get("ocr")
    if ocr_data:
        if isinstance(ocr_data, dict):
            detected = ocr_data.get("detected_text", [])
        else:
            detected = getattr(ocr_data, "detected_text", [])
        parts.extend(detected)

    # Append object specifics
    for obj in meta.get("objects", []):
        color = obj.get("color", "")
        subtype = obj.get("subtype", "")
        parts.append(f"{color} {subtype}".strip())
        parts.extend(obj.get("attributes", []))

    full_text = " ".join(parts).lower()
    cleaned_text = re.sub(r"\s+", " ", full_text).strip()
    return cleaned_text
