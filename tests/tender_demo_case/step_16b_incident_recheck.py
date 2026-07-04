from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any


INCIDENT_RECHECK_OUTPUT_NAME = "16b_incident_recheck_outputs.json"
INCIDENT_RECHECK_REPORT_NAME = "16b_incident_recheck_report.json"
DEFAULT_ENABLE_INCIDENT_RECHECK = False
DEFAULT_INCIDENT_RECHECK_ALL_TOPK = False
DEFAULT_INCIDENT_FOCUS = "general"
ALLOWED_INCIDENT_FOCUS = {
    "general",
    "robbery",
    "theft",
    "violence",
    "weapon",
    "traffic",
    "fall",
    "intrusion",
}

NORMAL_ACTIVITY = "normal_activity"
POSSIBLE_REVIEW_CLIP = "possible_review_clip"
PRIORITY_SUSPICIOUS_EVENT = "priority_suspicious_event"
UNCERTAIN_ACTIVITY = "uncertain_activity"

PRIMARY_EVENT_LABELS = {
    NORMAL_ACTIVITY,
    "possible_robbery",
    "possible_theft_from_display",
    "possible_weapon_visible",
    "possible_assault_or_grabbing",
    "possible_employee_threat_or_restraint",
    "possible_group_robbery",
    "possible_fight",
    "possible_collision",
    "possible_fall",
    "possible_intrusion",
    UNCERTAIN_ACTIVITY,
}
RELATED_ROBBERY_LABELS = {
    "possible_robbery",
    "possible_weapon_visible",
    "possible_assault_or_grabbing",
    "possible_employee_threat_or_restraint",
    "possible_theft_from_display",
    "possible_group_robbery",
}
CATEGORY_VALUES = {
    NORMAL_ACTIVITY,
    POSSIBLE_REVIEW_CLIP,
    PRIORITY_SUSPICIOUS_EVENT,
    UNCERTAIN_ACTIVITY,
}
RISK_VALUES = {"low", "medium", "high", "unknown"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
EVIDENCE_STRENGTH_VALUES = {"none", "weak", "medium", "strong"}
YES_NO_UNCLEAR = {"yes", "no", "unclear"}
PEOPLE_COUNT_VALUES = {"0", "1", "2", "3", "4_plus", "unclear"}

WEAPON_TERMS = {
    "weapon",
    "gun",
    "knife",
    "stick",
    "sharp object",
    "pointing object",
    "object pointed",
    "weapon-like",
}
GRABBING_TERMS = {
    "grab",
    "grabbing",
    "holding person",
    "hold person",
    "pulling",
    "restrain",
    "restraining",
    "forced to move",
    "physical struggle",
    "body contact",
    "fight",
    "assault",
}
THREAT_TERMS = {
    "threat",
    "threaten",
    "threatening",
    "controlled",
    "forced",
    "coercion",
    "restrained",
    "employee appears controlled",
    "person appears controlled",
}
DISPLAY_INTERACTION_TERMS = {
    "display",
    "display case",
    "counter",
    "glass case",
    "countertop",
    "product area",
    "case",
}
TAKING_TERMS = {
    "taking item",
    "take item",
    "takes item",
    "collecting items",
    "collect items",
    "collecting",
    "hiding item",
    "hide item",
    "conceal",
    "concealing",
    "bagging",
    "passing item",
    "quickly collecting",
    "removing item",
    "taking from display",
}
ROBBERY_TERMS = {
    "robbery",
    "group robbery",
    "coordinated movement",
    "blocking exits",
    "surrounding staff",
    "controls people",
}
TRAFFIC_TERMS = {"collision", "vehicle", "car", "truck", "motorcycle", "traffic", "road"}
FALL_TERMS = {"fall", "fallen", "injury", "collapse"}
INTRUSION_TERMS = {"intrusion", "restricted area", "unauthorized", "trespass"}
WEAK_ONLY_TERMS = {
    "walking",
    "moving",
    "standing",
    "shop",
    "person",
    "customer",
    "looking",
    "browsing",
    "near display",
    "near counter",
}


def _load_required_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Step 16B input file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> dict[str, Any] | list[dict[str, Any]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_bool_env(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _clean_text(text: Any) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"```(?:json)?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _load_tender_demo_vlm_factory():
    try:
        from tests.tender_demo_case.tender_demo_vlm_adapter import create_tender_demo_vlm
        return create_tender_demo_vlm
    except ModuleNotFoundError:
        adapter_path = Path(__file__).resolve().parent / "tender_demo_vlm_adapter.py"
        spec = importlib.util.spec_from_file_location("tender_demo_vlm_adapter", adapter_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load tender demo VLM adapter from: {adapter_path}")
        adapter_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter_module)
        return adapter_module.create_tender_demo_vlm


def _extract_json_text(raw_output: str) -> str:
    cleaned = _clean_text(raw_output)
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]
    elif first_brace != -1:
        cleaned = cleaned[first_brace:]
    return cleaned.strip()


def _repair_json_candidate(text: str) -> str:
    repaired = str(text or "").strip()
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = repaired.replace("\r", " ").replace("\n", " ")
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if repaired.count("{") > repaired.count("}"):
        repaired += "}" * (repaired.count("{") - repaired.count("}"))
    if repaired.count("[") > repaired.count("]"):
        repaired += "]" * (repaired.count("[") - repaired.count("]"))
    return re.sub(r"\s+", " ", repaired).strip()


def _dedupe_preserve_order(values: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _clean_text(value).strip(" .")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def _read_incident_focus() -> str:
    value = _clean_text(os.environ.get("TENDER_DEMO_INCIDENT_FOCUS", DEFAULT_INCIDENT_FOCUS)).lower()
    return value if value in ALLOWED_INCIDENT_FOCUS else DEFAULT_INCIDENT_FOCUS


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    for term in terms:
        escaped = re.escape(term.lower())
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        if re.search(pattern, lowered):
            return True
    return False


def _extract_yolo_classes(item: dict[str, Any]) -> list[str]:
    yolo = item.get("yolo", {})
    if not isinstance(yolo, dict):
        return []
    values: list[str] = []
    top_classes = yolo.get("top_classes", [])
    if isinstance(top_classes, list):
        for entry in top_classes:
            if isinstance(entry, dict):
                class_name = _clean_text(entry.get("class_name")).lower()
                if class_name:
                    values.append(class_name)
    return _dedupe_preserve_order(values, limit=10)


def _text_sources_for_item(item: dict[str, Any]) -> tuple[dict[str, Any], str]:
    parsed_json = item.get("parsed_json", {})
    if not isinstance(parsed_json, dict):
        parsed_json = {}
    parts = [
        _clean_text(parsed_json.get("scene_type")),
        _clean_text(parsed_json.get("caption")),
        _clean_text(parsed_json.get("description")),
        _clean_text(parsed_json.get("event_label")),
        _clean_text(parsed_json.get("motion_summary")),
        _clean_text(item.get("raw_vlm_output")),
        " ".join(str(reason) for reason in item.get("ranking_reasons", []) if str(reason).strip()),
        " ".join(_extract_yolo_classes(item)),
    ]
    return parsed_json, " ".join(part for part in parts if part).lower()


def _normalize_people_estimate(value: Any, fallback_people_count: int) -> str:
    cleaned = _clean_text(value).lower()
    if cleaned in PEOPLE_COUNT_VALUES:
        return cleaned
    if cleaned in {"4+", "4 plus", "4plus", "many"}:
        return "4_plus"
    if cleaned.isdigit():
        people_count = int(cleaned)
    else:
        people_count = fallback_people_count
    if people_count <= 0:
        return "0"
    if people_count == 1:
        return "1"
    if people_count == 2:
        return "2"
    if people_count == 3:
        return "3"
    return "4_plus"


def _normalize_yes_no_unclear(value: Any, fallback: str = "unclear") -> str:
    cleaned = _clean_text(value).lower()
    if cleaned in YES_NO_UNCLEAR:
        return cleaned
    if cleaned in {"true", "1", "present", "visible"}:
        return "yes"
    if cleaned in {"false", "0", "absent", "not visible"}:
        return "no"
    return fallback


def _extract_secondary_labels(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    labels: list[str] = []
    for item in value:
        label = _clean_text(item).lower().replace(" ", "_")
        if label in PRIMARY_EVENT_LABELS and label not in {NORMAL_ACTIVITY, UNCERTAIN_ACTIVITY}:
            labels.append(label)
    return _dedupe_preserve_order(labels, limit=5)


def _keyword_evidence_from_text(text: str) -> list[str]:
    evidence: list[str] = []
    if _contains_any(text, WEAPON_TERMS):
        evidence.append("possible weapon-like object or threatening object posture")
    if _contains_any(text, GRABBING_TERMS):
        evidence.append("possible grabbing, pulling, restraint, or physical struggle")
    if _contains_any(text, THREAT_TERMS):
        evidence.append("possible threat, control, or coercive posture")
    if _contains_any(text, DISPLAY_INTERACTION_TERMS):
        evidence.append("display or counter interaction is visible")
    if _contains_any(text, TAKING_TERMS):
        evidence.append("possible taking, collecting, hiding, or passing of items")
    if _contains_any(text, ROBBERY_TERMS):
        evidence.append("multiple people may be coordinating around a target area")
    if _contains_any(text, FALL_TERMS):
        evidence.append("possible fall or injury-related posture")
    if _contains_any(text, TRAFFIC_TERMS):
        evidence.append("traffic or vehicle activity is visible")
    if _contains_any(text, INTRUSION_TERMS):
        evidence.append("possible intrusion or restricted-area behavior")
    return _dedupe_preserve_order(evidence, limit=8)


def _baseline_incident_recheck(item: dict[str, Any], incident_focus: str) -> dict[str, Any]:
    parsed_json, text = _text_sources_for_item(item)
    yolo_classes = _extract_yolo_classes(item)
    people_count = _safe_int(parsed_json.get("people_count"), _safe_int(item.get("yolo", {}).get("person_max"), 0))
    scene_type = _clean_text(parsed_json.get("scene_type")).lower() or "unknown"
    caption = _clean_text(parsed_json.get("caption"))
    description = _clean_text(parsed_json.get("description")) or _clean_text(item.get("raw_vlm_output"))
    visible_evidence = _keyword_evidence_from_text(text)

    display_interaction = "yes" if _contains_any(text, DISPLAY_INTERACTION_TERMS) else "unclear"
    taking_items = "yes" if _contains_any(text, TAKING_TERMS) else "unclear"
    weapon_visible = "yes" if _contains_any(text, WEAPON_TERMS) else "unclear"
    grabbing = "yes" if _contains_any(text, GRABBING_TERMS) else "unclear"
    threatened = "yes" if _contains_any(text, THREAT_TERMS) else "unclear"

    record = {
        "incident_category": NORMAL_ACTIVITY,
        "primary_event_label": NORMAL_ACTIVITY,
        "secondary_event_labels": [],
        "risk_level": "low",
        "confidence": "low",
        "evidence_strength": "none",
        "incident_score": 0.1,
        "suspicious_activity": "no",
        "weapon_visible": weapon_visible,
        "weapon_description": "possible weapon-like object mentioned or implied" if weapon_visible == "yes" else "",
        "person_grabbing_or_restraining": grabbing,
        "grabbing_description": "possible grabbing, pulling, or restraint posture" if grabbing == "yes" else "",
        "person_threatened_or_controlled": threatened,
        "threat_description": "possible threatening or controlling posture" if threatened == "yes" else "",
        "taking_items_visible": taking_items,
        "item_taking_description": "possible taking, collecting, hiding, or passing of items" if taking_items == "yes" else "",
        "display_or_counter_interaction": display_interaction,
        "display_interaction_description": "possible reaching or handling near a display or counter" if display_interaction == "yes" else "",
        "people_involved_estimate": _normalize_people_estimate(parsed_json.get("people_count"), people_count),
        "visible_evidence": visible_evidence,
        "normal_explanation": "This clip may show routine walking, browsing, or normal customer/staff movement.",
        "suspicious_explanation": "",
        "review_reason": "This clip appears to show routine activity.",
        "description": description or caption or "No useful description was produced for this clip.",
        "incident_focus": incident_focus,
        "scene_type": scene_type,
        "event_label": NORMAL_ACTIVITY,
    }

    if weapon_visible == "yes":
        record.update(
            {
                "incident_category": PRIORITY_SUSPICIOUS_EVENT,
                "primary_event_label": "possible_weapon_visible",
                "risk_level": "high",
                "confidence": "medium",
                "evidence_strength": "strong",
                "incident_score": 0.9,
                "suspicious_activity": "yes",
                "suspicious_explanation": "A weapon-like object or threatening object posture may be visible.",
                "review_reason": "Priority review recommended because a weapon-like object may be visible.",
            }
        )
    elif grabbing == "yes" or threatened == "yes":
        label = "possible_assault_or_grabbing" if grabbing == "yes" else "possible_employee_threat_or_restraint"
        record.update(
            {
                "incident_category": PRIORITY_SUSPICIOUS_EVENT,
                "primary_event_label": label,
                "risk_level": "high",
                "confidence": "medium",
                "evidence_strength": "strong",
                "incident_score": 0.82,
                "suspicious_activity": "yes",
                "suspicious_explanation": "Visible body posture may indicate grabbing, restraint, threat, or control.",
                "review_reason": "Priority review recommended because a person may be grabbing, restraining, or threatening another person.",
            }
        )
    elif taking_items == "yes":
        label = "possible_robbery" if incident_focus == "robbery" else "possible_theft_from_display"
        record.update(
            {
                "incident_category": POSSIBLE_REVIEW_CLIP,
                "primary_event_label": label,
                "risk_level": "medium",
                "confidence": "medium",
                "evidence_strength": "medium",
                "incident_score": 0.72,
                "suspicious_activity": "yes",
                "suspicious_explanation": "Visible handling suggests a person may be taking, collecting, hiding, or passing items.",
                "review_reason": "Review recommended because a person may be taking or handling items from a display or counter.",
            }
        )
    elif display_interaction == "yes":
        record.update(
            {
                "incident_category": POSSIBLE_REVIEW_CLIP,
                "primary_event_label": "possible_theft_from_display",
                "risk_level": "medium",
                "confidence": "low",
                "evidence_strength": "weak",
                "incident_score": 0.56,
                "suspicious_activity": "yes",
                "suspicious_explanation": "A person appears to interact with a display or counter area in a way that may need review.",
                "review_reason": "Review recommended because visible display/counter interaction may be relevant.",
            }
        )
    elif incident_focus == "traffic" and (_contains_any(text, TRAFFIC_TERMS) or any(term in yolo_classes for term in ["car", "truck", "bus", "motorcycle", "bicycle"])):
        record.update(
            {
                "incident_category": POSSIBLE_REVIEW_CLIP if "collision" in text else NORMAL_ACTIVITY,
                "primary_event_label": "possible_collision" if "collision" in text else NORMAL_ACTIVITY,
                "risk_level": "medium" if "collision" in text else "low",
                "confidence": "low",
                "evidence_strength": "weak" if "collision" in text else "none",
                "incident_score": 0.52 if "collision" in text else 0.15,
                "suspicious_activity": "yes" if "collision" in text else "no",
                "review_reason": "Review recommended because traffic interaction may be relevant." if "collision" in text else "This clip appears to show routine traffic activity.",
            }
        )
    elif incident_focus == "fall" and _contains_any(text, FALL_TERMS):
        record.update(
            {
                "incident_category": POSSIBLE_REVIEW_CLIP,
                "primary_event_label": "possible_fall",
                "risk_level": "medium",
                "confidence": "low",
                "evidence_strength": "weak",
                "incident_score": 0.6,
                "suspicious_activity": "yes",
                "review_reason": "Review recommended because a fall or injury-related posture may be visible.",
            }
        )
    elif incident_focus == "intrusion" and _contains_any(text, INTRUSION_TERMS):
        record.update(
            {
                "incident_category": POSSIBLE_REVIEW_CLIP,
                "primary_event_label": "possible_intrusion",
                "risk_level": "medium",
                "confidence": "low",
                "evidence_strength": "weak",
                "incident_score": 0.58,
                "suspicious_activity": "yes",
                "review_reason": "Review recommended because intrusion or restricted-area behavior may be visible.",
            }
        )

    record["event_label"] = record["primary_event_label"]
    return record


def _build_incident_prompt(item: dict[str, Any], incident_focus: str) -> str:
    parsed_json, _ = _text_sources_for_item(item)
    yolo_classes = _extract_yolo_classes(item)
    ranking_reasons = [str(reason) for reason in item.get("ranking_reasons", []) if str(reason).strip()]
    motion_score = _safe_float(item.get("motion", {}).get("clip_motion_score", item.get("motion", {}).get("clip_score", 0.0)))
    baseline = _baseline_incident_recheck(item, incident_focus)

    prompt = f"""You are reviewing CCTV evidence for possible serious incidents.

The image may be a 3-panel temporal strip:
PREVIOUS | CURRENT | NEXT.

Focus on the CURRENT panel.
Use PREVIOUS and NEXT only to understand movement, escalation, and interaction.

Incident focus mode: {incident_focus}

You must classify the clip carefully.

Do not only describe walking direction.
Look for serious actions, human interaction, weapons, threats, and theft behavior.

Important incident checks:

1. Robbery / theft from display:
- person reaching into display case or counter
- person taking items from display/counter
- person handling items unusually
- person passing items to another person
- person quickly collecting items
- person hiding or concealing items
- multiple people coordinating around display/counter

2. Weapon / threat:
- visible gun, knife, stick, sharp object, or weapon-like object
- person pointing an object at another person
- person holding object in threatening way
- people reacting to threat or moving away
- employee/person appears controlled or forced

3. Assault / grabbing / restraint:
- one person grabbing another person
- holding or pulling an employee/person
- physical struggle
- person forced to move
- person being restrained
- fight-like body contact

4. Group robbery behavior:
- multiple people entering/standing around target area
- one person controls people while another takes items
- coordinated movement around display/counter
- people blocking exits or surrounding staff

5. Normal activity:
- normal walking
- standing
- browsing
- staff/customer interaction
- person near display without suspicious hand action
- ordinary shop movement

Decision rules:
- Do not mark a clip suspicious only because it is in a shop.
- Do not mark a clip suspicious only because a person is walking or standing.
- If there is visible grabbing, restraint, threat posture, weapon-like object, or taking items, mark it as review-worthy or priority.
- If a weapon is clearly visible or a person is being grabbed/threatened, mark priority_suspicious_event.
- If a person clearly reaches into or takes from a display/counter, mark possible_review_clip or priority depending on strength.
- If evidence is weak but relevant, mark possible_review_clip.
- If unsure, use uncertain_activity and explain what is unclear.
- Do not claim confirmed robbery unless very clear.
- Use "possible robbery" or "review recommended" when evidence is not conclusive.

Return ONLY valid JSON.
No markdown.
No explanation outside JSON.

Schema:
{{
"incident_category": "normal_activity|possible_review_clip|priority_suspicious_event|uncertain_activity",
"primary_event_label": "normal_activity|possible_robbery|possible_theft_from_display|possible_weapon_visible|possible_assault_or_grabbing|possible_employee_threat_or_restraint|possible_group_robbery|possible_fight|possible_collision|possible_fall|possible_intrusion|uncertain_activity",
"secondary_event_labels": ["short labels if more than one event may be present"],
"risk_level": "low|medium|high|unknown",
"confidence": "low|medium|high",
"evidence_strength": "none|weak|medium|strong",
"incident_score": 0.0,
"suspicious_activity": "yes|no|unclear",
"weapon_visible": "yes|no|unclear",
"weapon_description": "short factual description or empty string",
"person_grabbing_or_restraining": "yes|no|unclear",
"grabbing_description": "short factual description or empty string",
"person_threatened_or_controlled": "yes|no|unclear",
"threat_description": "short factual description or empty string",
"taking_items_visible": "yes|no|unclear",
"item_taking_description": "short factual description or empty string",
"display_or_counter_interaction": "yes|no|unclear",
"display_interaction_description": "short factual description or empty string",
"people_involved_estimate": "0|1|2|3|4_plus|unclear",
"visible_evidence": ["short visible evidence item"],
"normal_explanation": "why this may be normal, if applicable",
"suspicious_explanation": "why this may be suspicious, if applicable",
"review_reason": "one sentence explaining why reviewer should or should not inspect this clip",
"description": "one factual sentence describing what is visible in the current panel"
}}"""

    context_lines = [
        f"Earlier Step 16 scene_type: {parsed_json.get('scene_type', 'unknown')}",
        f"Earlier Step 16 caption: {_clean_text(parsed_json.get('caption')) or 'unknown'}",
        f"Earlier Step 16 description: {_clean_text(parsed_json.get('description')) or 'unknown'}",
        f"Earlier Step 16 event_label: {parsed_json.get('event_label', 'unknown')}",
        f"Earlier Step 16 suspicious_activity: {parsed_json.get('suspicious_activity', 'unclear')}",
        f"YOLO classes: {', '.join(yolo_classes) if yolo_classes else 'unknown'}",
        f"Ranking reasons: {', '.join(ranking_reasons) if ranking_reasons else 'none'}",
        f"Motion score: {motion_score:.3f}",
        f"Baseline review hint: {baseline.get('review_reason', 'none')}",
    ]
    return prompt + "\n\nContext from earlier fast-pass analysis:\n" + "\n".join(context_lines)


def _normalize_primary_event_label(value: Any, incident_focus: str, text: str) -> str:
    cleaned = _clean_text(value).lower().replace(" ", "_")
    label_aliases = {
        "possible_theft_or_shoplifting": "possible_theft_from_display",
        "possible_shoplifting": "possible_theft_from_display",
        "possible_employee_threat": "possible_employee_threat_or_restraint",
        "possible_grabbing": "possible_assault_or_grabbing",
        "possible_weapon": "possible_weapon_visible",
        "traffic_activity": "possible_collision" if "collision" in text else NORMAL_ACTIVITY,
    }
    cleaned = label_aliases.get(cleaned, cleaned)
    if cleaned in PRIMARY_EVENT_LABELS:
        return cleaned
    if incident_focus == "traffic" and _contains_any(text, TRAFFIC_TERMS):
        return "possible_collision" if "collision" in text else NORMAL_ACTIVITY
    if incident_focus == "fall" and _contains_any(text, FALL_TERMS):
        return "possible_fall"
    if incident_focus == "intrusion" and _contains_any(text, INTRUSION_TERMS):
        return "possible_intrusion"
    return UNCERTAIN_ACTIVITY


def _evidence_strength_rank(value: str) -> int:
    return {"none": 0, "weak": 1, "medium": 2, "strong": 3}.get(value, 0)


def _contains_concrete_evidence(payload: dict[str, Any], text: str) -> bool:
    if payload.get("weapon_visible") == "yes":
        return True
    if payload.get("person_grabbing_or_restraining") == "yes":
        return True
    if payload.get("person_threatened_or_controlled") == "yes":
        return True
    if payload.get("taking_items_visible") == "yes":
        return True
    display_interaction = payload.get("display_or_counter_interaction") == "yes"
    if display_interaction and _contains_any(text, {"reach", "reaching", "take", "taking", "handle", "handling", "item", "collect", "conceal"}):
        return True
    concrete_terms = {
        "grab",
        "grabbing",
        "hold",
        "holding",
        "pull",
        "pulling",
        "restrain",
        "force",
        "threaten",
        "weapon",
        "gun",
        "knife",
        "object pointed",
        "reach",
        "reaching",
        "display",
        "counter",
        "case",
        "take",
        "taking",
        "item",
        "product",
        "hide",
        "conceal",
        "collect",
        "bag",
    }
    return _contains_any(text, concrete_terms)


def _is_weak_only_evidence(text: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9_]+", text.lower()) if token}
    if not tokens:
        return True
    if _contains_any(text, WEAPON_TERMS | GRABBING_TERMS | THREAT_TERMS | TAKING_TERMS | FALL_TERMS | INTRUSION_TERMS):
        return False
    return all(
        token in {
            "walking",
            "moving",
            "standing",
            "shop",
            "person",
            "customer",
            "looking",
            "browsing",
            "near",
            "display",
            "counter",
            "case",
            "staff",
            "employee",
        }
        for token in tokens
    )


def _normalize_incident_output(parsed: dict[str, Any], item: dict[str, Any], incident_focus: str) -> dict[str, Any]:
    baseline = _baseline_incident_recheck(item, incident_focus)
    parsed_json, source_text = _text_sources_for_item(item)
    yolo_classes = _extract_yolo_classes(item)
    merged_text_parts = [
        source_text,
        _clean_text(parsed.get("description")),
        _clean_text(parsed.get("suspicious_explanation")),
        _clean_text(parsed.get("normal_explanation")),
        _clean_text(parsed.get("review_reason")),
        _clean_text(parsed.get("weapon_description")),
        _clean_text(parsed.get("grabbing_description")),
        _clean_text(parsed.get("threat_description")),
        _clean_text(parsed.get("item_taking_description")),
        _clean_text(parsed.get("display_interaction_description")),
        " ".join(str(item) for item in parsed.get("visible_evidence", []) if str(item).strip()) if isinstance(parsed.get("visible_evidence"), list) else "",
    ]
    merged_text = " ".join(part for part in merged_text_parts if part).lower()

    yolo_person_max = _safe_int(item.get("yolo", {}).get("person_max"), 0)
    people_count = _safe_int(parsed_json.get("people_count"), yolo_person_max)

    normalized = {
        "incident_category": _clean_text(parsed.get("incident_category")).lower(),
        "primary_event_label": _normalize_primary_event_label(parsed.get("primary_event_label") or parsed.get("event_label"), incident_focus, merged_text),
        "secondary_event_labels": _extract_secondary_labels(parsed.get("secondary_event_labels", [])),
        "risk_level": _clean_text(parsed.get("risk_level")).lower(),
        "confidence": _clean_text(parsed.get("confidence")).lower(),
        "evidence_strength": _clean_text(parsed.get("evidence_strength")).lower(),
        "incident_score": max(0.0, min(1.0, _safe_float(parsed.get("incident_score"), baseline.get("incident_score", 0.1)))),
        "suspicious_activity": _normalize_yes_no_unclear(parsed.get("suspicious_activity"), baseline.get("suspicious_activity", "unclear")),
        "weapon_visible": _normalize_yes_no_unclear(parsed.get("weapon_visible"), baseline.get("weapon_visible", "unclear")),
        "weapon_description": _clean_text(parsed.get("weapon_description")),
        "person_grabbing_or_restraining": _normalize_yes_no_unclear(parsed.get("person_grabbing_or_restraining"), baseline.get("person_grabbing_or_restraining", "unclear")),
        "grabbing_description": _clean_text(parsed.get("grabbing_description")),
        "person_threatened_or_controlled": _normalize_yes_no_unclear(parsed.get("person_threatened_or_controlled"), baseline.get("person_threatened_or_controlled", "unclear")),
        "threat_description": _clean_text(parsed.get("threat_description")),
        "taking_items_visible": _normalize_yes_no_unclear(parsed.get("taking_items_visible"), baseline.get("taking_items_visible", "unclear")),
        "item_taking_description": _clean_text(parsed.get("item_taking_description")),
        "display_or_counter_interaction": _normalize_yes_no_unclear(parsed.get("display_or_counter_interaction"), baseline.get("display_or_counter_interaction", "unclear")),
        "display_interaction_description": _clean_text(parsed.get("display_interaction_description")),
        "people_involved_estimate": _normalize_people_estimate(parsed.get("people_involved_estimate"), people_count),
        "visible_evidence": _dedupe_preserve_order([str(item) for item in parsed.get("visible_evidence", [])] if isinstance(parsed.get("visible_evidence"), list) else [], limit=8),
        "normal_explanation": _clean_text(parsed.get("normal_explanation")),
        "suspicious_explanation": _clean_text(parsed.get("suspicious_explanation")),
        "review_reason": _clean_text(parsed.get("review_reason")),
        "description": _clean_text(parsed.get("description")) or baseline.get("description", ""),
        "incident_focus": incident_focus,
        "scene_type": baseline.get("scene_type", "unknown"),
    }

    if normalized["incident_category"] not in CATEGORY_VALUES:
        normalized["incident_category"] = baseline["incident_category"]
    if normalized["risk_level"] not in RISK_VALUES:
        normalized["risk_level"] = baseline["risk_level"]
    if normalized["confidence"] not in CONFIDENCE_VALUES:
        normalized["confidence"] = baseline["confidence"]
    if normalized["evidence_strength"] not in EVIDENCE_STRENGTH_VALUES:
        normalized["evidence_strength"] = baseline["evidence_strength"]
    if normalized["primary_event_label"] == UNCERTAIN_ACTIVITY and normalized["incident_category"] == NORMAL_ACTIVITY:
        normalized["primary_event_label"] = NORMAL_ACTIVITY

    for field_name, description_field in [
        ("weapon_visible", "weapon_description"),
        ("person_grabbing_or_restraining", "grabbing_description"),
        ("person_threatened_or_controlled", "threat_description"),
        ("taking_items_visible", "item_taking_description"),
        ("display_or_counter_interaction", "display_interaction_description"),
    ]:
        if normalized[field_name] == "yes" and not normalized[description_field]:
            normalized[description_field] = baseline.get(description_field, "")

    normalized["visible_evidence"] = _dedupe_preserve_order(
        normalized["visible_evidence"] + _keyword_evidence_from_text(merged_text),
        limit=8,
    )
    if baseline.get("scene_type") in {"shop", "indoor", "office", "warehouse"} and not any(
        class_name in {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "scooter", "auto rickshaw"}
        for class_name in yolo_classes
    ):
        normalized["visible_evidence"] = [
            evidence
            for evidence in normalized["visible_evidence"]
            if "traffic" not in evidence.lower() and "vehicle" not in evidence.lower()
        ]

    if normalized["weapon_visible"] == "yes":
        normalized["primary_event_label"] = "possible_weapon_visible" if normalized["primary_event_label"] == NORMAL_ACTIVITY else normalized["primary_event_label"]
        normalized["incident_category"] = PRIORITY_SUSPICIOUS_EVENT
        normalized["risk_level"] = "high"
        normalized["evidence_strength"] = "strong"
        normalized["incident_score"] = max(normalized["incident_score"], 0.85)
        normalized["suspicious_activity"] = "yes"
        if not normalized["suspicious_explanation"]:
            normalized["suspicious_explanation"] = "A weapon-like object or threatening object posture may be visible."

    if normalized["person_grabbing_or_restraining"] == "yes":
        if normalized["primary_event_label"] in {NORMAL_ACTIVITY, UNCERTAIN_ACTIVITY}:
            normalized["primary_event_label"] = "possible_assault_or_grabbing"
        normalized["incident_category"] = PRIORITY_SUSPICIOUS_EVENT
        normalized["risk_level"] = "high"
        normalized["evidence_strength"] = "strong"
        normalized["incident_score"] = max(normalized["incident_score"], 0.75)
        normalized["suspicious_activity"] = "yes"

    if normalized["person_threatened_or_controlled"] == "yes":
        if normalized["primary_event_label"] in {NORMAL_ACTIVITY, UNCERTAIN_ACTIVITY}:
            normalized["primary_event_label"] = "possible_employee_threat_or_restraint"
        normalized["incident_category"] = PRIORITY_SUSPICIOUS_EVENT
        normalized["risk_level"] = "high"
        normalized["evidence_strength"] = "strong"
        normalized["incident_score"] = max(normalized["incident_score"], 0.78)
        normalized["suspicious_activity"] = "yes"

    if normalized["taking_items_visible"] == "yes":
        if normalized["primary_event_label"] in {NORMAL_ACTIVITY, UNCERTAIN_ACTIVITY}:
            normalized["primary_event_label"] = "possible_robbery" if incident_focus == "robbery" else "possible_theft_from_display"
        normalized["incident_category"] = POSSIBLE_REVIEW_CLIP if normalized["incident_category"] != PRIORITY_SUSPICIOUS_EVENT else PRIORITY_SUSPICIOUS_EVENT
        normalized["risk_level"] = "medium" if normalized["risk_level"] == "low" else normalized["risk_level"]
        normalized["evidence_strength"] = "medium" if _evidence_strength_rank(normalized["evidence_strength"]) < 2 else normalized["evidence_strength"]
        normalized["incident_score"] = max(normalized["incident_score"], 0.7)
        normalized["suspicious_activity"] = "yes"

    if normalized["display_or_counter_interaction"] == "yes":
        if normalized["primary_event_label"] in {NORMAL_ACTIVITY, UNCERTAIN_ACTIVITY}:
            normalized["primary_event_label"] = "possible_theft_from_display"
        if normalized["incident_category"] == NORMAL_ACTIVITY:
            normalized["incident_category"] = POSSIBLE_REVIEW_CLIP
        normalized["incident_score"] = max(normalized["incident_score"], 0.5)
        normalized["suspicious_activity"] = "yes" if normalized["taking_items_visible"] == "yes" else normalized["suspicious_activity"]
        normalized["evidence_strength"] = "weak" if normalized["evidence_strength"] == "none" else normalized["evidence_strength"]

    if incident_focus == "robbery":
        if len(_extract_yolo_classes(item)) >= 3 and normalized["display_or_counter_interaction"] == "yes":
            normalized["secondary_event_labels"] = _dedupe_preserve_order(
                normalized["secondary_event_labels"] + ["possible_group_robbery"],
                limit=5,
            )
        if normalized["display_or_counter_interaction"] == "yes" and normalized["people_involved_estimate"] in {"2", "3", "4_plus"}:
            normalized["secondary_event_labels"] = _dedupe_preserve_order(
                normalized["secondary_event_labels"] + ["possible_group_robbery"],
                limit=5,
            )
        if normalized["weapon_visible"] == "yes" and normalized["taking_items_visible"] == "yes":
            normalized["primary_event_label"] = "possible_robbery"
            normalized["secondary_event_labels"] = _dedupe_preserve_order(
                normalized["secondary_event_labels"] + ["possible_weapon_visible", "possible_theft_from_display"],
                limit=5,
            )
        if (
            normalized["display_or_counter_interaction"] == "yes"
            and (
                normalized["person_grabbing_or_restraining"] == "yes"
                or normalized["person_threatened_or_controlled"] == "yes"
                or normalized["weapon_visible"] == "yes"
                or normalized["taking_items_visible"] == "yes"
            )
        ):
            normalized["primary_event_label"] = "possible_robbery"
            normalized["secondary_event_labels"] = _dedupe_preserve_order(
                normalized["secondary_event_labels"] + ["possible_theft_from_display"],
                limit=5,
            )

    if normalized["primary_event_label"] == "possible_collision":
        normalized["incident_category"] = POSSIBLE_REVIEW_CLIP if normalized["incident_category"] == NORMAL_ACTIVITY else normalized["incident_category"]
        normalized["incident_score"] = max(normalized["incident_score"], 0.55)
    if normalized["primary_event_label"] == "possible_fall":
        normalized["incident_category"] = POSSIBLE_REVIEW_CLIP if normalized["incident_category"] == NORMAL_ACTIVITY else normalized["incident_category"]
        normalized["incident_score"] = max(normalized["incident_score"], 0.58)
    if normalized["primary_event_label"] == "possible_intrusion":
        normalized["incident_category"] = POSSIBLE_REVIEW_CLIP if normalized["incident_category"] == NORMAL_ACTIVITY else normalized["incident_category"]
        normalized["incident_score"] = max(normalized["incident_score"], 0.58)

    concrete_evidence = _contains_concrete_evidence(normalized, merged_text)
    weak_only = _is_weak_only_evidence(merged_text)
    display_only = (
        normalized["display_or_counter_interaction"] == "yes"
        and normalized["taking_items_visible"] != "yes"
        and normalized["weapon_visible"] != "yes"
        and normalized["person_grabbing_or_restraining"] != "yes"
        and normalized["person_threatened_or_controlled"] != "yes"
        and not _contains_any(merged_text, {"reach", "reaching", "take", "taking", "collect", "conceal", "hide", "handling"})
    )

    if normalized["incident_category"] in {POSSIBLE_REVIEW_CLIP, PRIORITY_SUSPICIOUS_EVENT} and not concrete_evidence:
        if weak_only:
            normalized.update(
                {
                    "incident_category": NORMAL_ACTIVITY,
                    "primary_event_label": NORMAL_ACTIVITY,
                    "risk_level": "low",
                    "evidence_strength": "none",
                    "incident_score": min(normalized["incident_score"], 0.2),
                    "suspicious_activity": "no",
                    "suspicious_explanation": "",
                    "review_reason": "This clip appears to show routine activity.",
                }
            )
        elif display_only:
            normalized.update(
                {
                    "incident_category": UNCERTAIN_ACTIVITY,
                    "primary_event_label": UNCERTAIN_ACTIVITY,
                    "risk_level": "low",
                    "evidence_strength": "weak",
                    "incident_score": min(normalized["incident_score"], 0.35),
                    "suspicious_activity": "unclear",
                    "review_reason": "Display or counter proximity is visible, but clear suspicious hand action is not established.",
                }
            )

    if normalized["incident_category"] == NORMAL_ACTIVITY:
        normalized["incident_score"] = min(max(normalized["incident_score"], 0.05), 0.2)
        normalized["primary_event_label"] = NORMAL_ACTIVITY
        normalized["risk_level"] = "low"
        normalized["suspicious_activity"] = "no"
        normalized["evidence_strength"] = "none" if weak_only else normalized["evidence_strength"]
    elif normalized["incident_category"] == UNCERTAIN_ACTIVITY:
        normalized["incident_score"] = min(max(normalized["incident_score"], 0.2), 0.35 if display_only else 0.49)
        normalized["primary_event_label"] = UNCERTAIN_ACTIVITY if normalized["primary_event_label"] == NORMAL_ACTIVITY else normalized["primary_event_label"]
        normalized["suspicious_activity"] = "unclear"
    elif normalized["incident_category"] == POSSIBLE_REVIEW_CLIP:
        normalized["incident_score"] = max(normalized["incident_score"], 0.5)
        normalized["risk_level"] = "medium" if normalized["risk_level"] in {"low", "unknown"} else normalized["risk_level"]
        normalized["suspicious_activity"] = "yes"
    elif normalized["incident_category"] == PRIORITY_SUSPICIOUS_EVENT:
        normalized["incident_score"] = max(normalized["incident_score"], 0.75)
        normalized["risk_level"] = "high" if normalized["risk_level"] in {"low", "medium", "unknown"} else normalized["risk_level"]
        normalized["suspicious_activity"] = "yes"

    if normalized["primary_event_label"] == "possible_theft_from_display" and incident_focus == "robbery" and (
        normalized["person_grabbing_or_restraining"] == "yes"
        or normalized["person_threatened_or_controlled"] == "yes"
        or normalized["weapon_visible"] == "yes"
    ):
        normalized["secondary_event_labels"] = _dedupe_preserve_order(
            normalized["secondary_event_labels"] + ["possible_robbery"],
            limit=5,
        )
    if normalized["primary_event_label"] == "possible_weapon_visible" and normalized["taking_items_visible"] == "yes":
        normalized["secondary_event_labels"] = _dedupe_preserve_order(
            normalized["secondary_event_labels"] + ["possible_robbery", "possible_theft_from_display"],
            limit=5,
        )

    if not normalized["normal_explanation"]:
        normalized["normal_explanation"] = baseline["normal_explanation"]
    if not normalized["review_reason"]:
        normalized["review_reason"] = baseline["review_reason"]
    if not normalized["description"]:
        normalized["description"] = baseline["description"]

    normalized["event_label"] = normalized["primary_event_label"]
    return normalized


def _parse_incident_output(raw_text: str, item: dict[str, Any], incident_focus: str) -> tuple[bool, dict[str, Any], str | None, bool]:
    cleaned = _extract_json_text(raw_text)
    parse_errors: list[str] = []

    if cleaned:
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return True, _normalize_incident_output(parsed, item, incident_focus), None, False
            parse_errors.append("Parsed JSON was not an object.")
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))

        repaired = _repair_json_candidate(cleaned)
        if repaired:
            try:
                parsed = json.loads(repaired)
                if isinstance(parsed, dict):
                    return True, _normalize_incident_output(parsed, item, incident_focus), None, False
                parse_errors.append("Repaired JSON was not an object.")
            except json.JSONDecodeError as exc:
                parse_errors.append(str(exc))
    else:
        parse_errors.append("No JSON object found in VLM output.")

    return False, _baseline_incident_recheck(item, incident_focus), " | ".join(parse_errors), True


def _load_existing_raw_outputs(output_path: Path) -> dict[str, str]:
    if not output_path.exists():
        return {}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return {}
    raw_by_clip_id: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("source_clip_id", "")).strip()
        raw_text = str(item.get("raw_vlm_output", "") or "")
        if clip_id and raw_text.strip():
            raw_by_clip_id[clip_id] = raw_text
    return raw_by_clip_id


def _should_recheck_item(item: dict[str, Any], recheck_all: bool, incident_focus: str) -> bool:
    if recheck_all:
        return True
    _, text = _text_sources_for_item(item)
    yolo_classes = _extract_yolo_classes(item)
    scene_type = _clean_text(item.get("parsed_json", {}).get("scene_type") if isinstance(item.get("parsed_json"), dict) else "").lower()
    if incident_focus == "robbery":
        return (
            scene_type in {"shop", "indoor", "office", "warehouse", "unknown"}
            or _contains_any(text, DISPLAY_INTERACTION_TERMS | TAKING_TERMS | WEAPON_TERMS | GRABBING_TERMS | THREAT_TERMS | ROBBERY_TERMS)
            or "person" in yolo_classes
        )
    if incident_focus == "traffic":
        return scene_type in {"road", "street", "outdoor", "unknown"} or _contains_any(text, TRAFFIC_TERMS)
    if incident_focus == "fall":
        return _contains_any(text, FALL_TERMS) or "person" in yolo_classes
    if incident_focus == "intrusion":
        return _contains_any(text, INTRUSION_TERMS)
    return (
        scene_type in {"shop", "road", "street", "outdoor", "indoor", "unknown"}
        or _contains_any(text, DISPLAY_INTERACTION_TERMS | TAKING_TERMS | WEAPON_TERMS | GRABBING_TERMS | THREAT_TERMS | FALL_TERMS | TRAFFIC_TERMS | INTRUSION_TERMS)
        or "person" in yolo_classes
    )


def _top_primary_event_labels(items: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for item in items:
        incident = item.get("incident_recheck", {})
        if not isinstance(incident, dict):
            continue
        label = _clean_text(incident.get("primary_event_label") or incident.get("event_label")).lower()
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return [
        {"event_label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    ]


def run_incident_recheck_reasoning(run_dir: Path) -> dict[str, Any]:
    print("[tender-demo] Starting Step 16B: incident recheck reasoning")

    step15_payload = _load_required_json(run_dir / "15_topk_vlm_inputs.json")
    step16_payload = _load_required_json(run_dir / "16_topk_vlm_outputs.json")
    yolo_payload = _load_optional_json(run_dir / "11_yolo_object_scores.json")
    ranked_payload = _load_optional_json(run_dir / "13_ranked_clips.json")
    selected_payload = _load_optional_json(run_dir / "14_selected_top_clips.json")

    step15_items = step15_payload.get("items", []) if isinstance(step15_payload, dict) else step15_payload
    step16_items = step16_payload.get("items", []) if isinstance(step16_payload, dict) else step16_payload
    if not isinstance(step15_items, list) or not isinstance(step16_items, list):
        raise ValueError("Step 15/16 manifests must contain item lists.")

    step15_by_clip_id = {
        str(item.get("source_clip_id", "")).strip(): item
        for item in step15_items
        if isinstance(item, dict) and str(item.get("source_clip_id", "")).strip()
    }

    output_path = run_dir / INCIDENT_RECHECK_OUTPUT_NAME
    existing_raw_outputs = _load_existing_raw_outputs(output_path)
    recheck_all = _safe_bool_env("TENDER_DEMO_INCIDENT_RECHECK_ALL_TOPK", DEFAULT_INCIDENT_RECHECK_ALL_TOPK)
    incident_focus = _read_incident_focus()
    requested_backend = os.environ.get("TENDER_DEMO_VLM_BACKEND", "qwen").strip().lower() or "qwen"

    vlm = None
    vlm_health: dict[str, Any] = {}
    resolved_backend = requested_backend
    adapter_error: str | None = None
    try:
        create_tender_demo_vlm = _load_tender_demo_vlm_factory()
        vlm = create_tender_demo_vlm()
        if hasattr(vlm, "health_check"):
            try:
                vlm_health = vlm.health_check()
            except Exception:
                vlm_health = {}
        resolved_backend = str(vlm_health.get("backend", requested_backend)).strip() or requested_backend
    except Exception as exc:
        adapter_error = str(exc)
        if existing_raw_outputs:
            print("[tender-demo] Step 16B adapter unavailable. Reusing existing incident recheck outputs.")
        else:
            print(f"[tender-demo] Step 16B adapter unavailable. Using heuristic incident recheck fallback. Reason: {exc}")

    results: list[dict[str, Any]] = []
    valid_requests: list[tuple[dict[str, Any], Path, dict[str, Any]]] = []

    for item in step16_items:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("source_clip_id", "")).strip()
        if not clip_id:
            continue
        step15_item = step15_by_clip_id.get(clip_id, {})
        merged_item = {**step15_item, **item}
        baseline_recheck = _baseline_incident_recheck(merged_item, incident_focus)
        record = {
            "source_clip_id": clip_id,
            "start_time": merged_item.get("start_time"),
            "end_time": merged_item.get("end_time"),
            "strip_path": merged_item.get("strip_path"),
            "step16_summary": {
                "caption": _clean_text(merged_item.get("parsed_json", {}).get("caption") if isinstance(merged_item.get("parsed_json"), dict) else ""),
                "description": _clean_text(merged_item.get("parsed_json", {}).get("description") if isinstance(merged_item.get("parsed_json"), dict) else "")
                or _clean_text(merged_item.get("raw_vlm_output")),
                "event_label": _clean_text(merged_item.get("parsed_json", {}).get("event_label") if isinstance(merged_item.get("parsed_json"), dict) else "") or "unknown",
            },
            "incident_recheck": baseline_recheck,
            "recheck_performed": False,
            "parse_success": False,
            "fallback_used": True,
            "parse_error": None,
            "raw_vlm_output": "",
            "prompt": "",
        }
        results.append(record)

        if not _should_recheck_item(merged_item, recheck_all, incident_focus):
            record["parse_error"] = "Skipped incident recheck due to fast-mode filter."
            continue

        strip_path_value = merged_item.get("strip_path")
        strip_path = Path(str(strip_path_value)) if strip_path_value else None
        record["prompt"] = _build_incident_prompt(merged_item, incident_focus)
        record["recheck_performed"] = True

        if strip_path is None or not strip_path.exists():
            record["parse_error"] = f"Missing strip image path: {strip_path_value}"
            continue

        if vlm is None:
            existing_raw_text = existing_raw_outputs.get(clip_id, "")
            if existing_raw_text.strip():
                parse_success, parsed_json, parse_error, fallback_used = _parse_incident_output(existing_raw_text, merged_item, incident_focus)
                record["raw_vlm_output"] = existing_raw_text
                record["incident_recheck"] = parsed_json
                record["parse_success"] = parse_success
                record["fallback_used"] = fallback_used
                record["parse_error"] = parse_error or adapter_error
            else:
                record["parse_error"] = adapter_error or "VLM adapter unavailable; heuristic incident recheck used."
            continue

        valid_requests.append((record, strip_path, merged_item))

    if valid_requests and vlm is not None:
        image_paths = [strip_path for _, strip_path, _ in valid_requests]
        prompts = [record["prompt"] for record, _, _ in valid_requests]
        try:
            raw_outputs = vlm.generate_batch(image_paths=image_paths, prompts=prompts)
        except Exception as exc:
            raw_outputs = []
            error_message = f"Incident recheck generation failed: {exc}"
            for record, _, _ in valid_requests:
                record["parse_error"] = error_message
        else:
            for (record, _, merged_reference), raw_output in zip(valid_requests, raw_outputs):
                raw_text = raw_output if isinstance(raw_output, str) else str(raw_output)
                parse_success, parsed_json, parse_error, fallback_used = _parse_incident_output(raw_text, merged_reference, incident_focus)
                record["raw_vlm_output"] = raw_text
                record["incident_recheck"] = parsed_json
                record["parse_success"] = parse_success
                record["fallback_used"] = fallback_used
                record["parse_error"] = parse_error

            if len(raw_outputs) < len(valid_requests):
                for record, _, merged_reference in valid_requests[len(raw_outputs) :]:
                    record["incident_recheck"] = _baseline_incident_recheck(merged_reference, incident_focus)
                    record["parse_error"] = "Adapter returned fewer outputs than requested."

    rechecked_items = [item for item in results if item.get("recheck_performed") is True]
    successful_outputs = sum(1 for item in rechecked_items if item.get("parse_success") is True)
    failed_outputs = sum(1 for item in rechecked_items if item.get("parse_success") is not True)
    priority_items = [
        item for item in results if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("incident_category") == PRIORITY_SUSPICIOUS_EVENT
    ]
    review_items = [
        item for item in results if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("incident_category") == POSSIBLE_REVIEW_CLIP
    ]
    normal_items = [
        item for item in results if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("incident_category") == NORMAL_ACTIVITY
    ]
    uncertain_items = [
        item for item in results if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("incident_category") == UNCERTAIN_ACTIVITY
    ]

    payload = {
        "vlm_backend": resolved_backend,
        "requested_vlm_backend": requested_backend,
        "vlm_health": vlm_health,
        "total_inputs": len(results),
        "rechecked_clips": len(rechecked_items),
        "successful_outputs": successful_outputs,
        "failed_outputs": failed_outputs,
        "settings": {
            "incident_recheck_enabled": True,
            "incident_recheck_all_topk": recheck_all,
            "incident_focus": incident_focus,
            "vlm_backend": resolved_backend,
        },
        "inputs_summary": {
            "step15_items": len(step15_items),
            "step16_items": len(step16_items),
            "yolo_available": isinstance(yolo_payload, list),
            "ranked_clips_available": isinstance(ranked_payload, list),
            "selected_topk_available": isinstance(selected_payload, dict) or isinstance(selected_payload, list),
        },
        "items": results,
    }
    report = {
        "priority_suspicious_events": len(priority_items),
        "possible_review_clips": len(review_items),
        "normal_activity": len(normal_items),
        "uncertain_activity": len(uncertain_items),
        "failed_outputs": failed_outputs,
        "rechecked_clips": len(rechecked_items),
        "top_primary_event_labels": _top_primary_event_labels(results),
        "top_event_labels": _top_primary_event_labels(results),
        "weapon_visible_clips": sum(
            1
            for item in results
            if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("weapon_visible") == "yes"
        ),
        "grabbing_or_restraint_clips": sum(
            1
            for item in results
            if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("person_grabbing_or_restraining") == "yes"
        ),
        "threat_or_control_clips": sum(
            1
            for item in results
            if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("person_threatened_or_controlled") == "yes"
        ),
        "taking_items_clips": sum(
            1
            for item in results
            if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("taking_items_visible") == "yes"
        ),
        "display_interaction_clips": sum(
            1
            for item in results
            if isinstance(item.get("incident_recheck"), dict) and item["incident_recheck"].get("display_or_counter_interaction") == "yes"
        ),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = run_dir / INCIDENT_RECHECK_REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[tender-demo] Incident recheck VLM backend: {resolved_backend}")
    print(f"[tender-demo] Rechecked clips: {len(rechecked_items)}")
    print(f"[tender-demo] Incident recheck priority clips: {len(priority_items)}")
    print(f"[tender-demo] Incident recheck review clips: {len(review_items)}")
    print(f"[tender-demo] Incident recheck normal clips: {len(normal_items)}")
    print(f"[tender-demo] Incident recheck uncertain clips: {len(uncertain_items)}")
    print(f"[tender-demo] Incident recheck failed outputs: {failed_outputs}")
    print(f"[tender-demo] Output path: {output_path}")
    print(f"[tender-demo] Report path: {report_path}")

    return payload
