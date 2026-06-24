"""Activity Recovery Layer.

Post-processing service that recovers semantic activity information from frame
metadata when the primary ``activities`` field returned by the VLM is empty.

The VLM consistently places activity/state information in three locations:
  - ``objects[].attributes`` (e.g. "stationary", "parked position", "walking away")
  - ``caption``              (e.g. "a silver sedan parked curbside...")
  - ``keywords``             (e.g. "vehicle parked", "pedestrian crossing")

...but frequently returns ``"activities": []``.

This layer mines those three sources in priority order and back-fills the
``activities`` list before the metadata is validated by Pydantic and written
to disk.  The recovery source is recorded in ``activity_recovery_source`` so
callers can distinguish VLM-native activities from recovered ones.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class ActivityRecoveryService:
    """Recovers activity labels from VLM frame metadata when ``activities`` is empty.

    Recovery priority order:
    1. Object ``attributes``  — highest confidence (model already classified state)
    2. ``caption`` patterns   — medium confidence (model described activity in prose)
    3. ``keywords``           — lower confidence (model tagged relevant concepts)
    """

    # ------------------------------------------------------------------ #
    # Rule tables                                                          #
    # Each table is a list of (trigger_pattern, activity_label) tuples.   #
    # Patterns are matched as substrings (case-insensitive).              #
    # First matching rule per source string wins.                         #
    # ------------------------------------------------------------------ #

    # Attribute text → activity label
    # Ordered most-specific → most-general so early rules don't shadow later ones.
    ATTRIBUTE_RULES: List[Tuple[str, str]] = [
        # --- Vehicle motion / parking state ---
        ("parked position",         "vehicle parked"),
        ("parked",                  "vehicle parked"),
        ("stationary",              "vehicle stationary"),
        ("taillights lit",          "vehicle stationary"),
        ("brake lights",            "vehicle stationary"),
        ("stopped",                 "vehicle stopped"),
        ("reversing",               "vehicle reversing"),
        ("in motion",               "vehicle moving"),
        ("moving",                  "vehicle moving"),
        ("entering",                "vehicle entering"),
        ("exiting",                 "vehicle exiting"),
        ("arriving",                "vehicle arriving"),
        ("departing",               "vehicle departing"),
        # --- Vehicle occupancy ---
        ("person seated on it",     "vehicle occupied"),
        ("driver inside",           "vehicle occupied"),
        ("occupied",                "vehicle occupied"),
        # --- Person motion ---
        ("walking away",            "walking"),
        ("walking towards",         "walking"),
        ("walking",                 "walking"),
        ("running",                 "running"),
        ("crossing road",           "crossing road"),
        ("crossing street",         "crossing road"),
        ("crosswalk",               "crossing road"),
        ("standing",                "standing"),
        ("sitting",                 "sitting"),
        ("loitering",               "loitering"),
        ("placing",                 "placing object"),
        ("dropping",                "dropping object"),
        ("putting down",            "dropping object"),
        ("picking up",              "picking up object"),
        ("picked up",               "picking up object"),
        ("carrying",                "carrying object"),
        ("holding",                 "carrying object"),
        ("holding object",          "carrying object"),
    ]

    # Caption regex pattern → activity label
    # Ordered most-specific → most-general.
    CAPTION_RULES: List[Tuple[str, str]] = [
        # Vehicle parked / stationary
        (r"\bpark(?:ed|ing|s)?\b",          "vehicle parked"),
        (r"\bstation(?:ary|ed)\b",          "vehicle stationary"),
        (r"\bstopp(?:ed|ing)\b",            "vehicle stopped"),
        # Vehicle motion
        (r"\bdriv(?:ing|es|en|e|er)\b",     "driving"),
        (r"\btravell?(?:ing|ed|s)\b",       "moving"),
        (r"\benter(?:ing|s|ed)\b",          "entering"),
        (r"\bexit(?:ing|s|ed)\b",           "exiting"),
        (r"\barr(?:iving|ived|ives)\b",     "arriving"),
        (r"\bdepart(?:ing|s|ed)\b",         "departing"),
        (r"\brevers(?:ing|ed|es)\b",        "reversing"),
        (r"\bmoving?\b",                    "moving"),
        # Person motion
        (r"\bcross(?:ing|es|ed)?\b.*\b(?:across|crosswalk|intersection|road|street)\b", "crossing road"),
        (r"\bwalk(?:ing|s|ed)?\b.*\bacross\b.*\b(?:crosswalk|intersection|road|street)\b", "crossing road"),
        (r"\bwalk(?:ing|s|ed|er)\b",        "walking"),
        (r"\brun(?:ning|s|ner)\b",          "running"),
        (r"\bstand(?:ing|s)\b",             "standing"),
        (r"\bsit(?:ting|s|sat|ted)\b",      "sitting"),
        (r"\bapproach(?:ing|es|ed)\b",      "approaching"),
        (r"\bloiter(?:ing)?\b",             "loitering"),
        (r"\bplac(?:ing|es|ed)\b.*\b(?:object|bag|box|package|item)\b", "placing object"),
        (r"\b(?:drop(?:ping|s|ped)?|put(?:ting)? down|left)\b.*\b(?:object|bag|box|package|item)\b", "dropping object"),
        (r"\bpick(?:ing)? (?:something|it|an? object|a bag|a box|a package|an item)?\s*up\b", "picking up object"),
        (r"\bpicked (?:something|it|an? object|a bag|a box|a package|an item)?\s*up\b", "picking up object"),
        (r"\bholding (?:an? )?(?:object|bag|box|package|item)\b", "carrying object"),
        (r"\bcarrying?\b",                  "carrying object"),
        (r"\bholds?\b",                     "carrying object"),
        (r"\bseated\b",                     "seated"),
    ]

    # Keyword text → activity label
    KEYWORD_RULES: List[Tuple[str, str]] = [
        ("vehicle parked",          "vehicle parked"),
        ("car parked",              "vehicle parked"),
        ("vehicle stopped",         "vehicle stopped"),
        ("pedestrian crossing",     "crossing road"),
        ("vehicle crossing",        "crossing road"),
        ("motorbike rider",         "riding motorcycle"),
        ("construction activity",   "construction activity"),
        ("pedestrian activity",     "walking"),
        ("person walking",          "walking"),
        ("person standing",         "standing"),
        ("person running",          "running"),
        ("placing object",          "placing object"),
        ("dropping object",         "dropping object"),
        ("picking up object",       "picking up object"),
        ("traveling",               "moving"),
    ]

    ACTIVITY_ALIASES: Dict[str, str] = {
        "crossing street": "crossing road",
        "crossing road": "crossing road",
        "pedestrian crossing": "crossing road",
        "walking across road": "crossing road",
        "walking across street": "crossing road",
        "crossing": "crossing road",
        "walking_with": "walking",
        "walking with": "walking",
        "walk": "walking",
        "walks": "walking",
        "walking": "walking",
        "run": "running",
        "runs": "running",
        "running": "running",
        "drive": "driving",
        "drives": "driving",
        "driving": "driving",
        "parked": "vehicle parked",
        "parking": "vehicle parked",
        "vehicle parked": "vehicle parked",
        "stationary": "vehicle stationary",
        "vehicle stationary": "vehicle stationary",
        "stopped": "vehicle stopped",
        "vehicle stopped": "vehicle stopped",
        "moving": "moving",
        "vehicle moving": "vehicle moving",
        "enter": "entering",
        "entering": "entering",
        "exit": "exiting",
        "exiting": "exiting",
        "arrive": "arriving",
        "arriving": "arriving",
        "depart": "departing",
        "departing": "departing",
        "reverse": "reversing",
        "reversing": "reversing",
        "stand": "standing",
        "standing": "standing",
        "sit": "sitting",
        "sitting": "sitting",
        "seated": "seated",
        "carrying": "carrying object",
        "holding": "carrying object",
        "carrying object": "carrying object",
        "placing": "placing object",
        "placing object": "placing object",
        "dropping": "dropping object",
        "dropping object": "dropping object",
        "object drop": "dropping object",
        "picking up": "picking up object",
        "picking up object": "picking up object",
        "picked up object": "picking up object",
    }

    # Captions with these exact values carry no useful information
    _EMPTY_CAPTIONS: frozenset = frozenset({
        "",
        "no description available.",
        "no description available",
        "n/a",
    })

    # ------------------------------------------------------------------ #
    # Recovery methods                                                     #
    # ------------------------------------------------------------------ #

    ROAD_CONTEXT_TERMS = (
        "road", "street", "crosswalk", "intersection", "sidewalk",
        "lane", "traffic", "parking", "parking_area", "curb",
    )
    NON_ROAD_CONTEXT_TERMS = (
        "office", "corridor", "hallway", "indoor", "desk", "workspace",
        "meeting room", "shop", "warehouse",
    )

    @classmethod
    def normalize_activity_label(cls, activity: Any) -> str:
        """Normalize model/recovery activity variants to the pipeline vocabulary."""
        label = str(activity or "").strip().lower()
        if not label or label == "none":
            return ""
        label = re.sub(r"[_\-]+", " ", label)
        label = re.sub(r"\s+", " ", label).strip()
        return cls.ACTIVITY_ALIASES.get(label, label)

    @classmethod
    def normalize_activities(cls, activities: List[Any]) -> List[str]:
        """Normalize and deduplicate an activity list while preserving order."""
        normalized: List[str] = []
        for activity in activities or []:
            label = cls.normalize_activity_label(activity)
            if label and label not in normalized:
                normalized.append(label)
        return normalized

    @classmethod
    def _has_road_context(cls, frame_data: Dict[str, Any]) -> bool:
        """Return true when the frame context is consistent with road crossing."""
        parts: List[str] = []
        parts.append(str(frame_data.get("scene_type", "") or ""))
        parts.append(str(frame_data.get("scene_description", "") or ""))
        parts.append(str(frame_data.get("caption", "") or ""))
        parts.extend(str(k) for k in (frame_data.get("keywords", []) or []))

        for loc in frame_data.get("location_context", []) or []:
            if isinstance(loc, dict):
                parts.append(str(loc.get("location", "") or ""))

        text = " ".join(parts).lower()
        return any(term in text for term in cls.ROAD_CONTEXT_TERMS)

    @classmethod
    def _has_non_road_context(cls, frame_data: Dict[str, Any]) -> bool:
        parts: List[str] = []
        parts.append(str(frame_data.get("scene_type", "") or ""))
        parts.append(str(frame_data.get("scene_description", "") or ""))
        parts.append(str(frame_data.get("caption", "") or ""))
        parts.extend(str(k) for k in (frame_data.get("keywords", []) or []))
        text = " ".join(parts).lower()
        return any(term in text for term in cls.NON_ROAD_CONTEXT_TERMS)

    @classmethod
    def _contextualize_activities(cls, activities: List[str], frame_data: Dict[str, Any]) -> List[str]:
        """Map ambiguous crossing labels using scene context to avoid office hallucinations."""
        if not activities:
            return []

        has_road_context = cls._has_road_context(frame_data)
        has_non_road_context = cls._has_non_road_context(frame_data)
        normalized: List[str] = []
        for activity in activities:
            label = cls.normalize_activity_label(activity)
            if label == "crossing road" and not has_road_context and has_non_road_context:
                label = "walking"
            if label and label not in normalized:
                normalized.append(label)
        return normalized

    @classmethod
    def recover_from_attributes(cls, objects: List[Dict[str, Any]]) -> List[str]:
        """Mine activity labels from each object's ``attributes`` list.

        Args:
            objects: List of object dicts from frame metadata.

        Returns:
            Deduplicated list of recovered activity labels.
        """
        recovered: List[str] = []
        for obj in objects:
            attrs = obj.get("attributes", [])
            if not isinstance(attrs, list):
                continue
            for attr in attrs:
                attr_lower = str(attr).lower()
                for trigger, activity in cls.ATTRIBUTE_RULES:
                    normalized = cls.normalize_activity_label(activity)
                    if trigger in attr_lower and normalized not in recovered:
                        recovered.append(normalized)
                        break  # one activity per attribute string
        return recovered

    @classmethod
    def recover_from_caption(cls, caption: str) -> List[str]:
        """Mine activity labels from the frame caption using regex patterns.

        Args:
            caption: Caption string from frame metadata.

        Returns:
            Deduplicated list of recovered activity labels.
        """
        recovered: List[str] = []
        caption_stripped = caption.strip().lower()
        if caption_stripped in cls._EMPTY_CAPTIONS:
            return recovered
        for pattern, activity in cls.CAPTION_RULES:
            normalized = cls.normalize_activity_label(activity)
            if re.search(pattern, caption_stripped) and normalized not in recovered:
                recovered.append(normalized)
        return recovered

    @classmethod
    def recover_from_keywords(cls, keywords: List[str]) -> List[str]:
        """Mine activity labels from the keywords list.

        Args:
            keywords: Keywords list from frame metadata.

        Returns:
            Deduplicated list of recovered activity labels.
        """
        recovered: List[str] = []
        if not keywords:
            return recovered
        keywords_text = " ".join(keywords).lower()
        for trigger, activity in cls.KEYWORD_RULES:
            normalized = cls.normalize_activity_label(activity)
            if trigger in keywords_text and normalized not in recovered:
                recovered.append(normalized)
        return recovered

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def recover(
        cls,
        frame_data: Dict[str, Any],
    ) -> Tuple[List[str], str]:
        """Attempt to recover activities from frame metadata when ``activities`` is empty.

        Args:
            frame_data: Dictionary containing frame metadata fields.

        Returns:
            Tuple ``(activities, source)`` where ``source`` is one of:
              - ``"original"``   — VLM already returned non-empty activities (no recovery needed)
              - ``"attributes"`` — recovered from object attribute descriptions
              - ``"caption"``    — recovered from caption text patterns
              - ``"keywords"``   — recovered from keywords list
              - ``"none"``       — no activities recoverable from any source
        """
        frame_id = frame_data.get("frame_id", "unknown")
        existing = cls.normalize_activities(frame_data.get("activities", []))
        objects  = frame_data.get("objects", [])
        caption  = str(frame_data.get("caption", ""))
        keywords = frame_data.get("keywords", []) or []

        if existing:
            enriched = list(existing)
            for activity in (
                cls.recover_from_attributes(objects)
                + cls.recover_from_caption(caption)
                + cls.recover_from_keywords(keywords)
            ):
                if activity == "walking" and "crossing road" in enriched:
                    continue
                if activity not in enriched:
                    enriched.append(activity)
            return cls._contextualize_activities(enriched, frame_data), "original"

        # Priority 1 — object attributes (highest confidence)
        attr_activities = cls.recover_from_attributes(objects)
        if attr_activities:
            logger.debug(
                f"[ActivityRecovery] attributes → {attr_activities} | frame={frame_id}"
            )
            return cls._contextualize_activities(attr_activities, frame_data), "attributes"

        # Priority 2 — caption patterns (medium confidence)
        caption_activities = cls.recover_from_caption(caption)
        if caption_activities:
            logger.debug(
                f"[ActivityRecovery] caption → {caption_activities} | frame={frame_id}"
            )
            return cls._contextualize_activities(caption_activities, frame_data), "caption"

        # Priority 3 — keywords (lower confidence)
        keyword_activities = cls.recover_from_keywords(keywords)
        if keyword_activities:
            logger.debug(
                f"[ActivityRecovery] keywords → {keyword_activities} | frame={frame_id}"
            )
            return cls._contextualize_activities(keyword_activities, frame_data), "keywords"

        logger.debug(f"[ActivityRecovery] no recovery possible | frame={frame_id}")
        return [], "none"

    @classmethod
    def apply(cls, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Apply activity recovery to a parsed frame metadata dictionary.

        Mutates and returns ``parsed``.  If activities are recovered:
          - ``parsed["activities"]``               is updated with recovered labels.
          - ``parsed["activity_recovery_source"]`` is set to the recovery source.

        If activities were already non-empty:
          - ``parsed["activities"]`` is left unchanged.
          - ``activity_recovery_source`` is not set (remains ``None`` after Pydantic
            applies its default).

        Args:
            parsed: Normalized frame metadata dictionary (pre-Pydantic).

        Returns:
            Updated frame metadata dictionary.
        """
        activities, source = cls.recover(parsed)
        parsed["activities"] = activities

        if source not in ("original", "none"):
            parsed["activity_recovery_source"] = source
        elif source == "none":
            # Explicitly record that no recovery was possible
            parsed["activity_recovery_source"] = "none"
        # source == "original": leave activity_recovery_source absent (Pydantic default None)

        return parsed
