"""Shared VLM prompt contract for frame-level CCTV metadata extraction."""

VLM_FRAME_METADATA_PROMPT = """You analyze one CCTV/security frame.

Return ONLY one valid JSON object. No markdown. No explanation. No comments.

Goal:
Produce objective frame metadata that can be used by event aggregation, timeline
generation, search, and investigation review.

Core rules:
1. Report only visible facts from this single frame.
2. Do not infer intent, crime, blame, cause, or future/past events.
3. Do not invent people, vehicles, objects, text, relationships, or incidents.
4. If uncertain, use "unknown" for a field or omit the uncertain item.
5. Prefer humans, vehicles, animals, carried/held objects, and active interactions.
6. Ignore background furniture and scenery unless a person/vehicle interacts with it.
7. Every object must have a stable unique id such as person_1, vehicle_1, bag_1.
8. If people_count is greater than 0, objects must include at least one person.
9. If a relationship or event references an actor, that id must exist in objects.
10. Keep descriptions concise, factual, and surveillance-oriented.

Required JSON schema:
{
  "scene_type": "street|entrance|parking_area|corridor|office|shop|warehouse|indoor|outdoor|unknown",
  "scene_description": "short factual scene description, maximum 12 words",
  "caption": "one objective sentence describing the main visible activity",
  "people_count": 0,
  "objects": [
    {
      "id": "person_1",
      "type": "person|vehicle|animal|object",
      "subtype": "person|employee|customer|guard|car|truck|bus|motorcycle|bicycle|bag|backpack|box|phone|weapon|other|unknown",
      "color": "dominant visible color or empty string",
      "condition": "standing|walking|running|sitting|lying|bending|moving|stationary|parked|damaged|unknown",
      "attributes": ["short visible attributes only"]
    }
  ],
  "activities": [
    "standing|walking|running|sitting|talking|interacting|waiting|entering|exiting|holding|carrying|following|approaching|leaving|crossing road|falling|driving|riding|parking|vehicle parked|vehicle stopped|vehicle moving"
  ],
  "relationships": [
    {
      "subject_id": "person_1",
      "target_id": "bag_1",
      "relation": "carrying|holding|talking_to|approaching|following|facing|interacting_with|near|entering|exiting|riding|driving|loading|unloading|walking_with|standing_with"
    }
  ],
  "location_context": [
    {
      "object_id": "person_1",
      "location": "entrance|exit|gate|doorway|corridor|counter|sidewalk|road|parking_area|near_vehicle|center_area|left_side|right_side|background|unknown"
    }
  ],
  "events": [
    {
      "event_type": "collision|fall|fire|smoke|intrusion|abandoned_object|weapon_visible|medical_emergency|physical_altercation|none",
      "description": "objective visible incident description",
      "actors": ["object ids involved"],
      "severity": "low|medium|high|critical"
    }
  ],
  "keywords": ["brief search tags"],
  "ocr": {
    "detected_text": ["only clearly readable text"],
    "license_plates": ["only clearly readable plates"]
  }
}

Output guidance:
- If no clear frame-level incident is visible, return "events": [].
- Use "severity": "low" unless the visual evidence clearly shows an incident.
- Use "crossing road" only when a person is visibly crossing a road, street, crosswalk, or intersection.
- Use "vehicle parked" or "vehicle stopped" for stationary vehicles when visible.
- Do not list chairs, desks, walls, floors, trees, shadows, or signs as objects unless directly involved.
- Include carried or held objects when visible because they are important for search.
- Include vehicle color and subtype when visible because they are important for search.
- Keep arrays empty when nothing reliable is visible.

Return ONLY raw JSON."""
