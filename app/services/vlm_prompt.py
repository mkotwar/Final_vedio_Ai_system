"""Shared VLM prompt contract for frame-level CCTV metadata extraction."""

VLM_FRAME_METADATA_PROMPT = """You analyze CCTV/security imagery.

The image may be either:
- one normal frame, or
- a 3-panel temporal strip labeled PREVIOUS, CURRENT, NEXT.

If a 3-panel strip is shown, produce metadata for the CURRENT panel only. Use
PREVIOUS and NEXT only as temporal context to identify object interactions,
such as a person placing an object, leaving an object behind, picking an object
up, or removing an object.

Return ONLY one valid JSON object. No markdown. No explanation. No comments.

Goal:
Produce objective frame metadata that can be used by event aggregation, timeline
generation, search, and investigation review.

Core rules:
1. Report only visible facts from the current frame/current panel.
2. Do not infer intent, crime, blame, cause, or future/past events.
3. Do not invent people, vehicles, objects, text, relationships, or incidents.
4. If uncertain, use "unknown" for a field or omit the uncertain item.
5. Prefer humans, vehicles, animals, carried/held objects, and active interactions.
6. Ignore background furniture and scenery unless a person/vehicle interacts with it.
7. Every object must have a stable unique id such as person_1, vehicle_1, bag_1.
8. If people_count is greater than 0, objects must include at least one person.
9. If a relationship or event references an actor, that id must exist in objects.
10. Keep descriptions concise, factual, and surveillance-oriented.
11. Do not count the same person shown in PREVIOUS/CURRENT/NEXT panels as multiple current people.
12. If an object is small but visible near a person's hand, feet, counter, desk, or floor, include it as object subtype "bag", "box", "package", "phone", or "other".

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
    "standing|walking|running|sitting|talking|interacting|waiting|entering|exiting|holding|carrying|placing object|dropping object|picking up object|following|approaching|leaving|crossing road|falling|driving|riding|parking|vehicle parked|vehicle stopped|vehicle moving"
  ],
  "relationships": [
    {
      "subject_id": "person_1",
      "target_id": "bag_1",
      "relation": "carrying|holding|placing|dropping|picking_up|talking_to|approaching|following|facing|interacting_with|near|entering|exiting|riding|driving|loading|unloading|walking_with|standing_with"
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
      "event_type": "collision|fall|fire|smoke|intrusion|abandoned_object|object_removed|weapon_visible|medical_emergency|physical_altercation|none",
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
- Use "placing object", "dropping object", or "picking up object" when a person is visibly interacting with a bag, box, package, or similar object.
- Use event_type "abandoned_object" when a bag/box/package is visible unattended after a person leaves it behind.
- Use event_type "object_removed" when a person is visibly picking up or removing a previously unattended bag/box/package.
- In temporal strips, if the object appears in CURRENT/NEXT but was absent in PREVIOUS near the same person, classify the action as "placing object" or "dropping object".
- In temporal strips, if the object is present in PREVIOUS/CURRENT but absent in NEXT after a person bends or reaches toward it, classify the action as "picking up object" or event_type "object_removed".
- Include vehicle color and subtype when visible because they are important for search.
- Keep arrays empty when nothing reliable is visible.

Return ONLY raw JSON."""
