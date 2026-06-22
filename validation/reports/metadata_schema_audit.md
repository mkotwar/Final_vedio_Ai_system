# Metadata Schema Audit

## Source

[frame.py](file:///c:/Mukul%20K/vinfo1/video-search-engine/app/schemas/frame.py)

---

## Object Schema (`ObjectMetadata`)

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `id` | `str` | Yes (empty ok) | `""` | Unique identifier for cross-referencing |
| `type` | `str` | Yes | `"unknown"` | Broad category (e.g. `person`, `furniture`) |
| `subtype` | `str` | Yes (empty ok) | `""` | Specific type (e.g. `customer`, `employee`) |
| `color` | `str` | Optional | `""` | Dominant color |
| `condition` | `str` | Yes | `"normal"` | Physical state |
| `attributes` | `List[str]` | Optional | `[]` | Additional descriptors |

### Ambiguity Notes
- `condition` accepts free text but documentation suggests: `normal/damaged/displaced/moving/stationary/fallen`
- The VLM frequently uses non-standard conditions like `"empty"`, `"unoccupied"`, `"closed"`, `"on"`, `"off"`
- No validation enforces the allowed condition vocabulary

---

## Activity Schema

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| (list item) | `str` | — | — | Must be a plain string from approved vocabulary |

### Approved Vocabulary
`standing`, `walking`, `running`, `sitting`, `talking`, `interacting`, `waiting`, `working`, `driving`, `entering`, `exiting`, `none`

### Ambiguity Notes
- **CRITICAL SCHEMA DRIFT**: The VLM sometimes returns activities as objects:
  ```json
  {"subject_id": "person_1", "type": "standing", "condition": "holding phone"}
  ```
  instead of:
  ```json
  "standing"
  ```
- The `normalize_metadata_dict()` function in `vlm_utils.py` coerces activities via `str(item)`, which would stringify the dict to `"{'subject_id': 'person_1', ...}"` — completely wrong
- No validation enforces the approved vocabulary at the schema level

---

## Relationship Schema (`RelationshipMetadata`)

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `subject_id` | `str` | Yes (empty ok) | `""` | Must reference an `ObjectMetadata.id` |
| `target_id` | `str` | Yes (empty ok) | `""` | Must reference an `ObjectMetadata.id` |
| `relation` | `str` | Yes (empty ok) | `""` | Relationship type (e.g. `talking_to`, `standing_near`) |

### Ambiguity Notes
- No validation enforces that `subject_id` / `target_id` actually reference existing object IDs
- The VLM sometimes creates phantom references (e.g. `person_1`) that don't exist in the `objects` array
- No vocabulary constraint on `relation` values

---

## Location Context Schema (`LocationContextMetadata`)

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `object_id` | `str` | Yes (empty ok) | `""` | Must reference an `ObjectMetadata.id` |
| `location` | `str` | Yes (empty ok) | `""` | Spatial position (e.g. `near_counter`, `center_area`) |

### Ambiguity Notes
- No validation enforces that `object_id` references an existing object ID
- No vocabulary constraint on `location` values
- VLM sometimes references objects by generic names not matching any object ID

---

## Normalization Layer Gaps

The current `normalize_metadata_dict()` in [vlm_utils.py](file:///c:/Mukul%20K/vinfo1/video-search-engine/app/services/vlm_utils.py):

1. **Does NOT normalize `relationships`** — raw VLM output passes through unchanged
2. **Does NOT normalize `location_context`** — raw VLM output passes through unchanged
3. **Does NOT validate `activities` vocabulary** — only coerces to `List[str]` via `str(item)`
4. **Does NOT check reference integrity** — dangling subject_id/target_id/object_id silently accepted
5. **Does NOT handle activity schema drift** — dict activities are stringified, not extracted
