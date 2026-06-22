"""
Metadata Stability Audit – Phase 0
===================================
Validates VLM output against FrameRichMetadata schema to measure
quality of objects, activities, relationships, and location_context
BEFORE redesigning the Event Aggregator.

DO NOT modify any production code. This is a read-only audit script.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List
import traceback

project_root = Path(r"c:\Mukul K\vinfo1\video-search-engine")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

os.environ["MOCK_MODEL"] = "False"
os.environ["VLM_ENGINE_TYPE"] = "native_hf"

from app.services.qwen_vlm_hf import NativeQwenTransformersService
from app.services.vlm_utils import clean_json_response

# ─── Constants ──────────────────────────────────────────────────────────
APPROVED_ACTIVITIES = {
    "standing", "walking", "running", "sitting", "talking",
    "interacting", "waiting", "working", "driving",
    "entering", "exiting", "none",
}

FRAMES_DIR = project_root / "validation" / "vlm" / "frames"

TEST_FRAMES = [
    FRAMES_DIR / "empy_room_ts1.jpg",
    FRAMES_DIR / "empy_room_ts7.jpg",
    FRAMES_DIR / "empy_room_ts14.jpg",
    FRAMES_DIR / "person_walk_ts5.jpg",
    FRAMES_DIR / "person_walk_ts18.jpg",
    FRAMES_DIR / "person_walk_ts28.jpg",
    FRAMES_DIR / "customer_int_ts5.jpg",
    FRAMES_DIR / "customer_int_ts15.jpg",
    FRAMES_DIR / "customer_int_ts35.jpg",
]

V3_PROMPT = """Analyze the image.

Return ONLY valid JSON.

Do NOT return:
* caption
* keywords
* scene_description
* events
* people_count

Return ONLY:
{
  "scene_type": "",
  "objects": [
    {
      "id": "",
      "type": "",
      "subtype": "",
      "condition": ""
    }
  ],
  "activities": [],
  "relationships": [
    {
      "subject_id": "",
      "target_id": "",
      "relation": ""
    }
  ],
  "location_context": [
    {
      "object_id": "",
      "location": ""
    }
  ]
}

Activities MUST be selected from:
* standing
* walking
* running
* sitting
* talking
* interacting
* waiting
* working
* driving
* entering
* exiting
* none

Activities MUST be plain strings, NOT objects.
Example: "activities": ["standing", "talking"]

Relationship examples:
* talking_to
* standing_near
* facing
* holding
* approaching
* interacting_with

Location examples:
* near_counter
* near_door
* left_side
* right_side
* center_area
* behind_counter

If evidence exists, do not leave relationships or location_context empty.
Return ONLY JSON."""

OUT_DIR = project_root / "validation" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log_path = OUT_DIR / "metadata_stability_audit.log"
json_out_path = OUT_DIR / "metadata_stability_results.json"

out_f = open(log_path, "w", encoding="utf-8")

def log(msg):
    out_f.write(str(msg) + "\n")
    out_f.flush()
    print(msg)

# ─── Scoring Functions ──────────────────────────────────────────────────

from app.schemas.frame import FrameRichMetadata
from app.services.vlm_utils import normalize_metadata_dict
from app.services.metadata_postprocessor import MetadataPostprocessor

def score_json_quality(raw_outputs: List[str]) -> Dict[str, Any]:
    """Attempt to parse each raw output as JSON, normalize, and postprocess. Returns success rate."""
    valid = 0
    invalid = 0
    parsed_list = []
    
    for i, raw in enumerate(raw_outputs):
        try:
            cleaned = clean_json_response(raw)
            parsed_dict = json.loads(cleaned)
            
            normalized_dict = normalize_metadata_dict(parsed_dict)
            
            normalized_dict["frame_id"] = f"frame_{i}"
            normalized_dict["video_id"] = "test_vid"
            normalized_dict["timestamp_seconds"] = 0.0
            normalized_dict["timestamp_human"] = "00:00:00"
            normalized_dict["frame_path"] = f"test_{i}.jpg"
            
            frame_meta = FrameRichMetadata(**normalized_dict)
            frame_meta = MetadataPostprocessor.process(frame_meta)
            
            valid += 1
            parsed_list.append(frame_meta.model_dump())
        except Exception as e:
            print(f"Validation failed for frame {i}: {e}")
            invalid += 1
            parsed_list.append(None)
    
    total = valid + invalid
    return {
        "valid": valid,
        "invalid": invalid,
        "total": total,
        "parse_success_pct": (valid / total * 100) if total > 0 else 0.0,
        "parsed_list": parsed_list,
    }


def score_object_quality(parsed_list: List[Any]) -> Dict[str, Any]:
    """Measure object completeness across all parsed frames."""
    total_frames = 0
    frames_with_objects = 0
    total_objects = 0
    objects_with_id = 0
    objects_with_type = 0
    objects_with_subtype = 0
    duplicate_id_frames = 0
    
    for parsed in parsed_list:
        if parsed is None:
            continue
        total_frames += 1
        objects = parsed.get("objects", [])
        if not isinstance(objects, list):
            continue
        
        if len(objects) > 0:
            frames_with_objects += 1
        
        ids_seen = set()
        has_dup = False
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            total_objects += 1
            
            obj_id = obj.get("id", "")
            if obj_id:
                objects_with_id += 1
                if obj_id in ids_seen:
                    has_dup = True
                ids_seen.add(obj_id)
            
            if obj.get("type"):
                objects_with_type += 1
            if obj.get("subtype"):
                objects_with_subtype += 1
        
        if has_dup:
            duplicate_id_frames += 1
    
    completeness = 0.0
    if total_objects > 0:
        id_pct = objects_with_id / total_objects
        type_pct = objects_with_type / total_objects
        subtype_pct = objects_with_subtype / total_objects
        dup_penalty = 1.0 - (duplicate_id_frames / max(1, total_frames))
        completeness = ((id_pct + type_pct + subtype_pct) / 3.0) * dup_penalty * 100
    
    return {
        "total_frames": total_frames,
        "frames_with_objects": frames_with_objects,
        "total_objects": total_objects,
        "objects_with_id": objects_with_id,
        "objects_with_type": objects_with_type,
        "objects_with_subtype": objects_with_subtype,
        "duplicate_id_frames": duplicate_id_frames,
        "completeness_pct": completeness,
    }


def score_activity_quality(parsed_list: List[Any]) -> Dict[str, Any]:
    """Measure activity compliance: are activities plain strings from approved vocab?"""
    total_frames = 0
    frames_with_activities = 0
    total_activities = 0
    compliant_activities = 0
    schema_drift_count = 0  # activities that are objects instead of strings
    
    for parsed in parsed_list:
        if parsed is None:
            continue
        total_frames += 1
        activities = parsed.get("activities", [])
        if not isinstance(activities, list):
            continue
        
        if len(activities) > 0:
            frames_with_activities += 1
        
        for act in activities:
            total_activities += 1
            if isinstance(act, str):
                if act.lower().strip() in APPROVED_ACTIVITIES:
                    compliant_activities += 1
            elif isinstance(act, dict):
                schema_drift_count += 1
    
    compliance_pct = (compliant_activities / total_activities * 100) if total_activities > 0 else 100.0
    
    return {
        "total_frames": total_frames,
        "frames_with_activities": frames_with_activities,
        "total_activities": total_activities,
        "compliant_activities": compliant_activities,
        "schema_drift_count": schema_drift_count,
        "compliance_pct": compliance_pct,
    }


def score_relationship_quality(parsed_list: List[Any]) -> Dict[str, Any]:
    """Measure relationship completeness."""
    total_frames = 0
    frames_with_relationships = 0
    total_relationships = 0
    complete_relationships = 0
    missing_subject = 0
    missing_target = 0
    missing_relation = 0
    
    for parsed in parsed_list:
        if parsed is None:
            continue
        total_frames += 1
        rels = parsed.get("relationships", [])
        if not isinstance(rels, list):
            continue
        
        if len(rels) > 0:
            frames_with_relationships += 1
        
        for rel in rels:
            if not isinstance(rel, dict):
                continue
            total_relationships += 1
            
            has_subject = bool(rel.get("subject_id"))
            has_target = bool(rel.get("target_id"))
            has_relation = bool(rel.get("relation"))
            
            if not has_subject:
                missing_subject += 1
            if not has_target:
                missing_target += 1
            if not has_relation:
                missing_relation += 1
            
            if has_subject and has_target and has_relation:
                complete_relationships += 1
    
    completeness_pct = (complete_relationships / total_relationships * 100) if total_relationships > 0 else 100.0
    
    return {
        "total_frames": total_frames,
        "frames_with_relationships": frames_with_relationships,
        "total_relationships": total_relationships,
        "complete_relationships": complete_relationships,
        "missing_subject": missing_subject,
        "missing_target": missing_target,
        "missing_relation": missing_relation,
        "completeness_pct": completeness_pct,
    }


def score_location_context_quality(parsed_list: List[Any]) -> Dict[str, Any]:
    """Measure location context completeness."""
    total_frames = 0
    frames_with_loc = 0
    total_loc = 0
    complete_loc = 0
    missing_object_id = 0
    missing_location = 0
    
    for parsed in parsed_list:
        if parsed is None:
            continue
        total_frames += 1
        locs = parsed.get("location_context", [])
        if not isinstance(locs, list):
            continue
        
        if len(locs) > 0:
            frames_with_loc += 1
        
        for loc in locs:
            if not isinstance(loc, dict):
                continue
            total_loc += 1
            
            has_obj_id = bool(loc.get("object_id"))
            has_location = bool(loc.get("location"))
            
            if not has_obj_id:
                missing_object_id += 1
            if not has_location:
                missing_location += 1
            
            if has_obj_id and has_location:
                complete_loc += 1
    
    completeness_pct = (complete_loc / total_loc * 100) if total_loc > 0 else 100.0
    
    return {
        "total_frames": total_frames,
        "frames_with_loc": frames_with_loc,
        "total_loc": total_loc,
        "complete_loc": complete_loc,
        "missing_object_id": missing_object_id,
        "missing_location": missing_location,
        "completeness_pct": completeness_pct,
    }


def score_reference_integrity(parsed_list: List[Any]) -> Dict[str, Any]:
    """Check that relationship subject_id/target_id and location_context object_id
    reference real object IDs within the same frame."""
    total_refs = 0
    valid_refs = 0
    dangling_refs = 0
    dangling_details = []
    
    for i, parsed in enumerate(parsed_list):
        if parsed is None:
            continue
        
        objects = parsed.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        object_ids = {str(obj.get("id", "")) for obj in objects if isinstance(obj, dict) and obj.get("id")}
        
        # Check relationships
        rels = parsed.get("relationships", [])
        if isinstance(rels, list):
            for rel in rels:
                if not isinstance(rel, dict):
                    continue
                for field in ["subject_id", "target_id"]:
                    ref_id = rel.get(field, "")
                    if ref_id:
                        total_refs += 1
                        if ref_id in object_ids:
                            valid_refs += 1
                        else:
                            dangling_refs += 1
                            dangling_details.append(f"Frame {i}: relationship.{field}='{ref_id}' not in objects")
        
        # Check location_context
        locs = parsed.get("location_context", [])
        if isinstance(locs, list):
            for loc in locs:
                if not isinstance(loc, dict):
                    continue
                ref_id = loc.get("object_id", "")
                if ref_id:
                    total_refs += 1
                    if ref_id in object_ids:
                        valid_refs += 1
                    else:
                        dangling_refs += 1
                        dangling_details.append(f"Frame {i}: location_context.object_id='{ref_id}' not in objects")
    
    integrity_pct = (valid_refs / total_refs * 100) if total_refs > 0 else 100.0
    
    return {
        "total_refs": total_refs,
        "valid_refs": valid_refs,
        "dangling_refs": dangling_refs,
        "integrity_pct": integrity_pct,
        "dangling_details": dangling_details,
    }


# ─── Main ────────────────────────────────────────────────────────────────

def run_audit():
    # Verify frames exist
    missing = [f for f in TEST_FRAMES if not f.exists()]
    if missing:
        log(f"ERROR: Missing frames: {missing}")
        return
    
    log(f"=== METADATA STABILITY AUDIT ===")
    log(f"Frames: {len(TEST_FRAMES)}")
    log(f"Prompt: V3 Minimal Schema")
    log(f"Backend: NativeQwenTransformersService")
    log(f"")
    
    # Run inference
    log("Running VLM inference on all frames...")
    raw_outputs = NativeQwenTransformersService.generate_batch(TEST_FRAMES, V3_PROMPT)
    log(f"Inference complete. Got {len(raw_outputs)} outputs.")
    log("")
    
    # Log raw outputs
    for i, raw in enumerate(raw_outputs):
        log(f"--- Frame {i+1}: {TEST_FRAMES[i].name} ---")
        log(raw)
        log("")
    
    # Task 3: Measure quality
    log("=== SCORING ===")
    
    # JSON Quality
    json_q = score_json_quality(raw_outputs)
    log(f"JSON Quality: {json_q['valid']}/{json_q['total']} valid ({json_q['parse_success_pct']:.1f}%)")
    
    parsed_list = json_q["parsed_list"]
    
    # Object Quality
    obj_q = score_object_quality(parsed_list)
    log(f"Object Completeness: {obj_q['completeness_pct']:.1f}%")
    log(f"  Total objects: {obj_q['total_objects']}")
    log(f"  With IDs: {obj_q['objects_with_id']}")
    log(f"  With types: {obj_q['objects_with_type']}")
    log(f"  With subtypes: {obj_q['objects_with_subtype']}")
    log(f"  Frames with duplicate IDs: {obj_q['duplicate_id_frames']}")
    
    # Activity Quality
    act_q = score_activity_quality(parsed_list)
    log(f"Activity Compliance: {act_q['compliance_pct']:.1f}%")
    log(f"  Total activities: {act_q['total_activities']}")
    log(f"  Compliant: {act_q['compliant_activities']}")
    log(f"  Schema drift (objects instead of strings): {act_q['schema_drift_count']}")
    
    # Relationship Quality
    rel_q = score_relationship_quality(parsed_list)
    log(f"Relationship Completeness: {rel_q['completeness_pct']:.1f}%")
    log(f"  Total relationships: {rel_q['total_relationships']}")
    log(f"  Complete: {rel_q['complete_relationships']}")
    log(f"  Missing subject_id: {rel_q['missing_subject']}")
    log(f"  Missing target_id: {rel_q['missing_target']}")
    log(f"  Missing relation: {rel_q['missing_relation']}")
    
    # Location Context Quality
    loc_q = score_location_context_quality(parsed_list)
    log(f"Location Context Completeness: {loc_q['completeness_pct']:.1f}%")
    log(f"  Total entries: {loc_q['total_loc']}")
    log(f"  Complete: {loc_q['complete_loc']}")
    log(f"  Missing object_id: {loc_q['missing_object_id']}")
    log(f"  Missing location: {loc_q['missing_location']}")
    
    # Task 4: Reference Integrity
    ref_q = score_reference_integrity(parsed_list)
    log(f"Reference Integrity: {ref_q['integrity_pct']:.1f}%")
    log(f"  Total refs: {ref_q['total_refs']}")
    log(f"  Valid refs: {ref_q['valid_refs']}")
    log(f"  Dangling refs: {ref_q['dangling_refs']}")
    if ref_q['dangling_details']:
        log("  Dangling details:")
        for d in ref_q['dangling_details']:
            log(f"    - {d}")
    
    # Task 5: Aggregator Readiness Score
    readiness = (
        json_q["parse_success_pct"] +
        obj_q["completeness_pct"] +
        act_q["compliance_pct"] +
        rel_q["completeness_pct"] +
        loc_q["completeness_pct"] +
        ref_q["integrity_pct"]
    ) / 6.0
    
    log("")
    log(f"=== AGGREGATOR READINESS SCORE: {readiness:.1f}% ===")
    
    # Save JSON results
    results = {
        "frames": [str(f.name) for f in TEST_FRAMES],
        "json_quality": {k: v for k, v in json_q.items() if k != "parsed_list"},
        "object_quality": obj_q,
        "activity_quality": act_q,
        "relationship_quality": rel_q,
        "location_context_quality": loc_q,
        "reference_integrity": ref_q,
        "aggregator_readiness_pct": readiness,
        "raw_outputs": raw_outputs,
        "parsed_outputs": [p for p in parsed_list if p is not None],
    }
    
    with open(json_out_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, default=str)
    
    log(f"\nResults saved to {json_out_path}")


if __name__ == "__main__":
    try:
        run_audit()
    except Exception as e:
        log(f"UNHANDLED EXCEPTION: {e}")
        log(traceback.format_exc())
    log("AUDIT FINISHED.")
    out_f.close()
