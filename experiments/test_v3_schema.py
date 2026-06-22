import sys
import os
import json
from pathlib import Path
import traceback

project_root = Path(r"c:\Mukul K\vinfo1\video-search-engine")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

out_file_path = project_root / "experiments" / "v3_results.log"
json_out_path = project_root / "experiments" / "v3_results.json"
out_f = open(out_file_path, "w", encoding="utf-8")

def log(msg):
    out_f.write(str(msg) + "\n")
    out_f.flush()
    print(msg) # also print to console so user sees progress

log("SCRIPT STARTED.")

try:
    from app.services.qwen_vlm_hf import NativeQwenTransformersService
    log("IMPORTED NativeQwenTransformersService")
except Exception as e:
    log(f"IMPORT FAILED: {e}")
    log(traceback.format_exc())
    sys.exit(1)

def run_experiment():
    frames_dir = project_root / "validation" / "vlm" / "frames"
    
    image_paths = [
        frames_dir / "empy_room_ts7.jpg",
        frames_dir / "person_walk_ts18.jpg",
        frames_dir / "customer_int_ts35.jpg"
    ]
    
    for ip in image_paths:
        if not ip.exists():
            log(f"Error: Could not find frame at {ip}")
            return

    prompt = """Analyze the image.

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

    log(f"Running Native HF experiment on 3 frames.")
    
    try:
        log("Loading model and generating batch...")
        outputs = NativeQwenTransformersService.generate_batch(image_paths, prompt)
        log("Generation finished.")
        
        results = []
        
        total_objects = 0
        total_activities = 0
        total_relationships = 0
        total_loc_context = 0
        
        for i, raw_out in enumerate(outputs):
            log(f"\n=== FRAME {i+1} RAW MODEL OUTPUT ===")
            log(raw_out)
            
            # Try to parse JSON
            cleaned_out = raw_out.strip()
            if cleaned_out.startswith("```json"):
                cleaned_out = cleaned_out[7:]
            if cleaned_out.startswith("```"):
                cleaned_out = cleaned_out[3:]
            if cleaned_out.endswith("```"):
                cleaned_out = cleaned_out[:-3]
            cleaned_out = cleaned_out.strip()
            
            log(f"\n=== FRAME {i+1} PARSED JSON ===")
            try:
                parsed = json.loads(cleaned_out)
                log(json.dumps(parsed, indent=2))
                
                obj_c = len(parsed.get("objects", []))
                act_c = len(parsed.get("activities", []))
                rel_c = len(parsed.get("relationships", []))
                loc_c = len(parsed.get("location_context", []))
                
                total_objects += obj_c
                total_activities += act_c
                total_relationships += rel_c
                total_loc_context += loc_c
                
                results.append({
                    "frame": str(image_paths[i].name),
                    "raw": raw_out,
                    "parsed": parsed,
                    "metrics": {
                        "objects": obj_c,
                        "activities": act_c,
                        "relationships": rel_c,
                        "location_context": loc_c
                    }
                })
                
            except json.JSONDecodeError:
                log("Failed to parse JSON")
                log(f"Cleaned string: {cleaned_out}")
                results.append({
                    "frame": str(image_paths[i].name),
                    "raw": raw_out,
                    "error": "Failed to parse JSON"
                })
        
        avg_objects = total_objects / 3.0
        avg_activities = total_activities / 3.0
        avg_relationships = total_relationships / 3.0
        avg_loc_context = total_loc_context / 3.0
        
        log("\n=== AVERAGES ===")
        log(f"Avg Objects: {avg_objects:.2f}")
        log(f"Avg Activities: {avg_activities:.2f}")
        log(f"Avg Relationships: {avg_relationships:.2f}")
        log(f"Avg Location Context: {avg_loc_context:.2f}")
        
        final_output = {
            "averages": {
                "objects": avg_objects,
                "activities": avg_activities,
                "relationships": avg_relationships,
                "location_context": avg_loc_context
            },
            "results": results
        }
        
        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(final_output, jf, indent=2)
            
    except Exception as e:
        log(f"Error during API call: {e}")
        log(traceback.format_exc())

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        log(f"UNHANDLED EXCEPTION: {e}")
        log(traceback.format_exc())
    log("SCRIPT FINISHED.")
    out_f.close()
