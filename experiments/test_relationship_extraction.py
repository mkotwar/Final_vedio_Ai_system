import sys
import os
import json
from pathlib import Path
import traceback

project_root = Path(r"c:\Mukul K\vinfo1\video-search-engine")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

out_file_path = project_root / "experiments" / "test_output.log"
out_f = open(out_file_path, "w", encoding="utf-8")

def log(msg):
    out_f.write(str(msg) + "\n")
    out_f.flush()

log("SCRIPT STARTED.")

try:
    from app.services.qwen_vlm_hf import NativeQwenTransformersService
    log("IMPORTED NativeQwenTransformersService")
except Exception as e:
    log(f"IMPORT FAILED: {e}")
    log(traceback.format_exc())
    sys.exit(1)

def run_experiment():
    image_path = project_root / "validation" / "vlm_manual_audit" / "frame_041.jpg"
    
    if not image_path.exists():
        log(f"Error: Could not find frame at {image_path}")
        return

    prompt = """Analyze this image.

Return ONLY valid JSON.

Do NOT return scene descriptions.
Do NOT return captions.
Do NOT return keywords.
Do NOT return object lists.

Focus ONLY on:

1. Activities being performed.
2. Relationships between visible people or objects.
3. Spatial location context.

Return ONLY:

{
  "activities": [],
  "relationships": [],
  "location_context": []
}

Activities MUST use only:

standing
walking
running
sitting
talking
interacting
waiting
working
driving
entering
exiting
none

Relationship examples:

talking_to
standing_near
facing
holding
approaching
interacting_with

Location examples:

near_counter
near_door
left_side
right_side
center_area
behind_counter

Do not leave arrays empty if evidence exists in the image."""

    log(f"Running Native HF experiment on: {image_path}")
    
    try:
        log("Loading model and generating batch...")
        outputs = NativeQwenTransformersService.generate_batch([image_path], prompt)
        log("Generation finished.")
        raw_out = outputs[0]
        
        log("\n=== RAW MODEL OUTPUT ===")
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
        
        log("\n=== PARSED JSON ===")
        try:
            parsed = json.loads(cleaned_out)
            log(json.dumps(parsed, indent=2))
            
            act_count = len(parsed.get("activities", []))
            rel_count = len(parsed.get("relationships", []))
            loc_count = len(parsed.get("location_context", []))
            
            log("\n=== COUNTS ===")
            log(f"ACTIVITY COUNT: {act_count}")
            log(f"RELATIONSHIP COUNT: {rel_count}")
            log(f"LOCATION CONTEXT COUNT: {loc_count}")
            
        except json.JSONDecodeError:
            log("Failed to parse JSON")
            log(f"Cleaned string: {cleaned_out}")
            
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
