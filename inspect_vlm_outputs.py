import json
from pathlib import Path

run = Path(r"C:\Mukul K\vinfo1\video-search-engine\tests\tender_demo_case\debug_runs\localcam1_20260706_110059")
p = run / "16_topk_vlm_outputs.json"

if not p.exists():
    raise FileNotFoundError(p)

data = json.loads(p.read_text(encoding="utf-8"))

print("TOP LEVEL TYPE:", type(data).__name__)

if isinstance(data, dict):
    print("TOP LEVEL KEYS:", list(data.keys()))

items = None

for key in ["outputs", "results", "vlm_outputs", "items", "clip_outputs"]:
    if isinstance(data, dict) and isinstance(data.get(key), list):
        items = data[key]
        print("FOUND ITEMS KEY:", key)
        break

if items is None and isinstance(data, list):
    items = data

if not items:
    print("NO ITEMS FOUND")
else:
    print("ITEM COUNT:", len(items))
    first = items[0]
    print("FIRST ITEM KEYS:", list(first.keys()))
    print("\nFIRST ITEM PREVIEW:")
    print(json.dumps(first, indent=2, ensure_ascii=False)[:5000])

    print("\nALL ITEM SUMMARY:")
    for i, item in enumerate(items):
        print("\n--- ITEM", i, "---")
        if isinstance(item, dict):
            for key in [
                "clip_id",
                "success",
                "error",
                "raw_text",
                "raw_output",
                "generated_text",
                "output_text",
                "vlm_output",
                "model_output",
                "text",
                "response",
                "parsed_json",
            ]:
                value = item.get(key)
                if value is None:
                    continue

                value_str = str(value)
                print(f"{key}: {value_str[:500]}")
        else:
            print(str(item)[:500])