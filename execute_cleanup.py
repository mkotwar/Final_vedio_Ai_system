import os
import sys
import json
from pathlib import Path

ROOT = Path(r"c:\Mukul K\vinfo1\video-search-engine")

targets = [
    "patch.py", "patch2.py", "patch3.py", "patch4.py", "patch5.py", "patch_search.py", "patch_thumbnails.py",
    "scratch_metrics.py", "scratch_sim.py", "scratch_test_aggregation.py", "scratch_test_search.py",
    "debug_aggregation.py", "debug_overlap.py", "debug_trace.py",
    "out.txt", "out50.txt", "out_cmd.txt", "pure_log.txt", "diag_results.txt", "diagnostic_output.txt"
]

def search_references(filename):
    name_no_ext = Path(filename).stem
    search_dirs = [ROOT / "app", ROOT / "frontend"]
    for d in search_dirs:
        if not d.exists(): continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in [".py", ".ts", ".js", ".html"]:
                try:
                    text = p.read_text(encoding="utf-8")
                    if name_no_ext in text or filename in text:
                        return str(p.relative_to(ROOT))
                except Exception:
                    pass
    return None

results = {
    "deleted": [],
    "missing": [],
    "unexpected": [],
    "space_saved_mb": 0.0,
    "validation": "Failed"
}

space_saved = 0

for target in targets:
    p = ROOT / target
    if not p.exists():
        results["missing"].append(target)
        continue
    
    ref = search_references(target)
    if ref:
        results["unexpected"].append((target, ref))
    else:
        space_saved += p.stat().st_size
        try:
            p.unlink()
            results["deleted"].append(target)
        except Exception as e:
            results["unexpected"].append((target, f"Failed to delete: {e}"))

results["space_saved_mb"] = space_saved / (1024*1024)

try:
    sys.path.insert(0, str(ROOT))
    import app.main
    from app.services.event_aggregation import EventAggregationService
    from app.services.search_service import SearchService
    from app.services.summary_service import SummaryService
    results["validation"] = "Backend imports successful. No missing module errors."
except Exception as e:
    results["validation"] = f"IMPORT ERROR: {e}"

with open(ROOT / "cleanup_results.json", "w") as f:
    json.dump(results, f, indent=4)
