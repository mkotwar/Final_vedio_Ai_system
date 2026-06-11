import os
import sys
from pathlib import Path

ROOT = Path(r"C:\Mukul K\vinfo1\video-search-engine")

targets = [
    "patch.py", "patch2.py", "patch3.py", "patch4.py", "patch5.py", "patch_search.py", "patch_thumbnails.py",
    "scratch_metrics.py", "scratch_sim.py", "scratch_test_aggregation.py", "scratch_test_search.py",
    "debug_aggregation.py", "debug_overlap.py", "debug_trace.py",
    "out.txt", "out50.txt", "out_cmd.txt", "pure_log.txt", "diag_results.txt", "diagnostic_output.txt"
]

deleted = []
missing = []
unexpected = []
space_saved = 0

def search_refs(target):
    name_no_ext = Path(target).stem
    dirs = [ROOT / "app", ROOT / "frontend"]
    for d in dirs:
        if not d.exists(): continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix in [".py", ".ts", ".js", ".html"]:
                try:
                    text = p.read_text("utf-8")
                    if name_no_ext in text or target in text: return str(p)
                except: pass
    return None

for t in targets:
    p = ROOT / t
    if not p.exists():
        missing.append(t)
        continue
    ref = search_refs(t)
    if ref:
        unexpected.append(f"{t} (Found in {ref})")
    else:
        try:
            space_saved += p.stat().st_size
            p.unlink()
            deleted.append(t)
        except Exception as e:
            unexpected.append(f"{t} (Failed to delete: {e})")

val_msg = ""
try:
    sys.path.insert(0, str(ROOT))
    import app.main
    val_msg = "Backend imports successful. No broken imports or missing module errors."
except Exception as e:
    val_msg = f"IMPORT ERROR: {e}"

report = [
    "# Safe Cleanup Execution Report\n",
    "## Validation Before Deletion",
    "Every file was scanned across the `app/` and `frontend/` codebase. Zero references were found for the deleted files.\n",
    "## Deleted Files"
]
for d in deleted: report.append(f"- `[x]` {d}")
report.append("\n## Missing Files (Already Deleted or Missing)")
for m in missing: report.append(f"- {m}")
report.append("\n## Unexpected References or Errors")
if not unexpected: report.append("None. All targeted files were safely processed.")
else:
    for u in unexpected: report.append(f"- {u}")
report.append(f"\n## Space Recovered\n- **File count reduction**: {len(deleted)}")
report.append(f"- **Disk space recovered**: {space_saved / (1024*1024):.2f} MB\n")
report.append("## Post-Cleanup Validation\n- " + val_msg)

artifact_path = r"C:\Users\Vinfocom\.gemini\antigravity-ide\brain\3b9898a7-6b14-4a67-8f53-d36e2b526aa7\walkthrough.md"
with open(artifact_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
