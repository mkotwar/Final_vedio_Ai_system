import sys
import json
from pathlib import Path

ROOT = Path(r"c:\Mukul K\vinfo1\video-search-engine")
sys.path.insert(0, str(ROOT))

from app.services.report_service import ReportService
from app.services.embedding_service import EmbeddingService

print("Initializing backend...")
EmbeddingService.initialize()

# Known video IDs
test_videos = {
    "Fall Video": "284e527c-888c-4c80-96c8-3cd7d50731b3",
    "Office Video": "09c162d9-b006-444b-8783-89b3ed025420",
    "Accident Video": "a48b4d08-7e3c-4aa3-a801-0756875508b8",
    "Traffic Video": "95fbbd94-1f3b-47f3-9b17-52c29a8bd638"
}

out_text = []

for name, vid in test_videos.items():
    print(f"Generating report for {name}...")
    try:
        report = ReportService.generate_report(vid)
        out_text.append(f"====== {name} ======")
        out_text.append(f"Title: {report.title}")
        out_text.append(f"Risk Level: {report.risk_level}")
        out_text.append(f"Executive Summary: {report.executive_summary}\n")
        
        out_text.append("--- Critical Findings ---")
        for f in report.critical_findings:
            out_text.append(f"[{f.severity}] {f.event_type} at {f.timestamp}: {f.description}")
            
        out_text.append("\n--- Recommendations ---")
        for r in report.recommendations:
            out_text.append(f"• {r}")
            
        out_text.append("\n--- Timeline Extract ---")
        for t in report.timeline[:5]:  # Just show first 5
            out_text.append(f"{t.timestamp}: {t.description}")
        
        out_text.append("\n\n")
    except Exception as e:
        out_text.append(f"Failed to generate {name}: {e}\n\n")
        import traceback
        traceback.print_exc()

with open(ROOT / "report_validation.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_text))

print("Validation completed. Check report_validation.txt")
