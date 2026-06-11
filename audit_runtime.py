import sys
import json
from pathlib import Path

ROOT = Path(r"c:\Mukul K\vinfo1\video-search-engine")
sys.path.insert(0, str(ROOT))

from app.services.summary_service import SummaryService
from app.services.report_service import ReportService
from app.services.search_service import SearchService

def main():
    test_video = "284e527c-888c-4c80-96c8-3cd7d50731b3"  # Fall Video
    out = []
    
    out.append("=== PHASE 9 RUNTIME WIRING AUDIT ===\n")
    
    # 1. Check Events Loading
    events = SummaryService.load_events(test_video)
    out.append(f"1. Events Loaded: {len(events)} from _events_v2.json")
    
    # 2. Check Summary Service integration
    try:
        summary = SummaryService.generate_summary(test_video)
        out.append(f"2. SummaryService output:")
        out.append(f"   - Is `incidents` list populated? {'Yes' if summary.incidents else 'No'}")
        out.append(f"   - Count of macro-incidents: {len(summary.incidents)}")
        if summary.incidents:
            out.append(f"   - Sample Incident Type: {summary.incidents[0].incident_type}")
    except Exception as e:
        out.append(f"2. SummaryService Failed: {e}")
        import traceback
        out.append(traceback.format_exc())

    # 3. Check Report Service integration
    try:
        report = ReportService.generate_report(test_video)
        out.append(f"\n3. ReportService output:")
        out.append(f"   - Risk Level: {report.risk_level}")
        out.append(f"   - Critical Findings Count: {len(report.critical_findings)}")
        if report.critical_findings:
            out.append(f"   - Top Finding: {report.critical_findings[0].event_type} (Severity: {report.critical_findings[0].severity})")
            
        out.append("\n--- ACTUAL GENERATED REPORT JSON DUMP ---")
        out.append(report.model_dump_json(indent=2))
        out.append("------------------------------------------\n")
    except Exception as e:
        out.append(f"3. ReportService Failed: {e}")
        import traceback
        out.append(traceback.format_exc())

    with open(ROOT / "audit_results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out))

if __name__ == "__main__":
    main()
