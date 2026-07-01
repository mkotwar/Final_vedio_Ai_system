from __future__ import annotations

from pathlib import Path
from typing import Any

from tests.final_demo.services.class_propagation_audit import write_class_propagation_audit


def run_step_07C_class_propagation_audit(run_dir: Path) -> dict[str, Any]:
    print("[final-demo] Starting Step 7C: final class propagation audit")

    audit_result = write_class_propagation_audit(run_dir, mode="final")
    report_payload = audit_result["audit_result"]["report_payload"]
    print(f"[final-demo] Final class propagation status: {report_payload['class_propagation_status']}")
    print(f"[final-demo] Warning count: {len(list(report_payload.get('warnings') or []))}")
    print(f"[final-demo] Audit path: {audit_result['audit_path']}")
    print(f"[final-demo] Report path: {audit_result['report_path']}")

    return {
        "run_dir": run_dir,
        "audit_path": audit_result["audit_path"],
        "report_path": audit_result["report_path"],
        "audit_result": audit_result["audit_result"],
    }


def main() -> None:
    raise RuntimeError(
        "Run Step 7C from the final demo pipeline or call run_step_07C_class_propagation_audit(run_dir)."
    )


if __name__ == "__main__":
    main()
