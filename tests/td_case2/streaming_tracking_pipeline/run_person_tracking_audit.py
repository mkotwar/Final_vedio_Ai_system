from __future__ import annotations

import argparse
from pathlib import Path

from .person_tracking_support import build_person_tracking_audit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit person tracking support for saved streaming tracking artifacts.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--vehicle-model-path", default="object/vehical_detection/best_old.pt")
    parser.add_argument("--person-model-path", default="object/Person_detection.pt")
    parser.add_argument("--output-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output_path = args.output_path or str(Path(args.run_dir) / "person_tracking_audit.json")
    audit = build_person_tracking_audit(
        run_dir=args.run_dir,
        vehicle_model_path=args.vehicle_model_path,
        person_model_path=args.person_model_path,
        output_path=output_path,
    )
    print("Person tracking audit complete")
    print(f"vehicle_detector_supports_person={audit['vehicle_detector_supports_person']}")
    print(f"person_detector_configured={audit['person_detector_configured']}")
    print(f"person_detector_load_status={audit['person_detector_load_status']}")
    print(f"person_records_written={audit['person_records_written']}")
    print(f"root_causes={audit['root_causes']}")
    print(f"audit_path={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
