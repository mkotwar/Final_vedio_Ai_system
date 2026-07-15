from __future__ import annotations

import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from td_case2_workbench_ui import main


if __name__ == "__main__":
    main(configure_page=True, initial_section="Search & Clips")
