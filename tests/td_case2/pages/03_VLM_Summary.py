from __future__ import annotations

import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parents[1]
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from td_case2_results_ui import render_vlm_summary_dashboard


if __name__ == "__main__":
    render_vlm_summary_dashboard(configure_page=True, show_navigation_hint=False)
