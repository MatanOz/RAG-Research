"""Export utilities for the evaluation dashboard payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def export_ui_dashboard_data(payload: Dict[str, Any], output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path

