"""Export utilities for the evaluation dashboard payload."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _payload_passthrough(payload: Dict[str, Any]) -> Dict[str, Any]:
    # Preserve all evaluator fields (including future agent-trace keys) without pruning.
    return json.loads(json.dumps(payload, ensure_ascii=False))


def export_ui_dashboard_data(payload: Dict[str, Any], output_path: Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _payload_passthrough(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe_payload, handle, indent=2, ensure_ascii=False)
    return path
