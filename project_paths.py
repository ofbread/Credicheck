"""Central layout: datasets under data/, run artifacts under outputs/."""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent
DATA = _ROOT / "data"
OUTPUTS = _ROOT / "outputs"

__all__ = ["DATA", "OUTPUTS"]
