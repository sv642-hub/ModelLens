"""Lightweight JSON-friendly snapshots for comparing interpretability across training steps."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainingSnapshot:
    step: int
    epoch: Optional[float] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    notes: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class SnapshotStore:
    """Append-only list of snapshots serializable to JSON."""

    def __init__(self) -> None:
        self.snapshots: List[TrainingSnapshot] = []

    def append(self, snap: TrainingSnapshot) -> None:
        self.snapshots.append(snap)

    def append_dict(self, d: Dict[str, Any]) -> None:
        self.snapshots.append(
            TrainingSnapshot(
                step=int(d["step"]),
                epoch=d.get("epoch"),
                train_loss=d.get("train_loss"),
                val_loss=d.get("val_loss"),
                notes=str(d.get("notes", "")),
                metrics=dict(d.get("metrics", {})),
            )
        )

    def to_list(self) -> List[Dict[str, Any]]:
        return [asdict(s) for s in self.snapshots]

    def save_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_list(), indent=2))

    @classmethod
    def load_json(cls, path: Path) -> "SnapshotStore":
        raw = json.loads(path.read_text())
        st = cls()
        for row in raw:
            st.append_dict(row)
        return st
