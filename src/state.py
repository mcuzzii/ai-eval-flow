import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

STATUS_PENDING  = "pending"
STATUS_RUNNING  = "running"
STATUS_FLAGGED  = "flagged"
STATUS_APPROVED = "approved"
STATUS_EDITED   = "edited"
STATUS_SKIPPED  = "skipped"

@dataclass
class Pair:
    idx:          int
    question:     str
    answer:       str
    reference:    str
    final_answer: str         = ""
    critique:     Optional[dict] = None
    status:       str         = STATUS_PENDING
    flagged_at:   Optional[str] = None

    def __post_init__(self):
        if not self.final_answer:
            self.final_answer = self.answer


class AppState:
    """
    Single object shared between the Streamlit main thread and the
    background asyncio critic thread.  Plain attribute reads/writes on
    CPython are GIL-protected, so no explicit lock is needed for simple
    status updates.  The lock below is used only for the bulk load and
    save operations.
    """

    def __init__(self):
        self.pairs:           list[Pair] = []
        self.loaded:          bool       = False
        self.worker_started:  bool       = False
        self.max_concurrent:  int        = 3
        self._lock = threading.Lock()

    # ── loading ───────────────────────────────────────────────────────────────
    def load_pairs(self, data: list[dict]):
        with self._lock:
            self.pairs = [
                Pair(
                    idx       = i,
                    question  = item["question"],
                    answer    = item["answer"],
                    reference = item.get("answer_reference", ""),
                )
                for i, item in enumerate(data)
            ]
            self.loaded = True

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str):
        with self._lock:
            out = [
                {
                    "question":         p.question,
                    "answer":           p.final_answer,
                    "answer_reference": p.reference,
                    "original_answer":  p.answer,
                    "critique":         p.critique,
                    "status":           p.status,
                }
                for p in self.pairs
            ]
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    # ── convenience counts (called from Streamlit thread only) ────────────────
    def counts(self) -> dict:
        c = {s: 0 for s in [STATUS_PENDING, STATUS_RUNNING, STATUS_FLAGGED,
                             STATUS_APPROVED, STATUS_EDITED, STATUS_SKIPPED]}
        for p in self.pairs:
            c[p.status] = c.get(p.status, 0) + 1
        return c
