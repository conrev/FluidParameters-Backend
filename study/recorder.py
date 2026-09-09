#!/usr/bin/env python
"""
study/recorder.py — append-only telemetry for end-to-end (human) calibration sessions.

Everything needed to reconstruct and analyse a user-study session is written as newline-delimited
JSON, one event per line, flushed on every write, so a crash or a dropped WebSocket never costs
more than the event in flight. The recorder is NOT on the critical path of the optimisation: it
only stores values the session has already computed, plus two cheap posterior look-ups per duel
(the model's belief about the pair being shown, used later for consistency statistics).

Layout — one self-contained directory per session:

    sessions/20260909-141233_P01_a1b2c3/
        session.json    header: participant, config, parameter space, environment + versions
        events.jsonl    the full event stream (session_start, duel, preference, result, session_end)
        result.json     the final calibration result payload (also present in events.jsonl)

`study.analyse` turns these into per-session and cohort statistics; see study/metrics.py for what
is computed and why.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = 1

#: Where sessions are written. Override per deployment without touching code.
DEFAULT_ROOT = Path(os.environ.get("PBO_SESSION_LOG_DIR", "sessions"))

#: Set PBO_DISABLE_SESSION_LOG=1 to run the backend with recording off (e.g. public demos).
LOGGING_DISABLED = os.environ.get("PBO_DISABLE_SESSION_LOG", "").strip() in ("1", "true", "yes")


def _utc() -> str:
    """Wall-clock timestamp (ISO-8601, UTC, millisecond precision)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _git_commit() -> Optional[str]:
    """Short commit of the backend that produced this session (provenance for the paper)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
        return None


def _versions() -> dict:
    """Library versions, so a session can be reproduced years later."""
    out: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "botorch", "gpytorch", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


class SessionRecorder:
    """
    Records one calibration session to disk.

    Parameters
    ----------
    participant : de-identified participant code (e.g. "P01"). Never store names or emails here.
    condition   : study condition / arm, if the design has more than one.
    notes       : free-text protocol notes (site, facilitator, apparatus).
    extra       : any additional study metadata supplied by the client, stored verbatim.
    root        : session root directory (defaults to $PBO_SESSION_LOG_DIR or ./sessions).

    All methods are no-ops-safe: recording failures are swallowed and reported once, because losing
    telemetry must never take down a live session with a participant in the room.
    """

    def __init__(
        self,
        participant: Optional[str] = None,
        condition: Optional[str] = None,
        notes: Optional[str] = None,
        extra: Optional[dict] = None,
        root: Optional[Path | str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex
        self.participant = participant or "anon"
        self.condition = condition
        self.notes = notes
        self.extra = extra or {}

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        safe = "".join(c for c in self.participant if c.isalnum() or c in "-_")[:24] or "anon"
        self.dir = Path(root or DEFAULT_ROOT) / f"{stamp}_{safe}_{self.session_id[:6]}"
        self.events_path = self.dir / "events.jsonl"

        self._lock = threading.Lock()
        self._t0 = time.perf_counter()  # monotonic base for all durations
        self._duel_sent_perf: dict[str, float] = {}  # duel_id -> perf counter at send
        self._closed = False
        self._warned = False
        self._n_duels = 0
        self._n_preferences = 0

        self.dir.mkdir(parents=True, exist_ok=True)

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _write(self, kind: str, payload: dict) -> None:
        """Append one event. Never raises."""
        if self._closed:
            return
        record = {"schema": SCHEMA_VERSION, "event": kind, "at": _utc(),
                  "t": round(time.perf_counter() - self._t0, 6), **payload}
        try:
            with self._lock, open(self.events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                print(f"Warning > session recording failed ({exc}); continuing without telemetry")
                self._warned = True

    def _write_json(self, name: str, payload: dict) -> None:
        try:
            with open(self.dir / name, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning > could not write {name}: {exc}")

    # ── events ───────────────────────────────────────────────────────────────

    def session_start(self, config: dict, param_space: dict[str, list]) -> None:
        """Record the session header: who, what protocol, which candidate space, which build."""
        n_candidates = 1
        for values in param_space.values():
            n_candidates *= max(1, len(values))
        space = {
            "params": {k: list(v) for k, v in param_space.items()},
            "sizes": {k: len(v) for k, v in param_space.items()},
            "dims": len(param_space),
            "nCandidates": n_candidates,
        }
        header = {
            "sessionId": self.session_id,
            "participant": self.participant,
            "condition": self.condition,
            "notes": self.notes,
            "extra": self.extra,
            "config": config,
            "space": space,
            "env": {**_versions(), "gitCommit": _git_commit()},
            "startedAt": _utc(),
        }
        self._write_json("session.json", {"schema": SCHEMA_VERSION, **header})
        self._write("session_start", header)

    def duel(
        self,
        *,
        index: int,
        duel_id: str,
        phase: str,
        a_index: int,
        b_index: int,
        a_params: dict,
        b_params: dict,
        reference_index: Optional[int] = None,
        swapped: bool = False,
        prediction: Optional[dict] = None,
        acq_value: Optional[float] = None,
        timings: Optional[dict] = None,
    ) -> None:
        """
        A pair was proposed and sent to the participant.

        `prediction` holds the model's belief about THIS pair, fitted only on the *preceding*
        comparisons — the prequential (one-step-ahead) setup that makes judgement-consistency
        statistics honest. `reference_index` is the incumbent in the BO phase (it duels the
        challenger), and `swapped` records whether the sides were flipped before display.
        """
        self._n_duels += 1
        self._duel_sent_perf[duel_id] = time.perf_counter()
        self._write(
            "duel",
            {
                "index": index,
                "duelId": duel_id,
                "phase": phase,
                "aIndex": a_index,
                "bIndex": b_index,
                "aParams": a_params,
                "bParams": b_params,
                "referenceIndex": reference_index,
                "swapped": swapped,
                "prediction": prediction,
                "acqValue": acq_value,
                "timings": timings or {},
            },
        )

    def preference(
        self,
        *,
        index: int,
        duel_id: str,
        choice: str,
        winner_index: int,
        loser_index: int,
        reference_index: Optional[int] = None,
        client: Optional[dict] = None,
    ) -> None:
        """
        The participant committed a judgement.

        `serverDecisionMs` is measured from the moment the duel left the backend, so it includes
        transport, loading and rendering as well as deliberation. If the client reports its own
        `decisionMs` (time from first frame displayed to click) that is kept alongside, verbatim,
        and is the better number for task-time and fatigue analysis.
        """
        self._n_preferences += 1
        sent = self._duel_sent_perf.pop(duel_id, None)
        decision_ms = None if sent is None else round((time.perf_counter() - sent) * 1e3, 1)
        self._write(
            "preference",
            {
                "index": index,
                "duelId": duel_id,
                "choice": choice,
                "winnerIndex": winner_index,
                "loserIndex": loser_index,
                "referenceIndex": reference_index,
                "winnerIsReference": (
                    None if reference_index is None else winner_index == reference_index
                ),
                "serverDecisionMs": decision_ms,
                "client": client or {},
            },
        )

    def result(self, payload: dict, compute_ms: Optional[float] = None) -> None:
        """The final calibration result was produced and returned to the client."""
        self._write_json("result.json", payload)
        self._write("result", {"computeMs": compute_ms, "result": payload})

    def note(self, kind: str, **fields: Any) -> None:
        """Escape hatch for anything else worth recording (errors, facilitator notes, ...)."""
        self._write(kind, fields)

    def close(self, reason: str = "completed") -> None:
        """Finalise the session. Safe to call twice (e.g. result then disconnect)."""
        if self._closed:
            return
        self._write(
            "session_end",
            {
                "reason": reason,
                "durationS": round(time.perf_counter() - self._t0, 3),
                "duels": self._n_duels,
                "preferences": self._n_preferences,
            },
        )
        self._closed = True


def make_recorder(meta: Optional[dict] = None) -> Optional[SessionRecorder]:
    """
    Build a recorder from client-supplied study metadata, or None when logging is disabled.

    Accepts either a nested block or flat keys, so the Unity client can send whichever is easier:
        {"study": {"participant": "P01", "condition": "A", "notes": "..."}}
        {"participantId": "P01"}
    """
    if LOGGING_DISABLED:
        return None
    meta = dict(meta or {})
    participant = meta.pop("participant", None) or meta.pop("participantId", None)
    condition = meta.pop("condition", None)
    notes = meta.pop("notes", None)
    try:
        return SessionRecorder(
            participant=participant, condition=condition, notes=notes, extra=meta
        )
    except Exception as exc:  # noqa: BLE001 — never block a session on telemetry setup
        print(f"Warning > could not start session recorder: {exc}")
        return None
