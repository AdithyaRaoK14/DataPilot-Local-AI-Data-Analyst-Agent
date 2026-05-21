# agent/logger.py
"""
Structured logging for the agent.

Every agent action, tool call, error, and performance metric is written as
a JSON line to logs/agent.jsonl.  The Streamlit sidebar debug panel reads
from this file so you can trace exactly what happened without print-debugging.

Usage:
    from agent.logger import log
    log.tool_call("execute_python_code", code="...", session_id="abc")
    log.tool_result("execute_python_code", output="...", success=True, duration_ms=140)
    log.agent_step(iteration=3, tool_calls=["execute_python_code"])
    log.error("agent_node", exc=e)
"""

import json
import logging
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.config import cfg


# ─── Internal stdlib logger (INFO → logs/agent.log) ───────────────────────────

_LOG_FILE = Path(cfg.log_dir) / "agent.log"
_JSONL_FILE = Path(cfg.log_dir) / "agent.jsonl"

_stdlib_logger = logging.getLogger("analyst_agent")
_stdlib_logger.setLevel(getattr(logging, cfg.log_level, logging.INFO))

if cfg.log_to_file and not _stdlib_logger.handlers:
    _fh = logging.FileHandler(_LOG_FILE)
    _fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    _stdlib_logger.addHandler(_fh)

# Also log to stderr at WARNING+ so Streamlit can show critical issues
_ch = logging.StreamHandler()
_ch.setLevel(logging.WARNING)
_ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
_stdlib_logger.addHandler(_ch)


# ─── JSON-lines writer ─────────────────────────────────────────────────────────

def _write_jsonl(record: dict):
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    try:
        with open(_JSONL_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass  # logging must never crash the agent


# ─── Public API ───────────────────────────────────────────────────────────────

class AgentLogger:
    """Thin wrapper that emits structured JSON-lines and stdlib logs."""

    # ── Session ────────────────────────────────────────────────────────────────

    def session_start(self, session_id: str, csv_path: Optional[str] = None):
        _write_jsonl({"event": "session_start",
                     "session_id": session_id, "csv_path": csv_path})
        _stdlib_logger.info("Session started: %s  file=%s",
                            session_id, csv_path)

    def session_end(self, session_id: str, total_turns: int):
        _write_jsonl(
            {"event": "session_end", "session_id": session_id, "total_turns": total_turns})
        _stdlib_logger.info("Session ended: %s  turns=%d",
                            session_id, total_turns)

    # ── Agent steps ────────────────────────────────────────────────────────────

    def agent_step(
        self,
        iteration: int,
        tool_calls: list[str],
        session_id: str = "",
    ):
        _write_jsonl({
            "event": "agent_step",
            "session_id": session_id,
            "iteration": iteration,
            "tool_calls": tool_calls,
        })
        _stdlib_logger.info(
            "Step %d  tools=%s", iteration, tool_calls
        )

    # ── Tool calls ─────────────────────────────────────────────────────────────

    def tool_call(
        self,
        tool_name: str,
        session_id: str = "",
        **kwargs: Any,
    ):
        _write_jsonl({
            "event": "tool_call",
            "session_id": session_id,
            "tool": tool_name,
            # truncate large args
            **{k: str(v)[:500] for k, v in kwargs.items()},
        })
        _stdlib_logger.info("Tool call: %s  session=%s", tool_name, session_id)

    def tool_result(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        session_id: str = "",
        output_preview: str = "",
    ):
        _write_jsonl({
            "event": "tool_result",
            "session_id": session_id,
            "tool": tool_name,
            "success": success,
            "duration_ms": round(duration_ms, 1),
            "output_preview": output_preview[:300],
        })
        level = logging.INFO if success else logging.WARNING
        _stdlib_logger.log(
            level,
            "Tool result: %s  ok=%s  %dms",
            tool_name, success, int(duration_ms),
        )

    # ── Security events ────────────────────────────────────────────────────────

    def security_block(
        self,
        reason: str,
        code_preview: str = "",
        session_id: str = "",
    ):
        _write_jsonl({
            "event": "security_block",
            "session_id": session_id,
            "reason": reason,
            "code_preview": code_preview[:200],
        })
        _stdlib_logger.warning(
            "SECURITY BLOCK  reason=%s  session=%s", reason, session_id
        )

    # ── Errors ─────────────────────────────────────────────────────────────────

    def error(
        self,
        context: str,
        exc: Optional[Exception] = None,
        session_id: str = "",
        **extra: Any,
    ):
        tb = traceback.format_exc() if exc else ""
        _write_jsonl({
            "event": "error",
            "session_id": session_id,
            "context": context,
            "exception": str(exc) if exc else "",
            "traceback": tb[:1000],
            **extra,
        })
        _stdlib_logger.error(
            "Error in %s: %s", context, exc, exc_info=bool(exc)
        )

    # ── Performance ────────────────────────────────────────────────────────────

    def performance(
        self,
        session_id: str,
        elapsed_s: float,
        iterations: int,
        plots_generated: int,
        error_count: int,
    ):
        _write_jsonl({
            "event": "performance",
            "session_id": session_id,
            "elapsed_s": round(elapsed_s, 2),
            "iterations": iterations,
            "plots_generated": plots_generated,
            "error_count": error_count,
        })
        _stdlib_logger.info(
            "Performance  session=%s  %.1fs  %d iters  %d plots  %d errors",
            session_id, elapsed_s, iterations, plots_generated, error_count,
        )

    # ── Read helpers (for Streamlit sidebar) ───────────────────────────────────

    def read_recent_events(self, n: int = 20) -> list[dict]:
        """Return the last n JSON-lines records, newest last."""
        try:
            lines = _JSONL_FILE.read_text(encoding="utf-8").splitlines()
            records = []
            for line in lines[-n:]:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
            return records
        except FileNotFoundError:
            return []

    def read_session_events(self, session_id: str) -> list[dict]:
        """Return all events for a specific session_id."""
        all_events = self.read_recent_events(500)
        return [e for e in all_events if e.get("session_id") == session_id]


# Global singleton
log = AgentLogger()
