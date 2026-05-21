# agent/cache.py
"""
Simple disk-backed query cache.

Keys are SHA-256 hashes of (session_csv_path + file_mtime + normalised_question).
The mtime component means the cache auto-invalidates whenever the CSV is
modified, even if its path stays the same.

Values are stored as JSON files in cache/.
Expired entries are pruned lazily on each read.

Usage:
    from agent.cache import cache
    hit = cache.get(csv_path, question)
    if hit:
        return hit["response_text"], hit["plot_paths"]
    # ... run agent ...
    cache.set(csv_path, question, response_text, plot_paths)
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

from agent.config import cfg


class QueryCache:
    def __init__(self, cache_dir: str = cfg.cache_dir, ttl: int = cfg.cache_ttl_seconds):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = cfg.enable_cache

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _key(self, csv_path: str, question: str) -> str:
        """Stable hash key regardless of whitespace differences in question.

        Includes the CSV file's mtime so the cache automatically invalidates
        when the file is modified — even if the path stays the same.
        Falls back to mtime=0 for in-memory / synthetic paths used in tests.
        """
        normalised = " ".join(question.lower().split())

        try:
            mtime = os.path.getmtime(csv_path)
        except OSError:
            mtime = 0

        raw = f"{csv_path}::{mtime}::{normalised}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ─── Public API ───────────────────────────────────────────────────────────

    def get(self, csv_path: str, question: str) -> Optional[dict]:
        """
        Return cached result dict or None.
        Evicts the entry if it has expired.
        """
        if not self.enabled:
            return None

        key = self._key(csv_path, question)
        p = self._path(key)

        if not p.exists():
            return None

        try:
            entry = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            p.unlink(missing_ok=True)
            return None

        if time.time() - entry.get("created_at", 0) > self.ttl:
            p.unlink(missing_ok=True)
            return None

        # Validate that plot files still exist on disk
        entry["plot_paths"] = [
            pp for pp in entry.get("plot_paths", []) if os.path.exists(pp)
        ]
        return entry

    def set(
        self,
        csv_path: str,
        question: str,
        response_text: str,
        plot_paths: list[str],
        iteration_count: int = 0,
    ) -> None:
        """Persist a result to disk."""
        if not self.enabled:
            return

        key = self._key(csv_path, question)
        entry = {
            "created_at": time.time(),
            "csv_path": csv_path,
            "question": question,
            "response_text": response_text,
            "plot_paths": plot_paths,
            "iteration_count": iteration_count,
        }
        try:
            self._path(key).write_text(
                json.dumps(entry, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    def invalidate(self, csv_path: str, question: str) -> None:
        key = self._key(csv_path, question)
        self._path(key).unlink(missing_ok=True)

    def clear_all(self) -> int:
        """Remove all cache entries. Returns count removed."""
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count

    def stats(self) -> dict:
        """Return basic cache statistics."""
        entries = list(self.cache_dir.glob("*.json"))
        expired = 0
        now = time.time()
        for p in entries:
            try:
                e = json.loads(p.read_text())
                if now - e.get("created_at", 0) > self.ttl:
                    expired += 1
            except Exception:
                expired += 1
        return {
            "total_entries": len(entries),
            "expired_entries": expired,
            "live_entries": len(entries) - expired,
            "cache_dir": str(self.cache_dir),
            "ttl_seconds": self.ttl,
        }


# Global singleton
cache = QueryCache()
