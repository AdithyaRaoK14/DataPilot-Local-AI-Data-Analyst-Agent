# agent/config.py
"""
Single source of truth for all tuneable parameters.
Import this everywhere instead of hardcoding values.
"""

from dataclasses import dataclass, field
from typing import List
import os


@dataclass
class AgentConfig:
    # ── Model ──────────────────────────────────────────────────────────────────
    model_name: str = "qwen2.5:7b"
    temperature: float = 0.0
    max_tokens: int = 2048

    # ── Execution limits ───────────────────────────────────────────────────────
    max_iterations: int = 12
    max_consecutive_errors: int = 3
    code_timeout_seconds: int = 30

    # ── File ingestion ─────────────────────────────────────────────────────────
    max_upload_mb: float = 50.0
    allowed_extensions: List[str] = field(
        default_factory=lambda: [".csv", ".xlsx", ".xls", ".sqlite", ".db"]
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    sandbox_dir: str = "sandbox"
    log_dir: str = "logs"
    cache_dir: str = "cache"

    # ── Caching ────────────────────────────────────────────────────────────────
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600       # 1 hour

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_to_file: bool = True

    # ── Rate limiting ──────────────────────────────────────────────────────────
    max_requests_per_minute: int = 20

    # ── RAG ────────────────────────────────────────────────────────────────────
    enable_rag: bool = True
    rag_embed_model: str = "nomic-embed-text"   # Ollama embedding model
    rag_top_k: int = 4                          # chunks retrieved per query
    # When True, always fall back to TF-IDF instead of trying Ollama embeddings.
    # Useful in CI/CD where Ollama is not running.
    rag_force_tfidf: bool = False

    # ── SQL ────────────────────────────────────────────────────────────────────
    sql_row_limit: int = 100_000                # cap for table loads
    sql_allowed_pragmas: List[str] = field(
        default_factory=lambda: ["table_info", "index_list", "foreign_key_list"]
    )

    # ── EDA Report ─────────────────────────────────────────────────────────────
    report_max_cat_columns: int = 6            # categorical bar charts per report
    report_max_numeric_columns: int = 12       # histogram panels per report

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    def setup_dirs(self):
        for d in (self.sandbox_dir, self.log_dir, self.cache_dir):
            os.makedirs(d, exist_ok=True)


# Global singleton — import this everywhere
cfg = AgentConfig()
cfg.setup_dirs()
