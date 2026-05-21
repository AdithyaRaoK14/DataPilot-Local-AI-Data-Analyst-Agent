# agent/sql_tools.py
"""
SQL database support for the data analyst agent.

Supports SQLite natively (stdlib only — no extra driver needed).
Results are automatically loaded into the shared _df_store so every
downstream tool (execute_python_code, get_column_statistics, …) works
seamlessly after a SQL load, exactly like a CSV load does.

Safety model
------------
* Only SELECT queries are permitted; all write/DDL verbs are blocked.
* Allowed PRAGMA keys are whitelisted in cfg.sql_allowed_pragmas.
* Row count is capped at cfg.sql_row_limit (default 100 000).

Tools exported
--------------
    list_database_tables   — connect + inspect schema + row counts
    load_table_from_db     — load a full table into df
    execute_sql_query      — run an arbitrary SELECT and load result into df
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from agent.tools import _df_store
from agent.config import cfg
from agent.logger import log

# Per-process connection cache: db_path → sqlite3.Connection
_db_store: dict = {}

# ── Internal helpers ──────────────────────────────────────────────────────────

_WRITE_VERBS = frozenset(
    ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
     "TRUNCATE", "REPLACE", "ATTACH", "DETACH", "VACUUM"]
)


def _get_connection(db_path: str) -> sqlite3.Connection:
    """Return a cached connection, opening it on first access."""
    if db_path not in _db_store:
        p = Path(db_path)
        if not p.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        conn = sqlite3.connect(str(p), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Read-only URI is safest; fall back to regular open if unsupported
        _db_store[db_path] = conn
        _db_store["current_path"] = db_path
    return _db_store[db_path]


def _is_write_query(query: str) -> bool:
    first_word = query.strip().split()[0].upper() if query.strip() else ""
    return first_word in _WRITE_VERBS


def _is_unsafe_pragma(query: str) -> bool:
    """Block PRAGMA keys that could leak metadata or change DB state."""
    upper = query.strip().upper()
    if not upper.startswith("PRAGMA"):
        return False
    # Extract the pragma key: "PRAGMA key_name" or "PRAGMA key_name = ..."
    parts = upper[6:].strip().split("=")[0].split("(")[0].strip()
    return parts not in {p.upper() for p in cfg.sql_allowed_pragmas}


def _summarise_df(df: pd.DataFrame, source_label: str) -> str:
    """Build the standard load summary returned by all three SQL tools."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
    missing = {k: int(v) for k, v in df.isnull().sum().items() if v > 0}

    parts = [
        f"Source: {source_label}",
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
        f"Columns: {list(df.columns)}",
        f"Numeric columns: {numeric_cols}",
        f"Categorical columns: {categorical_cols}",
        f"Missing values: {missing if missing else 'None'}",
        f"\nNumerical summary:\n{df.describe().round(3).to_string()}",
    ]
    for col in categorical_cols[:2]:
        if df[col].nunique() <= 15:
            vc = df[col].value_counts()
            parts.append(f"\nValue counts for '{col}':\n{vc.head(10).to_string()}")
    return "\n".join(parts)


# ── Public LangChain tools ────────────────────────────────────────────────────

@tool
def list_database_tables(db_path: str) -> str:
    """
    Connect to a SQLite database and list every table with its schema,
    column types, constraints, and row count. Always call this first when
    working with a .sqlite or .db file — it both connects and inspects.
    """
    try:
        conn = _get_connection(db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        )
        tables = [row[0] for row in cur.fetchall()]

        if not tables:
            return f"Connected to {db_path}, but no user tables were found."

        parts = [
            f"Database: {db_path}",
            f"Tables ({len(tables)} found):\n",
        ]

        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM '{table}';")
            row_count = cur.fetchone()[0]

            cur.execute(f"PRAGMA table_info('{table}');")
            cols = cur.fetchall()

            parts.append(f"  📋 {table}  ({row_count:,} rows)")
            for c in cols:
                cid, name, dtype, notnull, dflt, pk = (
                    c[0], c[1], c[2], c[3], c[4], c[5]
                )
                flags = []
                if pk:
                    flags.append("PK")
                if notnull:
                    flags.append("NOT NULL")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                parts.append(f"      {name}  {dtype}{flag_str}")
            parts.append("")

        log.tool_result("list_database_tables", success=True, duration_ms=0)
        return "\n".join(parts)

    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        log.error("list_database_tables", exc=exc)
        return f"Error inspecting database: {exc}"


@tool
def load_table_from_db(db_path: str, table_name: str) -> str:
    """
    Load a specific table from a SQLite database into the analysis dataframe.
    After loading, all other tools (execute_python_code, get_column_statistics,
    etc.) operate on this table exactly like a CSV dataset.
    Row count is capped at the configured limit (default 100 000).
    """
    try:
        conn = _get_connection(db_path)
        query = f"SELECT * FROM '{table_name}' LIMIT {cfg.sql_row_limit};"
        df = pd.read_sql_query(query, conn)

        _df_store["current"] = df
        _df_store["path"] = f"{db_path}::{table_name}"
        _db_store["current_table"] = table_name

        log.tool_result("load_table_from_db", success=True, duration_ms=0,
                        output_preview=f"{df.shape[0]} rows × {df.shape[1]} cols")
        return (
            f"Table '{table_name}' loaded successfully.\n"
            + _summarise_df(df, source_label=f"{db_path}::{table_name}")
        )

    except FileNotFoundError as exc:
        return str(exc)
    except Exception as exc:
        log.error("load_table_from_db", exc=exc)
        return f"Error loading table '{table_name}': {exc}"


@tool
def execute_sql_query(query: str) -> str:
    """
    Execute a SELECT query against the currently connected SQLite database.
    Results are automatically loaded into 'df' for further Python analysis.
    Only SELECT statements are allowed — INSERT, UPDATE, DELETE, DROP, and
    other write/DDL operations are permanently blocked for safety.
    Call list_database_tables first to connect to a database.
    """
    if "current_path" not in _db_store:
        return (
            "No database connected. "
            "Call list_database_tables(db_path) first to connect."
        )

    if _is_write_query(query):
        return (
            "Safety error: only SELECT queries are permitted. "
            "Write and DDL operations are blocked."
        )

    if _is_unsafe_pragma(query):
        return (
            "Safety error: this PRAGMA is not on the allow-list. "
            f"Allowed: {cfg.sql_allowed_pragmas}"
        )

    try:
        conn = _db_store[_db_store["current_path"]]
        df = pd.read_sql_query(query, conn)

        if df.empty:
            return "Query executed successfully — no rows returned."

        # Cap silently and report the cap
        capped = ""
        if len(df) > cfg.sql_row_limit:
            df = df.head(cfg.sql_row_limit)
            capped = f"\n(Result capped at {cfg.sql_row_limit:,} rows)"

        _df_store["current"] = df
        _df_store["path"] = f"sql::{query[:80]}"

        preview_rows = min(20, len(df))
        result = (
            f"Query returned {len(df):,} rows × {df.shape[1]} columns.{capped}\n\n"
            f"{df.head(preview_rows).to_string(index=False)}"
        )
        if len(df) > preview_rows:
            result += f"\n... ({len(df) - preview_rows} more rows — df is loaded)"

        log.tool_result("execute_sql_query", success=True, duration_ms=0,
                        output_preview=result[:200])
        return result

    except Exception as exc:
        log.error("execute_sql_query", exc=exc)
        return f"SQL Error: {exc}"
