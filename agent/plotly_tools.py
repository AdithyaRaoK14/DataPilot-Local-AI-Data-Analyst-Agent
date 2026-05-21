# agent/plotly_tools.py
"""
Plotly interactive dashboard generation tool.

Produces self-contained HTML files (saved to sandbox/) that the Streamlit UI
can render inside an <iframe> with a download button alongside.

Namespace available inside user code
--------------------------------------
    df          — current dataframe (copy)
    pd          — pandas
    np          — numpy
    px          — plotly.express
    go          — plotly.graph_objects
    fig_path    — pre-set output path ending in .html  ← use this to save

Rules for agent code
---------------------
    - Save with:  fig.write_html(fig_path)
    - Never call: fig.show()
    - Never redefine fig_path
    - Never import anything — all libraries are pre-injected
"""

import contextlib
import io
import os
import traceback
import uuid

from langchain_core.tools import tool

from agent.tools import get_df
from agent.logger import log

SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    "round": round, "min": min, "max": max, "sum": sum, "abs": abs,
    "sorted": sorted, "reversed": reversed,
    "isinstance": isinstance, "type": type,
    "True": True, "False": False, "None": None,
    "ValueError": ValueError, "TypeError": TypeError,
}


def _strip_forbidden(code: str) -> str:
    """Remove import lines and fig_path reassignments."""
    cleaned = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith("fig_path"):
            # block  fig_path = ... and fig_path=... assignments
            if "=" in stripped and not stripped.startswith("fig_path =="):
                continue
        cleaned.append(line)
    return "\n".join(cleaned)


@tool
def execute_plotly_code(code: str) -> str:
    """
    Execute Plotly code to generate an interactive HTML dashboard.
    Available in the execution namespace: df, pd, np, px, go, fig_path.
    Save the figure using: fig.write_html(fig_path)
    Never redefine fig_path. Never call fig.show().
    Use plotly.express (px) for quick charts or plotly.graph_objects (go) for
    full control. Supports subplots via plotly.subplots.make_subplots imported
    directly as: from plotly.subplots import make_subplots — but this is the
    ONLY allowed import inside user code.
    Returns [PLOTLY_SAVED:<path>] on success so the UI can render the HTML.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.subplots as _ps
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        return f"Error: {exc}. Run: pip install plotly"

    try:
        df = get_df()
    except ValueError as exc:
        return str(exc)

    fig_filename = f"{SANDBOX_DIR}/dashboard_{uuid.uuid4().hex[:8]}.html"
    code = _strip_forbidden(code)

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "make_subplots": _ps.make_subplots,
        "fig_path": fig_filename,
        "__builtins__": _SAFE_BUILTINS,
    }

    stdout_cap = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_cap):
            exec(code, namespace)  # noqa: S102

        output = stdout_cap.getvalue()

        # Auto-save any figure left in namespace if the code didn't call write_html
        if not os.path.exists(fig_filename):
            fig = namespace.get("fig")
            if fig is not None:
                fig.write_html(fig_filename)

        if os.path.exists(fig_filename):
            size_kb = round(os.path.getsize(fig_filename) / 1024, 1)
            output += f"\n[PLOTLY_SAVED:{fig_filename}]"
            log.tool_result(
                "execute_plotly_code", success=True, duration_ms=0,
                output_preview=f"HTML {size_kb} KB → {fig_filename}",
            )
        else:
            output += (
                "\nNote: No HTML file was saved. "
                "Make sure to call fig.write_html(fig_path) at the end of your code."
            )

        return output.strip() or "Plotly code executed. (no print output)"

    except Exception as exc:
        log.error("execute_plotly_code", exc=exc)
        return (
            f"Plotly Execution Error:\n{exc}"
            f"\n\nTraceback:\n{traceback.format_exc()}"
        )
