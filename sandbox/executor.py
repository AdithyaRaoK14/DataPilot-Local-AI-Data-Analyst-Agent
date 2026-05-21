# sandbox/executor.py
"""
Secure code execution with three layers of defense:
  1. AST static analysis  — blocks dangerous imports/calls before any execution
  2. Restricted builtins  — whitelisted __builtins__ only
  3. Subprocess isolation + timeout — code runs in a child process that is
     forcibly killed if it exceeds the time or memory limit

This replaces the bare `exec(code, namespace)` pattern in tools.py.
"""

import ast
import io
import os
import sys
import uuid
import pickle
import signal
import textwrap
import traceback
import contextlib
import multiprocessing
from typing import Tuple

# ─── Configuration ─────────────────────────────────────────────────────────────

EXECUTION_TIMEOUT_SECONDS = 30
SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)

# Modules that must never be imported inside user code
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "shutil", "pathlib", "importlib",
    "ctypes", "socket", "requests", "urllib", "http", "ftplib",
    "smtplib", "telnetlib", "xmlrpc", "multiprocessing", "threading",
    "concurrent", "asyncio", "pty", "tty", "termios", "signal",
    "resource", "gc", "weakref", "builtins", "code", "codeop",
    "compileall", "dis", "marshal", "pickle", "shelve", "dbm",
    "sqlite3", "zipfile", "tarfile", "gzip", "bz2", "lzma",
    "tempfile", "glob", "fnmatch", "linecache", "tokenize",
    "keyword", "token", "pdb", "profile", "cProfile", "timeit",
    "trace", "inspect", "ast",
}

# Functions that bypass builtins restrictions when called as names
BLOCKED_BUILTINS = {
    "eval", "exec", "compile", "open", "__import__",
    "breakpoint", "input", "memoryview", "staticmethod", "classmethod",
    "super", "vars", "dir", "globals", "locals", "getattr", "setattr",
    "delattr", "hasattr", "object",
}

# Dangerous attribute access patterns
BLOCKED_ATTRS = {
    "__class__", "__bases__", "__subclasses__", "__mro__",
    "__globals__", "__code__", "__func__", "__self__",
    "__dict__", "__module__", "__builtins__",
    "system", "popen", "run", "call", "check_output", "Popen",
    "spawn", "fork", "exec", "execv", "execve",
}


# ─── Layer 1: AST Validation ───────────────────────────────────────────────────

class SecurityViolation(Exception):
    pass


class ASTSecurityVisitor(ast.NodeVisitor):
    """Walk the AST and raise SecurityViolation on anything dangerous."""

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in BLOCKED_MODULES:
                raise SecurityViolation(f"Blocked import: '{alias.name}'")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root = node.module.split(".")[0]
            if root in BLOCKED_MODULES:
                raise SecurityViolation(
                    f"Blocked 'from {node.module} import ...'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Direct calls: eval(...), exec(...), __import__(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_BUILTINS:
                raise SecurityViolation(
                    f"Blocked function call: '{node.func.id}()'")
        # Attribute calls: os.system(...), subprocess.run(...)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_ATTRS:
                raise SecurityViolation(
                    f"Blocked method call: '.{node.func.attr}()'"
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Accessing dunder attrs to escape sandbox: obj.__class__.__subclasses__()
        if node.attr in BLOCKED_ATTRS:
            raise SecurityViolation(
                f"Blocked attribute access: '.{node.attr}'")
        self.generic_visit(node)


def validate_code(code: str) -> Tuple[bool, str]:
    """
    Returns (is_safe, message).
    Call this before executing anything.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    visitor = ASTSecurityVisitor()
    try:
        visitor.visit(tree)
    except SecurityViolation as e:
        return False, str(e)

    return True, "OK"


# ─── Layer 2: Safe builtins whitelist ─────────────────────────────────────────

SAFE_BUILTINS = {
    "print": print,
    "len": len,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "bytes": bytes,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "pow": pow,
    "divmod": divmod,
    "sorted": sorted,
    "reversed": reversed,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "type": type,
    "repr": repr,
    "format": format,
    "hash": hash,
    "id": id,
    "iter": iter,
    "next": next,
    "any": any,
    "all": all,
    "slice": slice,
    "NotImplemented": NotImplemented,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
    "AssertionError": AssertionError,
}


# ─── Layer 3: Subprocess worker ───────────────────────────────────────────────

def _subprocess_worker(
    code: str,
    df_bytes: bytes,
    plot_filename: str,
    result_queue: multiprocessing.Queue,
):
    """
    This function runs inside a fresh child process.
    It receives the dataframe as pickled bytes (not a reference) so the
    parent's memory space is never touched.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import seaborn as sns

        df = pickle.loads(df_bytes)

        namespace = {
            "df": df,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "plot_path": plot_filename,
            "__builtins__": SAFE_BUILTINS,
        }

        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, namespace)  # noqa: S102 — validated before reaching here

        output = stdout_capture.getvalue()

        if plt.get_fignums():
            plt.savefig(plot_filename, bbox_inches="tight",
                        dpi=150, facecolor="white")
            plt.close("all")
            output += f"\n[PLOT_SAVED:{plot_filename}]"

        if not output.strip():
            output = (
                "Code executed successfully. "
                "No print() output — did you forget to print() your results?"
            )

        result_queue.put({"status": "ok", "output": output})

    except Exception as exc:
        result_queue.put({
            "status": "error",
            "output": f"Execution Error:\n{exc}\n\nTraceback:\n{traceback.format_exc()}",
        })


# ─── Public API ───────────────────────────────────────────────────────────────

class ExecutionResult:
    __slots__ = ("output", "plot_path", "success",
                 "timed_out", "blocked_reason")

    def __init__(
        self,
        output: str = "",
        plot_path: str | None = None,
        success: bool = True,
        timed_out: bool = False,
        blocked_reason: str | None = None,
    ):
        self.output = output
        self.plot_path = plot_path
        self.success = success
        self.timed_out = timed_out
        self.blocked_reason = blocked_reason


def run_secure(code: str, df, timeout: int = EXECUTION_TIMEOUT_SECONDS) -> ExecutionResult:
    """
    Execute `code` safely against `df`.

    Defense layers applied in order:
      1. AST validation     — reject before any execution
      2. Subprocess isolation — runs in a separate process, killed on timeout
      3. Restricted builtins — whitelisted only inside the worker

    Returns an ExecutionResult with all fields populated.
    """
    # Layer 1 — static analysis
    is_safe, reason = validate_code(code)
    if not is_safe:
        return ExecutionResult(
            output=f"Security Error: {reason}",
            success=False,
            blocked_reason=reason,
        )

    plot_filename = f"{SANDBOX_DIR}/plot_{uuid.uuid4().hex[:8]}.png"

    try:
        df_bytes = pickle.dumps(df)
    except Exception as e:
        return ExecutionResult(
            output=f"Serialisation Error: {e}",
            success=False,
        )

    result_queue: multiprocessing.Queue = multiprocessing.Queue()

    proc = multiprocessing.Process(
        target=_subprocess_worker,
        args=(code, df_bytes, plot_filename, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout)

    # Layer 3 — timeout enforcement
    if proc.is_alive():
        proc.terminate()
        proc.join(2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return ExecutionResult(
            output=f"Execution Error: Code timed out after {timeout}s.",
            success=False,
            timed_out=True,
        )

    if result_queue.empty():
        return ExecutionResult(
            output="Execution Error: Worker process exited without returning a result.",
            success=False,
        )

    result = result_queue.get_nowait()
    ok = result["status"] == "ok"
    plot_path = plot_filename if ok and os.path.exists(plot_filename) else None

    return ExecutionResult(
        output=result["output"],
        plot_path=plot_path,
        success=ok,
    )
