# tests/test_coverage_boost.py
# Run alongside test_agent.py to push total coverage above 80%.
#
#   pytest tests/ --cov=agent --cov=sandbox --cov-report=term-missing

import time
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from agent.logger import log
from agent.tracer import ExecutionTrace, TraceStep
from agent.cache import QueryCache
from agent.tools import (
    load_and_inspect_data,
    execute_python_code,
    get_column_statistics,
    get_correlation_analysis,
    _df_store,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_DF = pd.DataFrame({
    "age":      [21, 25, 30, 22, 35],
    "salary":   [50000, 65000, 80000, 52000, 95000],
    "city":     ["NY", "LA", "SF", "NY", "LA"],
    "target":   [0, 1, 1, 0, 1],
})


def _load_sample():
    """Inject SAMPLE_DF directly into the store so tools can use it."""
    _df_store["current"] = SAMPLE_DF.copy()


# ─────────────────────────────────────────────────────────────────────────────
# tools.py — load_and_inspect_data
# ─────────────────────────────────────────────────────────────────────────────

def test_load_and_inspect_valid_csv(tmp_path):
    csv_file = tmp_path / "data.csv"
    SAMPLE_DF.to_csv(csv_file, index=False)

    result = load_and_inspect_data.invoke({"file_path": str(csv_file)})

    assert "Dataset loaded successfully" in result
    assert "5" in result          # 5 rows
    assert "salary" in result
    assert "city" in result


def test_load_and_inspect_populates_df_store(tmp_path):
    csv_file = tmp_path / "data.csv"
    SAMPLE_DF.to_csv(csv_file, index=False)
    _df_store.pop("current", None)

    load_and_inspect_data.invoke({"file_path": str(csv_file)})

    assert "current" in _df_store
    assert list(_df_store["current"].columns) == list(SAMPLE_DF.columns)


def test_load_and_inspect_file_not_found():
    result = load_and_inspect_data.invoke(
        {"file_path": "/nonexistent/path/data.csv"})
    assert "Error" in result
    assert "not found" in result.lower()


def test_load_and_inspect_shows_value_counts_for_low_cardinality_column(tmp_path):
    """Columns with ≤10 unique values should trigger value_counts output."""
    csv_file = tmp_path / "data.csv"
    SAMPLE_DF.to_csv(csv_file, index=False)

    result = load_and_inspect_data.invoke({"file_path": str(csv_file)})

    # 'target' has 2 unique int values — should appear as a potential target
    assert "target" in result


# ─────────────────────────────────────────────────────────────────────────────
# tools.py — get_column_statistics (numeric)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_column_statistics_numeric_columns():
    _load_sample()
    result = get_column_statistics.invoke({"column_name": "salary"})

    assert "Statistics for column: 'salary'" in result
    assert "Outlier" in result
    assert "Skewness" in result
    assert "Kurtosis" in result


def test_get_column_statistics_shows_iqr_bounds():
    _load_sample()
    result = get_column_statistics.invoke({"column_name": "age"})

    assert "Lower bound" in result
    assert "Upper bound" in result


def test_get_column_statistics_categorical_column():
    _load_sample()
    result = get_column_statistics.invoke({"column_name": "city"})

    assert "Unique values" in result
    assert "NY" in result or "LA" in result or "SF" in result


def test_get_column_statistics_missing_column_name():
    _load_sample()
    result = get_column_statistics.invoke({"column_name": "nonexistent_col"})

    assert "not found" in result.lower()
    assert "salary" in result   # lists available columns


def test_get_column_statistics_no_df_loaded():
    _df_store.pop("current", None)
    result = get_column_statistics.invoke({"column_name": "salary"})

    assert "No dataframe loaded" in result


def test_get_column_statistics_null_values_reported():
    _df_store["current"] = pd.DataFrame({
        "score": [10, None, 30, None, 50],
    })
    result = get_column_statistics.invoke({"column_name": "score"})

    assert "Null count: 2" in result


# ─────────────────────────────────────────────────────────────────────────────
# tools.py — get_correlation_analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_get_correlation_analysis_returns_matrix():
    _load_sample()
    result = get_correlation_analysis.invoke({})

    assert "Correlation Analysis" in result
    assert "age" in result
    assert "salary" in result


def test_get_correlation_analysis_labels_strength():
    _load_sample()
    result = get_correlation_analysis.invoke({})

    # At least one of these strength labels must appear
    assert any(label in result for label in ["strong", "moderate", "weak"])


def test_get_correlation_analysis_not_enough_columns():
    _df_store["current"] = pd.DataFrame({"only_one": [1, 2, 3]})
    result = get_correlation_analysis.invoke({})

    assert "Not enough numeric columns" in result


def test_get_correlation_analysis_no_df_loaded():
    _df_store.pop("current", None)
    result = get_correlation_analysis.invoke({})

    assert "No dataframe loaded" in result


def test_get_correlation_analysis_ignores_string_columns():
    """String columns must not cause corr() to crash."""
    _load_sample()   # SAMPLE_DF has 'city' (string)
    result = get_correlation_analysis.invoke({})

    assert "Execution Error" not in result
    assert "Correlation Analysis" in result


# ─────────────────────────────────────────────────────────────────────────────
# tools.py — execute_python_code (no-df path)
# ─────────────────────────────────────────────────────────────────────────────

def test_execute_python_code_no_df_loaded():
    _df_store.pop("current", None)
    result = execute_python_code.invoke({"code": "print(df.head())"})

    assert "No dataframe loaded" in result


def test_execute_python_code_no_print_output_warning():
    _load_sample()
    # Code that runs but produces no print output
    result = execute_python_code.invoke({"code": "x = 1 + 1"})

    assert "forgot to print" in result.lower() or "No print" in result


# ─────────────────────────────────────────────────────────────────────────────
# tracer.py — TraceStep dataclass methods
# ─────────────────────────────────────────────────────────────────────────────

def test_trace_step_emoji_success():
    step = TraceStep(1, "execute_python_code",
                     "print(df.mean())", success=True)
    assert step.status_emoji() == "✅"


def test_trace_step_emoji_failure():
    step = TraceStep(1, "execute_python_code", "print(x)", success=False)
    assert step.status_emoji() == "❌"


def test_trace_step_emoji_plot():
    step = TraceStep(1, "execute_python_code", "plt.plot(df['age'])",
                     success=True, plot_generated=True)
    assert step.status_emoji() == "📊"


def test_trace_step_emoji_retry():
    step = TraceStep(2, "execute_python_code", "retry code",
                     success=True, is_retry=True)
    assert step.status_emoji() == "🔄"


def test_trace_step_short_label_known_tool():
    step = TraceStep(1, "load_and_inspect_data", "path.csv")
    label = step.short_label()
    assert "load_data" in label
    assert "Step 1" in label


def test_trace_step_short_label_execute():
    step = TraceStep(3, "execute_python_code", "code")
    assert "run_code" in step.short_label()
    assert "Step 3" in step.short_label()


def test_trace_step_short_label_unknown_tool():
    step = TraceStep(2, "some_unknown_tool", "input")
    label = step.short_label()
    assert "some_unknown_tool" in label


def test_trace_step_short_label_shows_retry_tag():
    step = TraceStep(2, "execute_python_code", "retry", is_retry=True)
    assert "retry" in step.short_label().lower()


def test_trace_step_short_label_col_stats():
    step = TraceStep(1, "get_column_statistics", "salary")
    assert "col_stats" in step.short_label()


def test_trace_step_short_label_correlations():
    step = TraceStep(1, "get_correlation_analysis", "")
    assert "correlations" in step.short_label()


# ─────────────────────────────────────────────────────────────────────────────
# logger.py — previously uncovered methods
# ─────────────────────────────────────────────────────────────────────────────

def test_logger_session_start_and_end():
    session_id = f"sess-{time.time_ns()}"
    log.session_start(session_id, csv_path="data.csv")
    log.session_end(session_id, total_turns=5)

    events = log.read_session_events(session_id)
    event_types = [e["event"] for e in events]

    assert "session_start" in event_types
    assert "session_end" in event_types

    start = next(e for e in events if e["event"] == "session_start")
    assert start["csv_path"] == "data.csv"

    end = next(e for e in events if e["event"] == "session_end")
    assert end["total_turns"] == 5


def test_logger_agent_step():
    session_id = f"step-{time.time_ns()}"
    log.agent_step(iteration=4, tool_calls=[
                   "execute_python_code"], session_id=session_id)

    events = log.read_session_events(session_id)
    step_event = next(
        (e for e in events if e.get("event") == "agent_step"), None)

    assert step_event is not None
    assert step_event["iteration"] == 4
    assert "execute_python_code" in step_event["tool_calls"]


def test_logger_performance_event():
    session_id = f"perf-{time.time_ns()}"
    log.performance(session_id, elapsed_s=2.3, iterations=5,
                    plots_generated=2, error_count=1)

    events = log.read_session_events(session_id)
    perf = next((e for e in events if e.get("event") == "performance"), None)

    assert perf is not None
    assert perf["iterations"] == 5
    assert perf["plots_generated"] == 2
    assert perf["error_count"] == 1
    assert perf["elapsed_s"] == 2.3


def test_logger_read_session_events_filters_by_session():
    """read_session_events must only return events for the requested session."""
    sid_a = f"session-A-{time.time_ns()}"
    sid_b = f"session-B-{time.time_ns()}"

    log.tool_call("execute_python_code", session_id=sid_a)
    log.tool_call("load_and_inspect_data", session_id=sid_b)

    events_a = log.read_session_events(sid_a)
    events_b = log.read_session_events(sid_b)

    assert all(e["session_id"] == sid_a for e in events_a)
    assert all(e["session_id"] == sid_b for e in events_b)


def test_logger_failed_tool_result():
    session_id = f"fail-{time.time_ns()}"
    log.tool_result("execute_python_code", success=False,
                    duration_ms=300, session_id=session_id,
                    output_preview="Execution Error: NameError")

    events = log.read_session_events(session_id)
    result_event = next(e for e in events if e.get("event") == "tool_result")

    assert result_event["success"] is False
    assert "NameError" in result_event["output_preview"]


# ─────────────────────────────────────────────────────────────────────────────
# cache.py — previously uncovered paths
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_clear_all(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.set("a.csv", "Q1", "A1", [])
    cache.set("b.csv", "Q2", "A2", [])
    cache.set("c.csv", "Q3", "A3", [])

    removed = cache.clear_all()

    assert removed == 3
    assert cache.stats()["total_entries"] == 0


def test_cache_filters_missing_plot_paths(tmp_path):
    """
    Cached plot paths that no longer exist on disk must be silently
    removed when the entry is retrieved.
    """
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.set(
        csv_path="data.csv",
        question="Show histogram",
        response_text="Here is the histogram",
        plot_paths=["/tmp/nonexistent_plot_abc123.png"],
    )

    result = cache.get("data.csv", "Show histogram")

    assert result is not None
    assert result["plot_paths"] == []   # ghost path removed


def test_cache_returns_none_when_disabled(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.enabled = False
    cache.set("data.csv", "Q", "A", [])

    assert cache.get("data.csv", "Q") is None


def test_cache_stats_counts_expired_entries(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=1)
    cache.set("a.csv", "Q1", "A1", [])
    time.sleep(2)

    stats = cache.stats()

    assert stats["expired_entries"] >= 1
    assert stats["live_entries"] == 0


def test_cache_set_skips_when_disabled(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.enabled = False
    cache.set("data.csv", "Q", "A", [])

    assert cache.stats()["total_entries"] == 0
