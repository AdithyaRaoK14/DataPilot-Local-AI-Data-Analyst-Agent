# tests/test_agent.py

import csv
import json
import time
import tempfile
import multiprocessing
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.cache import QueryCache
from agent.logger import log, _JSONL_FILE
from agent.prompts import build_data_context
from agent.tracer import ExecutionTrace
from agent.graph import (
    error_handler_node,
    visual_enforcer_node,
    track_plots_node,
    should_continue,
    MAX_ITERATIONS,
    _ENFORCEMENT_MARKER,
    agent,
)
from sandbox.executor import validate_code, run_secure

# macOS / multiprocessing safety
multiprocessing.set_start_method("spawn", force=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared test dataframes
# ─────────────────────────────────────────────────────────────────────────────

TEST_DF = pd.DataFrame({
    "age":    [21, 25, 30, 22],
    "salary": [50000, 65000, 80000, 52000],
    "target": [0, 1, 1, 0],
})

# Edge case DataFrames
EMPTY_DF = pd.DataFrame({"age": pd.Series([], dtype=float),
                         "salary": pd.Series([], dtype=float),
                         "target": pd.Series([], dtype=int)})

SINGLE_ROW_DF = pd.DataFrame({"age": [25], "salary": [50000], "target": [1]})

NULL_COLUMN_DF = pd.DataFrame({
    "age":    [21, 25, None, 22],
    "salary": [50000, None, 80000, 52000],
    "target": [0, 1, 1, 0],
})

NO_NUMERIC_DF = pd.DataFrame({
    "name":  ["Alice", "Bob", "Carol"],
    "city":  ["NY", "LA", "SF"],
    "label": ["yes", "no", "yes"],
})

ALL_NULL_DF = pd.DataFrame({
    "age":    [None, None, None],
    "salary": [None, None, None],
})


# ─────────────────────────────────────────────────────────────────────────────
# Cache Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_set_and_get(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.set(
        csv_path="test.csv",
        question="What is the mean age?",
        response_text="Mean age is 24.5",
        plot_paths=[],
        iteration_count=2,
    )
    result = cache.get("test.csv", "What is the mean age?")
    assert result is not None
    assert result["response_text"] == "Mean age is 24.5"
    assert result["iteration_count"] == 2


def test_cache_expiration(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=1)
    cache.set(csv_path="test.csv", question="Q",
              response_text="A", plot_paths=[])
    time.sleep(2)
    assert cache.get("test.csv", "Q") is None


def test_cache_invalidate(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.set(csv_path="test.csv", question="Delete me",
              response_text="A", plot_paths=[])
    cache.invalidate("test.csv", "Delete me")
    assert cache.get("test.csv", "Delete me") is None


def test_cache_stats(tmp_path):
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    cache.set(csv_path="a.csv", question="Q1",
              response_text="A1", plot_paths=[])
    stats = cache.stats()
    assert "total_entries" in stats
    assert stats["total_entries"] >= 1


def test_cache_key_changes_when_file_changes(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2")
    cache = QueryCache(cache_dir=str(tmp_path), ttl=60)
    key1 = cache._key(str(csv_file), "What is the mean?")
    time.sleep(1)
    csv_file.write_text("a,b\n3,4")
    key2 = cache._key(str(csv_file), "What is the mean?")
    assert key1 != key2


def test_cache_miss_after_file_modified(tmp_path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2")
    cache = QueryCache(cache_dir=str(tmp_path / "cache"), ttl=60)
    cache.set(csv_path=str(csv_file), question="What is the mean?",
              response_text="Mean is 1.5", plot_paths=[])
    assert cache.get(str(csv_file), "What is the mean?") is not None
    time.sleep(1)
    csv_file.write_text("a,b\n9,10")
    assert cache.get(str(csv_file), "What is the mean?") is None


# ─────────────────────────────────────────────────────────────────────────────
# Security Validation Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_blocked_import():
    safe, reason = validate_code("import os\nprint('bad')")
    assert safe is False
    assert "Blocked import" in reason


def test_blocked_builtin():
    safe, reason = validate_code('eval("2+2")')
    assert safe is False
    assert "Blocked function call" in reason


def test_blocked_attribute_access():
    safe, reason = validate_code("x.__class__")
    assert safe is False
    assert "Blocked attribute access" in reason


def test_blocked_subprocess_call():
    safe, reason = validate_code("import subprocess\nsubprocess.run(['ls'])")
    assert safe is False


def test_syntax_error_detection():
    safe, reason = validate_code("for")
    assert safe is False
    assert "Syntax error" in reason


# ─────────────────────────────────────────────────────────────────────────────
# Secure Execution — Happy Path
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_code_execution():
    result = run_secure("print(df['age'].mean())", TEST_DF)
    assert result.success is True
    assert "24.5" in result.output


def test_plot_generation():
    result = run_secure("plt.plot(df['age'])\nprint('plot done')", TEST_DF)
    assert result.success is True
    assert result.plot_path is not None


def test_timeout_enforcement():
    result = run_secure("while True:\n    pass", TEST_DF, timeout=2)
    assert result.success is False
    assert result.timed_out is True


def test_dataframe_operations():
    result = run_secure(
        "print(df.groupby('target')['salary'].mean())", TEST_DF)
    assert result.success is True
    assert "72500" in result.output


def test_execution_error_capture():
    result = run_secure("print(x)", TEST_DF)
    assert result.success is False
    assert "Execution Error" in result.output


# ─────────────────────────────────────────────────────────────────────────────
# Secure Execution — Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

def test_run_secure_with_empty_dataframe():
    """Agent must not crash on an empty dataset — just report 0 rows."""
    result = run_secure("print(len(df))", EMPTY_DF)
    assert result.success is True
    assert "0" in result.output


def test_run_secure_with_single_row():
    """Stats on a 1-row DataFrame must not raise errors."""
    result = run_secure("print(df['salary'].mean())", SINGLE_ROW_DF)
    assert result.success is True
    assert "50000" in result.output


def test_run_secure_with_null_columns():
    """Agent code must handle NaN values without crashing."""
    result = run_secure("print(df.isnull().sum().sum())", NULL_COLUMN_DF)
    assert result.success is True
    # NULL_COLUMN_DF has exactly 2 null values
    assert "2" in result.output


def test_run_secure_with_no_numeric_columns():
    """Selecting numeric cols on an all-string DataFrame must return an empty result gracefully."""
    code = "print(df.select_dtypes(include='number').columns.tolist())"
    result = run_secure(code, NO_NUMERIC_DF)
    assert result.success is True
    assert "[]" in result.output


def test_run_secure_with_all_null_column():
    """describe() on an all-null column should not raise."""
    result = run_secure("print(df.describe())", ALL_NULL_DF)
    assert result.success is True


def test_run_secure_correlation_on_two_columns():
    """Correlation on the minimum viable numeric DataFrame (2 cols) must succeed."""
    result = run_secure(
        "print(df[['age','salary']].corr().round(2))", TEST_DF
    )
    assert result.success is True


# ─────────────────────────────────────────────────────────────────────────────
# Trace Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_execution_trace():
    trace = ExecutionTrace(question="Analyze salary")
    trace.add_step(tool_name="execute_python_code",
                   tool_input_preview="print(df.mean())",
                   tool_output_preview="salary 61750",
                   success=True)
    trace.add_step(tool_name="execute_python_code",
                   tool_input_preview="retry",
                   tool_output_preview="fixed",
                   success=True)
    summary = trace.summary()
    assert summary["total_steps"] == 2
    assert summary["retries"] == 1


def test_trace_plot_tracking():
    trace = ExecutionTrace(question="Plot salary")
    trace.add_step(tool_name="execute_python_code",
                   tool_input_preview="plt.plot(df['salary'])",
                   tool_output_preview="done",
                   plot_generated=True)
    summary = trace.summary()
    assert summary["plots_generated"] == 1


def test_trace_failed_step_counted():
    trace = ExecutionTrace(question="Bad code")
    trace.add_step(tool_name="execute_python_code",
                   tool_input_preview="print(undefined_var)",
                   tool_output_preview="Execution Error: name 'undefined_var' is not defined",
                   success=False)
    summary = trace.summary()
    assert summary["failed_steps"] == 1
    assert summary["successful_steps"] == 0


def test_trace_summary_tools_used_order():
    """tools_used list must preserve insertion order with no duplicates."""
    trace = ExecutionTrace(question="Multi-tool")
    trace.add_step("load_and_inspect_data", "path.csv", "loaded")
    trace.add_step("execute_python_code", "print(df.head())", "ok")
    trace.add_step("execute_python_code", "print(df.mean())", "ok")
    summary = trace.summary()
    assert summary["tools_used"] == [
        "load_and_inspect_data", "execute_python_code"]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_data_context():
    context = build_data_context({
        "csv_path": "data.csv",
        "rows": 100,
        "cols": 3,
        "columns": ["a", "b", "c"],
        "numeric_cols": ["a", "b"],
        "categorical_cols": ["c"],
        "missing_summary": "None",
    })
    assert "100" in context
    assert "data.csv" in context
    assert "numeric columns" in context.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Logger Tests  (fixed — assert True removed)
# ─────────────────────────────────────────────────────────────────────────────

def test_logger_writes_tool_call_event():
    """tool_call must write a JSON-lines record with the correct fields."""
    session_id = f"test-session-{time.time_ns()}"
    log.tool_call("execute_python_code",
                  code="print(df.head())", session_id=session_id)

    events = log.read_session_events(session_id)
    assert len(events) >= 1

    call_event = next(
        (e for e in events if e.get("event") == "tool_call"), None)
    assert call_event is not None
    assert call_event["tool"] == "execute_python_code"
    assert call_event["session_id"] == session_id


def test_logger_writes_tool_result_event():
    """tool_result must persist success flag and duration."""
    session_id = f"test-result-{time.time_ns()}"
    log.tool_result("execute_python_code", success=True,
                    duration_ms=120, session_id=session_id, output_preview="ok")

    events = log.read_session_events(session_id)
    result_event = next(
        (e for e in events if e.get("event") == "tool_result"), None)
    assert result_event is not None
    assert result_event["success"] is True
    assert result_event["duration_ms"] == 120.0


def test_logger_writes_error_event():
    session_id = f"test-error-{time.time_ns()}"
    log.error("agent_node", exc=ValueError(
        "test error"), session_id=session_id)

    events = log.read_session_events(session_id)
    error_event = next((e for e in events if e.get("event") == "error"), None)
    assert error_event is not None
    assert "test error" in error_event["exception"]


def test_logger_recent_events_returns_list():
    events = log.read_recent_events(5)
    assert isinstance(events, list)


def test_logger_security_block_event():
    session_id = f"test-security-{time.time_ns()}"
    log.security_block("Blocked import: os",
                       code_preview="import os", session_id=session_id)

    events = log.read_session_events(session_id)
    sec_event = next((e for e in events if e.get(
        "event") == "security_block"), None)
    assert sec_event is not None
    assert "os" in sec_event["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# Graph Node Unit Tests — should_continue
# ─────────────────────────────────────────────────────────────────────────────

def test_should_continue_returns_end_at_max_iterations():
    state = {"messages": [HumanMessage(content="test")],
             "iteration_count": MAX_ITERATIONS, "error_count": 0}
    assert should_continue(state) == "end"


def test_should_continue_returns_end_at_max_errors():
    state = {"messages": [HumanMessage(content="test")],
             "iteration_count": 1, "error_count": 3}
    assert should_continue(state) == "end"


def test_should_continue_returns_tools_when_tool_calls_present():
    msg = MagicMock()
    msg.tool_calls = [{"name": "execute_python_code", "id": "tc1", "args": {}}]
    state = {"messages": [msg], "iteration_count": 1, "error_count": 0}
    assert should_continue(state) == "tools"


def test_should_continue_returns_end_with_no_tool_calls():
    state = {"messages": [AIMessage(content="Here are the results.")],
             "iteration_count": 1, "error_count": 0}
    assert should_continue(state) == "end"


def test_should_continue_prioritises_iteration_limit_over_tool_calls():
    """Even if the last message has tool_calls, stop at MAX_ITERATIONS."""
    msg = MagicMock()
    msg.tool_calls = [{"name": "execute_python_code", "id": "tc1", "args": {}}]
    state = {"messages": [msg],
             "iteration_count": MAX_ITERATIONS, "error_count": 0}
    assert should_continue(state) == "end"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Node Unit Tests — error_handler_node
# ─────────────────────────────────────────────────────────────────────────────

def test_error_handler_increments_on_execution_error():
    state = {
        "messages": [ToolMessage(content="Execution Error: name 'x' is not defined",
                                 tool_call_id="1")],
        "error_count": 0,
    }
    result = error_handler_node(state)
    assert result["error_count"] == 1


def test_error_handler_injects_recovery_message_at_three_errors():
    """At error_count 2 + one more error → inject recovery HumanMessage."""
    state = {
        "messages": [ToolMessage(content="Execution Error: something failed",
                                 tool_call_id="1")],
        "error_count": 2,
    }
    result = error_handler_node(state)
    assert result["error_count"] == 3
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "different approach" in result["messages"][0].content.lower()


def test_error_handler_resets_count_on_success():
    """A clean tool output must reset the error counter to 0."""
    state = {
        "messages": [ToolMessage(content="Mean salary: 61750.0", tool_call_id="1")],
        "error_count": 2,
    }
    result = error_handler_node(state)
    assert result["error_count"] == 0


def test_error_handler_no_messages_key_on_success():
    """No recovery message should be injected on a successful tool result."""
    state = {
        "messages": [ToolMessage(content="Shape: (100, 5)", tool_call_id="1")],
        "error_count": 0,
    }
    result = error_handler_node(state)
    assert "messages" not in result


# ─────────────────────────────────────────────────────────────────────────────
# Graph Node Unit Tests — visual_enforcer_node
# ─────────────────────────────────────────────────────────────────────────────

def test_visual_enforcer_injects_when_visual_keyword_and_no_execute():
    state = {
        "messages": [HumanMessage(content="Show me a histogram of age")],
        "plot_paths": [],
    }
    result = visual_enforcer_node(state)
    assert "messages" in result
    assert _ENFORCEMENT_MARKER in result["messages"][0].content


def test_visual_enforcer_skips_when_no_visual_keyword():
    state = {
        "messages": [HumanMessage(content="What is the mean salary?")],
        "plot_paths": [],
    }
    assert visual_enforcer_node(state) == {}


def test_visual_enforcer_skips_when_execute_already_called():
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [
        {"name": "execute_python_code", "id": "tc1", "args": {}}]
    # Make content a plain string so the ENFORCEMENT_MARKER check doesn't fire
    type(ai_msg).content = property(lambda self: "")

    state = {
        "messages": [HumanMessage(content="Plot a histogram"), ai_msg],
        "plot_paths": [],
    }
    assert visual_enforcer_node(state) == {}


def test_visual_enforcer_skips_when_plots_already_exist():
    state = {
        "messages": [HumanMessage(content="Show me a scatter plot")],
        "plot_paths": ["sandbox/plot_abc123.png"],
    }
    assert visual_enforcer_node(state) == {}


def test_visual_enforcer_does_not_reinject():
    """Second pass through the node must not inject a second enforcement message."""
    state = {
        "messages": [
            HumanMessage(content="Show me a histogram"),
            HumanMessage(
                content=f"{_ENFORCEMENT_MARKER} You must call execute_python_code"),
        ],
        "plot_paths": [],
    }
    assert visual_enforcer_node(state) == {}


def test_visual_enforcer_triggers_on_all_visual_keywords():
    keywords = ["plot", "chart", "graph", "heatmap", "histogram",
                "visualize", "scatter", "boxplot", "distribution"]
    for kw in keywords:
        state = {
            "messages": [HumanMessage(content=f"Please {kw} the data")],
            "plot_paths": [],
        }
        result = visual_enforcer_node(state)
        assert "messages" in result, f"Enforcer did not trigger for keyword: '{kw}'"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Node Unit Tests — track_plots_node
# ─────────────────────────────────────────────────────────────────────────────

def test_track_plots_detects_plot_saved_marker():
    msg = ToolMessage(
        content="Analysis complete.\n[PLOT_SAVED:sandbox/plot_abc.png]",
        tool_call_id="1"
    )
    state = {"messages": [msg], "plot_paths": []}
    result = track_plots_node(state)
    assert "plot_paths" in result
    assert "sandbox/plot_abc.png" in result["plot_paths"]


def test_track_plots_returns_empty_when_no_marker():
    msg = ToolMessage(content="Mean salary: 61750", tool_call_id="1")
    state = {"messages": [msg], "plot_paths": []}
    assert track_plots_node(state) == {}


def test_track_plots_accumulates_existing_paths():
    msg = ToolMessage(
        content="[PLOT_SAVED:sandbox/plot_new.png]",
        tool_call_id="1"
    )
    state = {"messages": [msg], "plot_paths": ["sandbox/plot_old.png"]}
    result = track_plots_node(state)
    assert "sandbox/plot_old.png" in result["plot_paths"]
    assert "sandbox/plot_new.png" in result["plot_paths"]


# ─────────────────────────────────────────────────────────────────────────────
# Retry System Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_retry_system_recovers_after_one_failure():
    """
    Simulate the agent generating bad code once, then correct code.
    Verify error_count increments on failure and resets on success.
    """
    # First pass: bad code → error
    error_state = {
        "messages": [ToolMessage(content="Execution Error: NameError 'undefined'",
                                 tool_call_id="1")],
        "error_count": 0,
    }
    after_error = error_handler_node(error_state)
    assert after_error["error_count"] == 1

    # Second pass: good code → success
    success_state = {
        "messages": [ToolMessage(content="Mean: 61750.0", tool_call_id="2")],
        "error_count": after_error["error_count"],
    }
    after_success = error_handler_node(success_state)
    assert after_success["error_count"] == 0


def test_retry_system_triggers_hard_stop_after_three_failures():
    """
    After 3 consecutive errors the error_handler must inject a recovery message
    AND should_continue must then return 'end'.
    """
    state = {
        "messages": [ToolMessage(content="Execution Error: repeated failure",
                                 tool_call_id="1")],
        "error_count": 2,
    }
    result = error_handler_node(state)

    # Recovery message injected
    assert result["error_count"] == 3
    assert "messages" in result

    # should_continue must stop the loop
    stop_state = {
        "messages": result["messages"],
        "iteration_count": 1,
        "error_count": result["error_count"],
    }
    assert should_continue(stop_state) == "end"


def test_retry_does_not_loop_indefinitely():
    """
    Simulate MAX_ITERATIONS agent loops — should_continue must eventually
    return 'end' regardless of tool_calls present.
    """
    for i in range(1, MAX_ITERATIONS + 2):
        msg = MagicMock()
        msg.tool_calls = [
            {"name": "execute_python_code", "id": f"tc{i}", "args": {}}]
        state = {"messages": [msg], "iteration_count": i, "error_count": 0}
        if i >= MAX_ITERATIONS:
            assert should_continue(state) == "end", \
                f"Loop did not stop at iteration {i}"


# ─────────────────────────────────────────────────────────────────────────────
# Integration Test — full agent turn (LLM mocked)
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_agent_loads_and_analyses_csv(tmp_path):
    """
    Full graph run: HumanMessage → load_and_inspect_data → execute_python_code
    → final AIMessage. LLM is mocked; only real tool calls execute.

    Verifies:
    - graph terminates without exception
    - at least one tool was called (iteration_count > 0)
    - the final state contains the computed value (61750)
    """
    # Write a real CSV the load tool can read
    csv_file = tmp_path / "salaries.csv"
    TEST_DF.to_csv(csv_file, index=False)

    # Define the mock LLM call sequence
    load_call = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc_load",
            "name": "load_and_inspect_data",
            "args": {"file_path": str(csv_file)},
        }]
    )
    execute_call = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc_exec",
            "name": "execute_python_code",
            "args": {"code": "print(df['salary'].mean())"},
        }]
    )
    final_answer = AIMessage(content="The mean salary is 61750.0")

    call_seq = [load_call, execute_call, final_answer]
    call_idx = {"n": 0}

    def mock_invoke(messages):
        idx = min(call_idx["n"], len(call_seq) - 1)
        call_idx["n"] += 1
        return call_seq[idx]

    initial_state = {
        "messages":           [HumanMessage(content="What is the mean salary?")],
        "iteration_count":    0,
        "error_count":        0,
        "plot_paths":         [],
        "csv_path":           str(csv_file),
        "df_columns":         None,
        "df_shape":           None,
        "df_dtypes":          None,
        "last_code_executed": None,
        "last_tool_output":   None,
        "last_error":         None,
    }

    with patch("agent.graph.llm_with_tools") as mock_llm:
        mock_llm.invoke.side_effect = mock_invoke
        result = agent.invoke(initial_state)

    # Graph must have completed at least one iteration
    assert result["iteration_count"] >= 1

    # The computed value must appear somewhere in the message history
    all_text = " ".join(
        str(getattr(m, "content", "")) for m in result["messages"]
    )
    assert "61750" in all_text


def test_end_to_end_agent_handles_bad_code_and_retries(tmp_path):
    """
    Graph run where the first execute call returns an error.
    Verifies the retry path: error is detected, agent is called again,
    and the graph eventually terminates cleanly.
    """
    csv_file = tmp_path / "data.csv"
    TEST_DF.to_csv(csv_file, index=False)

    bad_execute = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc_bad",
            "name": "execute_python_code",
            "args": {"code": "print(undefined_variable)"},
        }]
    )
    good_execute = AIMessage(
        content="",
        tool_calls=[{
            "id": "tc_good",
            "name": "execute_python_code",
            "args": {"code": "print(df['age'].mean())"},
        }]
    )
    final_answer = AIMessage(content="The mean age is 24.5")

    call_seq = [bad_execute, good_execute, final_answer]
    call_idx = {"n": 0}

    def mock_invoke(messages):
        idx = min(call_idx["n"], len(call_seq) - 1)
        call_idx["n"] += 1
        return call_seq[idx]

    initial_state = {
        "messages":           [HumanMessage(content="What is the mean age?")],
        "iteration_count":    0,
        "error_count":        0,
        "plot_paths":         [],
        "csv_path":           str(csv_file),
        "df_columns":         None,
        "df_shape":           None,
        "df_dtypes":          None,
        "last_code_executed": None,
        "last_tool_output":   None,
        "last_error":         None,
    }

    with patch("agent.graph.llm_with_tools") as mock_llm:
        mock_llm.invoke.side_effect = mock_invoke
        result = agent.invoke(initial_state)

    # Must have gone through at least 2 iterations (bad + good)
    assert result["iteration_count"] >= 2

    all_text = " ".join(str(getattr(m, "content", ""))
                        for m in result["messages"])
    assert "24.5" in all_text
