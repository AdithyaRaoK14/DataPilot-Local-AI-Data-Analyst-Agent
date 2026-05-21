# agent/tracer.py
"""
Execution trace recorder.

Every agent step is appended to a list of TraceStep objects stored in
Streamlit's session_state.  The UI renders this as a collapsible timeline
showing: question → tool selected → code → output → retry → final answer.

This gives reviewers (and interviewers) a glass-box view of the agent's
reasoning loop, which transforms "it just works" into a demonstrably
engineered system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class TraceStep:
    step_number: int
    tool_name: str
    tool_input_preview: str          # first 400 chars of input
    tool_output_preview: str = ""    # first 600 chars of output
    duration_ms: float = 0.0
    success: bool = True
    plot_generated: bool = False
    is_retry: bool = False           # True if same tool called consecutively
    timestamp: float = field(default_factory=time.time)

    def status_emoji(self) -> str:
        if not self.success:
            return "❌"
        if self.plot_generated:
            return "📊"
        if self.is_retry:
            return "🔄"
        return "✅"

    def short_label(self) -> str:
        tool_icons = {
            "load_and_inspect_data": "📂 load_data",
            "execute_python_code": "🐍 run_code",
            "get_column_statistics": "📈 col_stats",
            "get_correlation_analysis": "🔗 correlations",
            "load_excel_file": "📊 load_excel",
            "load_sqlite_database": "🗄️ load_sqlite",
        }
        icon = tool_icons.get(self.tool_name, f"🔧 {self.tool_name}")
        retry_tag = " (retry)" if self.is_retry else ""
        return f"Step {self.step_number} · {icon}{retry_tag}"


@dataclass
class ExecutionTrace:
    """Mutable trace object for one agent turn."""
    question: str = ""
    steps: list[TraceStep] = field(default_factory=list)
    final_answer_preview: str = ""
    total_elapsed_s: float = 0.0
    iteration_count: int = 0
    error_count: int = 0
    cache_hit: bool = False

    def add_step(
        self,
        tool_name: str,
        tool_input_preview: str,
        tool_output_preview: str = "",
        duration_ms: float = 0.0,
        success: bool = True,
        plot_generated: bool = False,
    ) -> TraceStep:
        is_retry = (
            bool(self.steps)
            and self.steps[-1].tool_name == tool_name
        )
        step = TraceStep(
            step_number=len(self.steps) + 1,
            tool_name=tool_name,
            tool_input_preview=tool_input_preview[:400],
            tool_output_preview=tool_output_preview[:600],
            duration_ms=duration_ms,
            success=success,
            plot_generated=plot_generated,
            is_retry=is_retry,
        )
        self.steps.append(step)
        return step

    def summary(self) -> dict:
        return {
            "total_steps": len(self.steps),
            "successful_steps": sum(1 for s in self.steps if s.success),
            "failed_steps": sum(1 for s in self.steps if not s.success),
            "plots_generated": sum(1 for s in self.steps if s.plot_generated),
            "retries": sum(1 for s in self.steps if s.is_retry),
            "tools_used": list(dict.fromkeys(s.tool_name for s in self.steps)),
            "total_elapsed_s": round(self.total_elapsed_s, 2),
            "cache_hit": self.cache_hit,
        }


# ─── Streamlit rendering helper ───────────────────────────────────────────────

def render_trace_in_streamlit(trace: ExecutionTrace):
    """
    Renders the full execution trace inside a Streamlit expander.
    Call this from app.py after agent.invoke() returns.
    """
    import streamlit as st

    if not trace.steps:
        return

    summary = trace.summary()

    with st.expander(
        f"🔍 Execution Trace  "
        f"({summary['total_steps']} steps · "
        f"{summary['total_elapsed_s']}s · "
        f"{summary['plots_generated']} plots)",
        expanded=False,
    ):
        # Top-level metrics row
        cols = st.columns(5)
        cols[0].metric("Steps", summary["total_steps"])
        cols[1].metric("✅ Success", summary["successful_steps"])
        cols[2].metric("❌ Errors", summary["failed_steps"])
        cols[3].metric("📊 Plots", summary["plots_generated"])
        cols[4].metric("🔄 Retries", summary["retries"])

        if summary["cache_hit"]:
            st.info("⚡ Cache hit — result served from cache")

        st.markdown("---")
        st.markdown("**Tools used:** " + " → ".join(summary["tools_used"]))
        st.markdown("---")

        # Individual step timeline
        for step in trace.steps:
            status = step.status_emoji()
            with st.expander(f"{status} {step.short_label()}  ({step.duration_ms:.0f}ms)", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Input**")
                    st.code(step.tool_input_preview or "(no input)",
                            language="python")
                with col_b:
                    st.markdown("**Output**")
                    st.code(step.tool_output_preview or "(no output)",
                            language="text")
