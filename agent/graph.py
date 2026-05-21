# agent/graph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.tools import (
    load_and_inspect_data,
    execute_python_code,
    get_column_statistics,
    get_correlation_analysis,
    _df_store,
)
from agent.sql_tools import (
    list_database_tables,
    load_table_from_db,
    execute_sql_query,
)
from agent.plotly_tools import execute_plotly_code
from agent.report_generator import generate_eda_report_tool
from agent.prompts import ANALYST_SYSTEM_PROMPT, build_data_context
from agent.rag import dataset_rag
from agent.config import cfg
from typing import Literal
import re

# Re-exported for backward compatibility with test_agent.py
MAX_ITERATIONS = cfg.max_iterations

# ── Visual / interactive keyword sets ─────────────────────────────────────────

_VISUAL_KWS = {
    "plot", "chart", "graph", "heatmap", "histogram",
    "visualize", "visualise", "visualization", "visualisation",
    "scatter", "bar", "boxplot", "distribution", "show me",
}

_INTERACTIVE_KWS = {
    "interactive", "dashboard", "plotly", "html chart",
    "html plot", "zoom", "hover", "pan",
}

_REPORT_KWS = {
    "eda report", "full report", "export report", "save report",
    "generate report", "analysis report", "download report",
}

# ── Model setup ────────────────────────────────────────────────────────────────

tools = [
    # original
    load_and_inspect_data,
    execute_python_code,
    get_column_statistics,
    get_correlation_analysis,
    # SQL
    list_database_tables,
    load_table_from_db,
    execute_sql_query,
    # Plotly
    execute_plotly_code,
    # EDA report
    generate_eda_report_tool,
]

llm = ChatOllama(
    model=cfg.model_name,
    temperature=cfg.temperature,
    num_predict=cfg.max_tokens,
)

try:
    llm_with_tools = llm.bind_tools(tools, tool_choice="any")
except Exception:
    llm_with_tools = llm.bind_tools(tools)


# ── System prompt addendum for new tools ──────────────────────────────────────

_NEW_TOOLS_ADDENDUM = """
## Additional Tools Available

### SQL Database Support
When the user provides a .sqlite or .db file:
1. Call list_database_tables(db_path) first — connects and shows schema
2. Call load_table_from_db(db_path, table_name) to load a table into df
3. Call execute_sql_query(query) for custom SELECT queries
Only SELECT queries are allowed. Results load into df for further analysis.

### Interactive Plotly Dashboards
When the user asks for "interactive", "dashboard", "plotly", or "zoom/hover":
- Call execute_plotly_code instead of execute_python_code
- Available: df, pd, np, px (plotly.express), go (plotly.graph_objects)
- Save with: fig.write_html(fig_path)
- Never call fig.show()

Common Plotly patterns:
  # Scatter plot
  fig = px.scatter(df, x='col_a', y='col_b', color='category')
  fig.write_html(fig_path)

  # Correlation heatmap (interactive)
  numeric_cols = df.select_dtypes(include='number').columns.tolist()
  corr = df[numeric_cols].corr().round(3)
  fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                              y=corr.columns.tolist(), colorscale='RdBu_r', zmid=0,
                              text=corr.values.round(2), texttemplate='%{text}'))
  fig.write_html(fig_path)

### EDA Report Export
When the user asks for an "EDA report", "full report", "export", or "download":
- Call generate_eda_report_tool(report_title="My Report")
- This produces a self-contained HTML file with all charts embedded
- Do NOT try to generate the report manually with execute_python_code
"""


# ── Nodes ──────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    """
    The thinking node. Assembles the system prompt (with optional RAG context)
    and asks the LLM which tool to call next.
    """
    system_content = ANALYST_SYSTEM_PROMPT + _NEW_TOOLS_ADDENDUM

    # Inject data-context block when a dataframe is loaded
    if state.get("df_columns") and "current" in _df_store:
        import pandas as pd
        df = _df_store["current"]
        numeric_cols = list(df.select_dtypes(include="number").columns)
        categorical_cols = list(df.select_dtypes(exclude="number").columns)
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].to_dict()

        system_content += build_data_context({
            "csv_path": state.get("csv_path", "unknown"),
            "rows": df.shape[0],
            "cols": df.shape[1],
            "columns": state["df_columns"],
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "missing_summary": str(missing_cols) if missing_cols else "None",
        })

        # ── RAG: index on first load, retrieve on every call ─────────────────
        if cfg.enable_rag:
            csv_path = state.get("csv_path", "dataset")
            if not dataset_rag.is_indexed() or dataset_rag.indexed_path != csv_path:
                try:
                    dataset_rag.index_dataframe(df, source_path=csv_path)
                except Exception:
                    pass  # RAG failure must never crash the agent

            if dataset_rag.is_indexed():
                last_human = next(
                    (m for m in reversed(state["messages"])
                     if getattr(m, "type", "") == "human"),
                    None,
                )
                query_text = str(
                    getattr(last_human, "content", "")) if last_human else ""
                if query_text:
                    try:
                        rag_context = dataset_rag.retrieve_as_context(
                            query_text)
                        if rag_context:
                            system_content += "\n\n" + rag_context
                    except Exception:
                        pass

    messages = [SystemMessage(content=system_content)] + state["messages"]
    response = llm_with_tools.invoke(messages)

    new_state: dict = {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }

    # Capture df metadata after first load
    if "current" in _df_store and not state.get("df_columns"):
        df = _df_store["current"]
        new_state["df_columns"] = list(df.columns)
        new_state["df_shape"] = df.shape

    return new_state


def track_plots_node(state: AgentState) -> dict:
    """
    Scans recent tool outputs for [PLOT_SAVED:...], [PLOTLY_SAVED:...],
    and [REPORT_SAVED:...] markers — all stored in plot_paths so the UI
    can render / offer downloads appropriately based on file extension.
    """
    new_paths: list[str] = []

    for msg in reversed(state["messages"][-5:]):
        raw = getattr(msg, "content", "")
        if isinstance(raw, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in raw
            )
        else:
            content = str(raw)

        for marker in ["[PLOT_SAVED:", "[PLOTLY_SAVED:", "[REPORT_SAVED:"]:
            if marker in content:
                matches = re.findall(
                    re.escape(marker) + r"(.*?)\]", content
                )
                new_paths.extend(matches)

    if new_paths:
        existing = state.get("plot_paths", [])
        return {"plot_paths": existing + new_paths}

    return {}


def error_handler_node(state: AgentState) -> dict:
    """
    Detects repeated errors and injects a recovery prompt after 3 failures.
    """
    last_tool_output = ""
    for msg in reversed(state["messages"][-3:]):
        content = str(getattr(msg, "content", ""))
        if "Execution Error" in content or "SQL Error" in content:
            last_tool_output = content
            break

    error_count = state.get("error_count", 0)

    if "Execution Error" in last_tool_output or "SQL Error" in last_tool_output:
        new_count = error_count + 1
        if new_count >= cfg.max_consecutive_errors:
            return {
                "error_count": new_count,
                "messages": [
                    HumanMessage(
                        content=(
                            "You have encountered multiple errors. "
                            "Please try a completely different approach, "
                            "or explain clearly why the question cannot be answered."
                        )
                    )
                ],
                "last_error": last_tool_output,
            }
        return {"error_count": new_count}

    return {"error_count": 0}


def _user_wants_visual(state: AgentState) -> bool:
    for msg in state["messages"]:
        if getattr(msg, "type", "") == "human":
            text = str(getattr(msg, "content", "")).lower()
            if any(kw in text for kw in _VISUAL_KWS | _INTERACTIVE_KWS):
                return True
    return False


def _execute_was_called(state: AgentState) -> bool:
    for msg in state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") in ("execute_python_code", "execute_plotly_code"):
                    return True
    return False


_ENFORCEMENT_MARKER = "VISUAL_ENFORCEMENT:"


def visual_enforcer_node(state: AgentState) -> dict:
    """
    Injects a correction when the user requested a visualisation but the agent
    has not yet called any code-execution tool.
    """
    if not _user_wants_visual(state):
        return {}

    if _execute_was_called(state) or state.get("plot_paths"):
        return {}

    already_injected = any(
        _ENFORCEMENT_MARKER in str(getattr(m, "content", ""))
        for m in state["messages"]
    )
    if already_injected:
        return {}

    # Decide which tool the agent should use
    last_human = next(
        (m for m in reversed(state["messages"])
         if getattr(m, "type", "") == "human"),
        None,
    )
    query = str(getattr(last_human, "content", "")
                ).lower() if last_human else ""
    wants_interactive = any(kw in query for kw in _INTERACTIVE_KWS)

    tool_instruction = (
        "execute_plotly_code with plotly.express/graph_objects code "
        "that builds an interactive figure and saves it with fig.write_html(fig_path)."
        if wants_interactive else
        "execute_python_code with matplotlib/seaborn code "
        "that creates the requested plot and saves it with plt.savefig(plot_path)."
    )

    return {
        "messages": [
            HumanMessage(
                content=(
                    f"{_ENFORCEMENT_MARKER} The user requested a visualization. "
                    f"You have NOT called a plotting tool yet. "
                    f"You MUST call {tool_instruction}"
                )
            )
        ]
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]

    if state.get("iteration_count", 0) >= cfg.max_iterations:
        return "end"
    if state.get("error_count", 0) >= cfg.max_consecutive_errors:
        return "end"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_agent():
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("track_plots", track_plots_node)
    graph.add_node("visual_enforcer", visual_enforcer_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )

    graph.add_edge("tools", "track_plots")
    graph.add_edge("track_plots", "visual_enforcer")
    graph.add_edge("visual_enforcer", "error_handler")
    graph.add_edge("error_handler", "agent")

    return graph.compile()


agent = build_agent()
