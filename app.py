# app.py
import streamlit as st
import streamlit.components.v1 as components
from agent.graph import agent
from agent.state import AgentState
from agent.tools import _df_store
from agent.tracer import ExecutionTrace, render_trace_in_streamlit
from agent.logger import log
from agent.cache import cache
from agent.config import cfg
from agent.rag import dataset_rag
from langchain_core.messages import HumanMessage, ToolMessage
import pandas as pd
import os
import time
import re
import glob

# ─── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Autonomous Data Analyst Agent")
st.caption(
    "Upload a CSV or SQLite database · Ask questions in plain English · "
    "Agent writes and executes code to answer"
)


# ─── Session state ─────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],
        "agent_messages": [],
        "csv_path": None,
        "db_path": None,
        "df_info": None,
        "plot_paths": [],
        "iteration_logs": [],
        "last_uploaded": None,
        "last_iteration_count": 0,
        "session_id": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if not st.session_state.session_id:
        import uuid
        st.session_state.session_id = uuid.uuid4().hex[:8]


init_session()


# ─── Constants ─────────────────────────────────────────────────────────────────

TOOL_ICONS = {
    "load_and_inspect_data": "📂",
    "execute_python_code": "🐍",
    "get_column_statistics": "📈",
    "get_correlation_analysis": "🔗",
    "list_database_tables": "🗄️",
    "load_table_from_db": "📥",
    "execute_sql_query": "🔍",
    "execute_plotly_code": "📊",
    "generate_eda_report_tool": "📝",
}

_VISUAL_KEYWORDS = {
    "plot", "graph", "chart", "heatmap", "histogram",
    "visualize", "visualise", "visualization", "visualisation",
    "scatter", "bar", "pie", "boxplot", "distribution",
    "interactive", "dashboard", "plotly",
}


def _needs_visual(prompt: str) -> bool:
    lower = prompt.lower()
    return any(kw in lower for kw in _VISUAL_KEYWORDS)


# ─── Output rendering helpers ─────────────────────────────────────────────────

def _render_output_file(path: str, col=None):
    """
    Render a sandbox output file appropriately:
      .png / .jpg  → st.image
      .html        → st.components iframe + download button
    """
    target = col if col is not None else st

    if not os.path.exists(path):
        return

    ext = os.path.splitext(path)[1].lower()

    if ext in (".png", ".jpg", ".jpeg", ".gif"):
        target.image(path, use_container_width=True)

    elif ext == ".html":
        html_content = open(path, encoding="utf-8").read()
        is_report = "eda_report_" in os.path.basename(path)

        label = "📝 EDA Report" if is_report else "📊 Interactive Chart"
        target.markdown(f"**{label}**")

        # Render inside an iframe — height heuristic: reports are tall
        iframe_height = 800 if is_report else 520
        components.html(html_content, height=iframe_height, scrolling=True)

        # Download button
        target.download_button(
            label=f"⬇️ Download {'report' if is_report else 'chart'} HTML",
            data=html_content,
            file_name=os.path.basename(path),
            mime="text/html",
        )


def _render_all_outputs(paths: list[str]):
    """Render a list of output files, side-by-side for images."""
    if not paths:
        return

    # Separate images from HTML
    images = [p for p in paths if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg")]
    html_files = [p for p in paths if os.path.splitext(p)[1].lower() == ".html"]

    if images:
        cols = st.columns(min(len(images), 2))
        for i, p in enumerate(images):
            _render_output_file(p, col=cols[i % 2])

    for p in html_files:
        _render_output_file(p)


# ─── Trace builder ─────────────────────────────────────────────────────────────

def build_trace_from_result(result: dict, question: str, elapsed_s: float) -> ExecutionTrace:
    trace = ExecutionTrace(question=question)
    trace.total_elapsed_s = elapsed_s
    trace.iteration_count = result.get("iteration_count", 0)
    trace.error_count = result.get("error_count", 0)

    messages = result.get("messages", [])
    tool_results: dict[str, ToolMessage] = {
        msg.tool_call_id: msg
        for msg in messages
        if isinstance(msg, ToolMessage)
    }

    for msg in messages:
        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
            continue
        for tc in msg.tool_calls:
            tool_name = tc.get("name", "unknown")
            args = tc.get("args", {})
            input_preview = str(next(iter(args.values()), "")) if args else ""

            result_msg = tool_results.get(tc.get("id", ""))
            output_preview = ""
            success = True
            plot_generated = False

            if result_msg:
                output_preview = str(result_msg.content)
                success = not any(
                    tag in output_preview
                    for tag in ("Execution Error", "Security Error",
                                "SQL Error", "Error loading", "Safety error")
                )
                plot_generated = any(
                    marker in output_preview
                    for marker in ("[PLOT_SAVED:", "[PLOTLY_SAVED:", "[REPORT_SAVED:")
                )

            trace.add_step(
                tool_name=tool_name,
                tool_input_preview=input_preview,
                tool_output_preview=output_preview,
                duration_ms=0.0,
                success=success,
                plot_generated=plot_generated,
            )

    return trace


# ─── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📁 Data")

    uploaded = st.file_uploader(
        "Upload CSV or SQLite database",
        type=["csv", "sqlite", "db"],
        key="file_uploader",
    )

    if uploaded:
        save_path = f"sandbox/{uploaded.name}"
        os.makedirs("sandbox", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        is_db = uploaded.name.lower().endswith((".sqlite", ".db"))

        if is_db:
            st.session_state.db_path = save_path
            st.session_state.csv_path = save_path   # reuse csv_path as primary key
            st.session_state.df_info = None
        else:
            df = pd.read_csv(save_path)
            st.session_state.csv_path = save_path
            st.session_state.db_path = None
            st.session_state.df_info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }

        if st.session_state.get("last_uploaded") != uploaded.name:
            st.session_state.messages = []
            st.session_state.agent_messages = []
            st.session_state.plot_paths = []
            st.session_state.last_uploaded = uploaded.name
            _df_store.clear()
            dataset_rag.chunks.clear()   # reset RAG index on new file
            log.session_start(st.session_state.session_id, save_path)

        st.success(f"✅ {uploaded.name}")

        if is_db:
            st.markdown("**SQLite database — ask the agent to list tables**")
        else:
            st.markdown(f"**{df.shape[0]:,} rows · {df.shape[1]} columns**")
            with st.expander("Preview (first 5 rows)", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            with st.expander("Column info", expanded=False):
                info_df = pd.DataFrame({
                    "Column": df.columns,
                    "Type": df.dtypes.astype(str).values,
                    "Nulls": df.isnull().sum().values,
                    "Unique": df.nunique().values,
                })
                st.dataframe(info_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Cache control ─────────────────────────────────────────────────────────
    st.header("🗄️ Cache")
    force_refresh = st.checkbox(
        "🔄 Bypass cache", value=False,
        help="Skip cached result and always run the agent fresh.",
    )

    st.divider()

    # ── System status ─────────────────────────────────────────────────────────
    st.header("⚙️ System Status")
    cache_stats = cache.stats()
    rag_status = (
        f"✅ {dataset_rag.backend} ({len(dataset_rag.chunks)} chunks)"
        if dataset_rag.is_indexed()
        else "⏳ not indexed yet"
    )
    for label, value in [
        ("Model",       f"`{cfg.model_name}`"),
        ("Temp",        f"`{cfg.temperature}`"),
        ("Cache",       "✅ enabled" if cfg.enable_cache else "❌ disabled"),
        ("RAG",         rag_status),
        ("Timeout",     f"`{cfg.code_timeout_seconds}s`"),
        ("Max iter",    f"`{cfg.max_iterations}`"),
        ("Cache entries",
         f"{cache_stats['live_entries']} live · {cache_stats['expired_entries']} expired"),
    ]:
        st.markdown(f"**{label}:** {value}")

    st.divider()

    st.header("💡 Try asking")
    suggestions = [
        "Load the dataset and give me a complete overview.",
        "Is there any class imbalance? Show me a visualization.",
        "Show an interactive correlation heatmap.",
        "Plot the distribution of all numeric columns.",
        "Generate a full EDA report and save it.",
        "Are there significant outliers? Which rows should I investigate?",
        "What preprocessing steps would you recommend before training an ML model?",
    ]
    for s in suggestions:
        if st.button(s, key=f"sugg_{s[:25]}", use_container_width=True):
            st.session_state["pending_query"] = s

    st.divider()

    with st.expander("🔧 Agent Debug", expanded=False):
        st.markdown(
            f"**Session:** `{st.session_state.session_id}`  \n"
            f"**Last iterations:** `{st.session_state.get('last_iteration_count', 0)}`"
        )
        if st.session_state.get("iteration_logs"):
            for entry in st.session_state["iteration_logs"][-5:]:
                st.markdown(f"- {entry}")

    with st.expander("📋 Recent Logs", expanded=False):
        events = log.read_recent_events(15)
        if events:
            for event in reversed(events):
                ts = event.get("ts", "")[:19].replace("T", " ")
                evt_type = event.get("event", "?")
                detail = event.get("tool", event.get("context", ""))
                ok = event.get("success", True)
                icon = "✅" if ok else "❌"
                if evt_type == "security_block":
                    icon = "🚫"
                st.text(f"{icon} {ts}  [{evt_type}]  {detail}")
        else:
            st.caption("No log entries yet.")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent_messages = []
        st.session_state.plot_paths = []
        st.rerun()


# ─── Chat history ───────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plots"):
            _render_all_outputs(msg["plots"])
        if msg.get("trace"):
            render_trace_in_streamlit(msg["trace"])


# ─── Input handling ────────────────────────────────────────────────────────────

query_from_suggestion = st.session_state.pop("pending_query", None)
prompt = st.chat_input("Ask anything about your data...") or query_from_suggestion

if prompt:
    if not st.session_state.csv_path:
        st.warning("⬆️ Upload a CSV or SQLite file first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        response_placeholder = st.empty()

        # ── Cache lookup ──────────────────────────────────────────────────────
        cached_result = None if force_refresh else cache.get(
            st.session_state.csv_path, prompt
        )

        if cached_result and _needs_visual(prompt):
            valid_cached = [
                p for p in cached_result.get("plot_paths", [])
                if os.path.exists(p)
            ]
            if not valid_cached:
                cached_result = None

        if cached_result:
            response_text = cached_result["response_text"]
            valid_plots = cached_result.get("plot_paths", [])

            status_placeholder.empty()
            st.info("⚡ Served from cache")
            response_placeholder.markdown(response_text)

            if valid_plots:
                _render_all_outputs(valid_plots)

            trace = ExecutionTrace(question=prompt)
            trace.cache_hit = True
            trace.iteration_count = cached_result.get("iteration_count", 0)
            render_trace_in_streamlit(trace)
            st.caption(
                f"⚡ Cache hit · {cached_result.get('iteration_count', 0)} steps (original run)"
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "plots": valid_plots,
                "trace": trace,
            })
            st.stop()

        # ── Live agent run ────────────────────────────────────────────────────
        status_placeholder.markdown("🤔 *Agent is thinking...*")

        visual_instruction = ""
        if _needs_visual(prompt):
            lower = prompt.lower()
            wants_interactive = any(
                kw in lower for kw in ("interactive", "dashboard", "plotly", "hover", "zoom")
            )
            if wants_interactive:
                visual_instruction = (
                    "\nINTERACTIVE TASK: The user wants an interactive chart. "
                    "You MUST call execute_plotly_code with plotly.express or "
                    "plotly.graph_objects code. Save with fig.write_html(fig_path). "
                    "Do NOT use execute_python_code for this task.\n"
                )
            else:
                visual_instruction = (
                    "\nVISUAL TASK: The user wants a visualization. "
                    "You MUST call execute_python_code with matplotlib/seaborn code "
                    "that creates the plot and saves it via plt.savefig(plot_path). "
                    "Do NOT use get_correlation_analysis as a substitute for drawing.\n"
                )

        agent_prompt = (
            f"File is at: {st.session_state.csv_path}\n"
            f"{visual_instruction}"
            f"IMPORTANT: Call tools to execute code. "
            f"Do not write raw code blocks in your text response.\n\n"
            f"User question: {prompt}"
        )

        initial_state = AgentState(
            messages=[HumanMessage(content=agent_prompt)],
            csv_path=st.session_state.csv_path,
            df_shape=st.session_state.df_info["shape"] if st.session_state.df_info else None,
            df_columns=st.session_state.df_info["columns"] if st.session_state.df_info else None,
            df_dtypes=None,
            iteration_count=0,
            last_code_executed=None,
            last_tool_output=None,
            plot_paths=[],
            error_count=0,
            last_error=None,
        )

        start_time = time.time()
        session_id = st.session_state.session_id
        log.session_start(session_id, st.session_state.csv_path)

        try:
            result = None
            for state_snapshot in agent.stream(initial_state, stream_mode="values"):
                result = state_snapshot
                msgs = state_snapshot.get("messages", [])
                if msgs:
                    last_msg = msgs[-1]
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        names = [tc["name"] for tc in last_msg.tool_calls]
                        icons = " · ".join(
                            f"{TOOL_ICONS.get(n, '🔧')} `{n}`" for n in names
                        )
                        status_placeholder.markdown(f"*Calling: {icons}…*")

            elapsed = round(time.time() - start_time, 1)

            # Extract response text
            final_message = result["messages"][-1]
            response_text = getattr(final_message, "content", str(final_message))
            # Strip all marker tags from visible response
            response_text = re.sub(
                r'\[(PLOT_SAVED|PLOTLY_SAVED|REPORT_SAVED):.*?\]',
                "",
                response_text,
            ).strip()

            status_placeholder.empty()
            response_placeholder.markdown(response_text)

            # Collect valid output files
            state_paths = result.get("plot_paths", [])
            seen: set[str] = set()
            valid_outputs: list[str] = []
            for p in state_paths:
                if p not in seen and os.path.exists(p):
                    seen.add(p)
                    valid_outputs.append(p)

            if valid_outputs:
                _render_all_outputs(valid_outputs)

            trace = build_trace_from_result(result, question=prompt, elapsed_s=elapsed)
            render_trace_in_streamlit(trace)

            cache.set(
                st.session_state.csv_path, prompt,
                response_text, valid_outputs,
                iteration_count=result.get("iteration_count", 0),
            )

            st.session_state.agent_messages = result["messages"]
            st.session_state.last_iteration_count = result.get("iteration_count", 0)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "plots": valid_outputs,
                "trace": trace,
            })

            log.performance(
                session_id=session_id,
                elapsed_s=elapsed,
                iterations=result.get("iteration_count", 0),
                plots_generated=len(valid_outputs),
                error_count=result.get("error_count", 0),
            )

            st.caption(
                f"⏱️ {elapsed}s · {result.get('iteration_count', 0)} agent steps"
            )

        except Exception as e:
            status_placeholder.empty()
            log.error("app_invoke", exc=e, session_id=session_id)
            error_msg = f"Agent error: {str(e)}\n\nTry rephrasing your question."
            response_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
