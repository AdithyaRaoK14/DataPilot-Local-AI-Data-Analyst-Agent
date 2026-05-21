# DataPilot — Local AI Data Analyst Agent

[![Tests](https://github.com/YOUR_USERNAME/DataPilot/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR_USERNAME/DataPilot/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen.svg)](#)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black.svg)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DataPilot** is a fully local, agentic data analysis assistant. Upload a CSV or SQLite database, ask questions in plain English, and the agent autonomously writes and executes Python/SQL code, generates publication-quality charts, and exports full EDA reports — all running on your machine with no cloud APIs required.

---

## Screenshots

| Chat Interface | EDA Report |
|---|---|
| ![UI Image1](Screenshots/UI%20Image1.png) | ![UI Image2](Screenshots/UI%20Image2.png) |

---

## Features

**Conversational data analysis** — ask statistical questions, get exact answers computed from real data, never hallucinated.

**Automatic visualisations** — matplotlib/seaborn static plots and interactive Plotly dashboards rendered inline in the chat.

**SQL database support** — connect to any SQLite `.db` / `.sqlite` file; list tables, load them, run arbitrary SELECT queries, all flowing into the same analysis pipeline.

**One-click EDA reports** — say "generate a full report" and get a polished, self-contained HTML file with histograms, box plots, a correlation heatmap, categorical distributions, outlier tables, and a data-quality score.

**RAG-enhanced context** — a per-dataset vector index (Ollama embeddings or TF-IDF fallback) injects the most relevant column statistics into the system prompt before every LLM call, reducing unnecessary tool calls.

**Disk-backed query cache** — identical questions on the same file are served instantly from cache, auto-invalidating whenever the file changes.

**Glass-box execution trace** — every agent step (tool chosen, code written, output received, retry triggered) is displayed as a collapsible timeline in the UI so you can see exactly what happened.

**Sandboxed code execution** — user-triggered Python runs inside a restricted namespace with no imports allowed; forbidden builtins and write operations are blocked.

**Structured logging** — every tool call, error, and performance metric is written as JSON lines to `logs/agent.jsonl` and surfaced in the sidebar debug panel.

**136 tests, 89% coverage** — full unit and integration suite covering the cache, RAG, SQL tools, Plotly tools, EDA report generator, graph nodes, and the tracer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                │
│   File upload · Chat interface · Plot rendering          │
│   Cache indicator · Execution trace timeline             │
└────────────────────────┬────────────────────────────────┘
                         │  AgentState
┌────────────────────────▼────────────────────────────────┐
│              LangGraph Agent Loop (graph.py)             │
│                                                          │
│  ┌──────────┐    ┌───────────────────────────────────┐  │
│  │  agent   │───▶│           ToolNode                │  │
│  │  node    │    │  load_and_inspect_data            │  │
│  │  (LLM)   │    │  execute_python_code              │  │
│  └──────────┘    │  get_column_statistics            │  │
│       ▲          │  get_correlation_analysis         │  │
│       │          │  list_database_tables             │  │
│  ┌────┴─────┐    │  load_table_from_db               │  │
│  │  error   │    │  execute_sql_query                │  │
│  │ handler  │    │  execute_plotly_code              │  │
│  └────┬─────┘    │  generate_eda_report_tool         │  │
│       │          └───────────────────────────────────┘  │
│  ┌────▼─────┐    ┌──────────────┐  ┌─────────────────┐  │
│  │ visual   │    │ track_plots  │  │  RAG retrieval  │  │
│  │ enforcer │    │    node      │  │  (rag.py)       │  │
│  └──────────┘    └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    QueryCache      sandbox/         logs/
    (cache.py)    plots & reports  agent.jsonl
```

The graph runs as: `agent → tools → track_plots → visual_enforcer → error_handler → agent` until the LLM stops requesting tool calls or the iteration / error limits are reached.

---

## Quick Start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) installed and running

### 1. Clone and install

```bash
git clone https://github.com/AdithyaRaoK14/DataPilot-Local-AI-Data-Analyst-Agent.git
cd DataPilot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull the LLM

```bash
ollama pull qwen2.5:7b
```

The embedding model for RAG is pulled automatically on first use:

```bash
ollama pull nomic-embed-text
```

> **No Ollama?** The agent still works — RAG falls back to TF-IDF automatically, and the LLM can be swapped to any LangChain-compatible provider by editing `cfg.model_name` in `agent/config.py`.

### 3. Launch

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload a CSV or SQLite file, and start asking questions.

---

## Configuration

All tuneable parameters live in `agent/config.py` as a single `AgentConfig` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `qwen2.5:7b` | Ollama model to use |
| `temperature` | `0.0` | LLM temperature |
| `max_iterations` | `12` | Hard cap on agent loop iterations |
| `max_consecutive_errors` | `3` | Errors before forced stop |
| `code_timeout_seconds` | `30` | Sandbox execution timeout |
| `enable_cache` | `True` | Disk-backed query cache |
| `cache_ttl_seconds` | `3600` | Cache entry lifetime |
| `enable_rag` | `True` | RAG context injection |
| `rag_top_k` | `4` | Chunks retrieved per query |
| `rag_force_tfidf` | `False` | Skip Ollama, use TF-IDF |
| `sql_row_limit` | `100 000` | Max rows loaded from SQL |
| `max_upload_mb` | `50.0` | File upload size cap |

---

## Supported File Types

| Format | How it's handled |
|---|---|
| `.csv` | Loaded directly into pandas via `load_and_inspect_data` |
| `.xlsx` / `.xls` | Loaded via pandas Excel reader |
| `.sqlite` / `.db` | Inspected with `list_database_tables`, loaded with `load_table_from_db` or `execute_sql_query` |

---

## Example Questions

```
What is the survival rate by passenger class?
Show me a correlation heatmap of all numeric columns.
Plot the age distribution as a histogram.
Which columns have the most missing values?
Create an interactive scatter plot of Age vs Fare coloured by Survived.
Run a SELECT query to find average fare by embarkation port.
Generate a full EDA report and save it.
```

---

## Project Structure

```
DataPilot/
├── app.py                  # Streamlit frontend
├── agent/
│   ├── config.py           # All tuneable parameters
│   ├── state.py            # LangGraph AgentState TypedDict
│   ├── graph.py            # LangGraph graph definition and all nodes
│   ├── tools.py            # Core tools: load, execute, stats, correlations
│   ├── sql_tools.py        # SQLite tools: list tables, load, query
│   ├── plotly_tools.py     # Interactive Plotly dashboard tool
│   ├── report_generator.py # Automated HTML EDA report builder
│   ├── rag.py              # RAG index (Ollama embeddings / TF-IDF)
│   ├── cache.py            # SHA-256-keyed disk cache
│   ├── logger.py           # Structured JSON-lines logger
│   ├── tracer.py           # Per-turn execution trace + Streamlit renderer
│   └── prompts.py          # System prompt and data-context template
├── tests/
│   ├── conftest.py
│   ├── test_agent.py           # Core agent tests (55 tests)
│   ├── test_coverage_boost.py  # Tools, tracer, logger, cache tests (36 tests)
│   └── test_improvements.py    # SQL, Plotly, EDA report, RAG tests (45 tests)
├── sandbox/                # Auto-created: plots and reports land here
├── logs/                   # Auto-created: agent.log + agent.jsonl
├── cache/                  # Auto-created: query result cache
└── requirements.txt
```

---

## Running Tests

```bash
# Run the full suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=agent --cov-report=term-missing

# Fast run skipping slow integration tests
pytest tests/ -v -m "not slow"
```

Current result: **136 passed** in ~58s, **89% overall coverage**.

---

## How the Agent Loop Works

1. **User uploads a file** and types a question.
2. The `agent_node` assembles a system prompt that includes the analyst instructions, any loaded data-context (shape, columns, dtypes, missing summary), and RAG-retrieved column statistics relevant to the query.
3. The LLM decides which tool to call (e.g. `load_and_inspect_data`, then `execute_python_code`).
4. The `ToolNode` executes the tool and returns the output.
5. `track_plots_node` scans the output for `[PLOT_SAVED:...]`, `[PLOTLY_SAVED:...]`, and `[REPORT_SAVED:...]` markers and records file paths in state.
6. `visual_enforcer_node` checks whether the user asked for a visualisation but no plotting tool was called yet — if so, it injects a correction message.
7. `error_handler_node` counts consecutive errors and injects a recovery prompt after three failures.
8. Control returns to the `agent_node`. The loop continues until the LLM emits a response with no tool calls, or the iteration / error limits are reached.
9. The Streamlit app renders the final text, any plots/HTML files, and the full execution trace.

---

## Security

Code submitted to `execute_python_code` runs in a restricted `exec` namespace:

- All `import` statements are stripped before execution.
- Only a whitelist of safe builtins (`print`, `len`, `range`, etc.) is available.
- `plot_path` and `fig_path` cannot be overwritten by generated code.
- SQL tools block all write/DDL verbs (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `CREATE`, `ALTER`, …) and unsafe `PRAGMA` keys.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) — `qwen2.5:7b` (local, no API key) |
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) + [LangChain](https://github.com/langchain-ai/langchain) |
| Embeddings | Ollama `nomic-embed-text` / scikit-learn TF-IDF fallback |
| UI | [Streamlit](https://streamlit.io) |
| Data | pandas, NumPy |
| Static plots | matplotlib, seaborn |
| Interactive plots | Plotly |
| Testing | pytest, pytest-cov |

---

## License

MIT — see [LICENSE](LICENSE) for details.
