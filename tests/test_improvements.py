# tests/test_improvements.py
"""
Production-ready tests for the four new agent improvements:
    1. SQL database support (agent/sql_tools.py)
    2. Plotly interactive dashboards (agent/plotly_tools.py)
    3. Automated EDA report export (agent/report_generator.py)
    4. RAG-enhanced dataset understanding (agent/rag.py)

Test strategy
-------------
* All tests are fully self-contained — they create their own temp files and
  clean up after themselves.
* SQL tests use the stdlib sqlite3 module to build in-memory / on-disk DBs.
* Plotly tests pre-load a dataframe into _df_store and check HTML output.
* EDA report tests run generate_eda_report() directly and inspect the HTML.
* RAG tests force TF-IDF mode (force_tfidf=True) so they pass without Ollama.
* The conftest.py multiprocessing guard is inherited automatically.
"""

import os
import re
import sqlite3
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

# ── Shared fixtures ────────────────────────────────────────────────────────────

SAMPLE_DF = pd.DataFrame({
    "age":    [21, 25, 30, 22, 45, 60, 18, 33],
    "salary": [50_000, 65_000, 80_000, 52_000, 120_000, 95_000, 30_000, 70_000],
    "target": [0, 1, 1, 0, 1, 1, 0, 1],
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
    "city":   ["NY", "LA", "NY", "SF", "LA", "NY", "SF", "LA"],
})


def _make_db(path: str) -> None:
    """Create a two-table SQLite database at *path*."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE employees "
        "(id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept TEXT, salary REAL)"
    )
    conn.executemany(
        "INSERT INTO employees VALUES (?,?,?,?)",
        [
            (1, "Alice", "Engineering", 120_000),
            (2, "Bob",   "Marketing",    85_000),
            (3, "Carol", "Engineering",  95_000),
            (4, "Dan",   "HR",           70_000),
        ],
    )
    conn.execute(
        "CREATE TABLE departments (dept TEXT PRIMARY KEY, head TEXT)"
    )
    conn.executemany(
        "INSERT INTO departments VALUES (?,?)",
        [("Engineering", "Alice"), ("Marketing", "Bob"), ("HR", "Dan")],
    )
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 1. SQL Database Support
# ══════════════════════════════════════════════════════════════════════════════

class TestSQLTools:

    def setup_method(self):
        """Reset the module-level db_store before each test."""
        from agent import sql_tools
        sql_tools._db_store.clear()

    def test_list_tables_shows_all_tables(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables
        result = list_database_tables.invoke({"db_path": db})
        assert "employees" in result
        assert "departments" in result
        assert "4" in result          # row count for employees

    def test_list_tables_shows_column_schema(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables
        result = list_database_tables.invoke({"db_path": db})
        assert "salary" in result
        assert "name" in result

    def test_list_tables_missing_file_returns_error(self):
        from agent.sql_tools import list_database_tables
        result = list_database_tables.invoke({"db_path": "/nonexistent/test.db"})
        assert "not found" in result.lower() or "error" in result.lower()

    def test_load_table_populates_df_store(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, load_table_from_db
        from agent.tools import _df_store
        list_database_tables.invoke({"db_path": db})     # connect
        result = load_table_from_db.invoke({"db_path": db, "table_name": "employees"})
        assert "current" in _df_store
        df = _df_store["current"]
        assert df.shape == (4, 4)
        assert "salary" in df.columns
        assert "employees" in result or "4" in result

    def test_load_table_missing_table_returns_error(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, load_table_from_db
        list_database_tables.invoke({"db_path": db})
        result = load_table_from_db.invoke({"db_path": db, "table_name": "ghost_table"})
        assert "error" in result.lower()

    def test_execute_sql_query_select(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, execute_sql_query
        list_database_tables.invoke({"db_path": db})
        result = execute_sql_query.invoke(
            {"query": "SELECT name, salary FROM employees WHERE salary > 90000"}
        )
        assert "Alice" in result or "Carol" in result
        assert "error" not in result.lower()

    def test_execute_sql_query_aggregation(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, execute_sql_query
        from agent.tools import _df_store
        list_database_tables.invoke({"db_path": db})
        execute_sql_query.invoke(
            {"query": "SELECT dept, AVG(salary) as avg_sal FROM employees GROUP BY dept"}
        )
        df = _df_store["current"]
        assert "avg_sal" in df.columns
        assert len(df) == 3           # 3 departments

    def test_execute_sql_query_blocks_insert(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, execute_sql_query
        list_database_tables.invoke({"db_path": db})
        result = execute_sql_query.invoke(
            {"query": "INSERT INTO employees VALUES (99,'Eve','IT',50000)"}
        )
        assert "safety" in result.lower() or "only select" in result.lower()

    def test_execute_sql_query_blocks_drop(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, execute_sql_query
        list_database_tables.invoke({"db_path": db})
        result = execute_sql_query.invoke({"query": "DROP TABLE employees"})
        assert "safety" in result.lower() or "only select" in result.lower()

    def test_execute_sql_query_without_connection_returns_hint(self):
        from agent import sql_tools
        sql_tools._db_store.clear()
        from agent.sql_tools import execute_sql_query
        result = execute_sql_query.invoke({"query": "SELECT 1"})
        assert "connect" in result.lower() or "list_database_tables" in result

    def test_result_loads_into_df_store(self, tmp_path):
        db = str(tmp_path / "test.db")
        _make_db(db)
        from agent.sql_tools import list_database_tables, execute_sql_query
        from agent.tools import _df_store
        list_database_tables.invoke({"db_path": db})
        execute_sql_query.invoke({"query": "SELECT * FROM employees"})
        assert "current" in _df_store
        assert "salary" in _df_store["current"].columns


# ══════════════════════════════════════════════════════════════════════════════
# 2. Plotly Interactive Dashboards
# ══════════════════════════════════════════════════════════════════════════════

class TestPlotlyTools:

    def setup_method(self):
        from agent.tools import _df_store
        _df_store["current"] = SAMPLE_DF.copy()

    def test_scatter_generates_html(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SANDBOX_DIR", str(tmp_path))
        import agent.plotly_tools as pt
        orig_dir = pt.SANDBOX_DIR
        pt.SANDBOX_DIR = str(tmp_path)
        try:
            from agent.plotly_tools import execute_plotly_code
            result = execute_plotly_code.invoke({
                "code": (
                    "fig = px.scatter(df, x='age', y='salary', color='gender')\n"
                    "fig.write_html(fig_path)"
                )
            })
            assert "[PLOTLY_SAVED:" in result
            # Verify the file actually exists
            saved = re.search(r'\[PLOTLY_SAVED:(.*?)\]', result)
            assert saved and os.path.exists(saved.group(1))
        finally:
            pt.SANDBOX_DIR = orig_dir

    def test_html_output_contains_plotly_data(self, tmp_path, monkeypatch):
        import agent.plotly_tools as pt
        pt.SANDBOX_DIR = str(tmp_path)
        from agent.plotly_tools import execute_plotly_code
        result = execute_plotly_code.invoke({
            "code": (
                "fig = px.bar(df, x='city', y='salary')\n"
                "fig.write_html(fig_path)"
            )
        })
        saved = re.search(r'\[PLOTLY_SAVED:(.*?)\]', result)
        assert saved
        html_content = Path(saved.group(1)).read_text(encoding="utf-8")
        # Real Plotly HTML always contains this script inclusion marker
        assert "plotly" in html_content.lower()
        assert "data" in html_content.lower()

    def test_plotly_error_is_captured(self):
        from agent.plotly_tools import execute_plotly_code
        result = execute_plotly_code.invoke({"code": "fig = px.scatter(df, x='nonexistent')"})
        assert "Error" in result or "error" in result

    def test_forbidden_import_stripped(self, tmp_path):
        import agent.plotly_tools as pt
        pt.SANDBOX_DIR = str(tmp_path)
        from agent.plotly_tools import execute_plotly_code
        # import lines should be stripped — code should still run
        result = execute_plotly_code.invoke({
            "code": (
                "import plotly.express as px\n"
                "fig = px.histogram(df, x='age')\n"
                "fig.write_html(fig_path)"
            )
        })
        assert "[PLOTLY_SAVED:" in result

    def test_fig_path_cannot_be_overwritten(self, tmp_path):
        import agent.plotly_tools as pt
        pt.SANDBOX_DIR = str(tmp_path)
        from agent.plotly_tools import execute_plotly_code
        result = execute_plotly_code.invoke({
            "code": (
                "fig_path = '/tmp/hacked.html'\n"
                "fig = px.scatter(df, x='age', y='salary')\n"
                "fig.write_html(fig_path)"
            )
        })
        assert "[PLOTLY_SAVED:" in result
        saved = re.search(r'\[PLOTLY_SAVED:(.*?)\]', result)
        if saved:
            assert "hacked" not in saved.group(1)

    def test_no_df_returns_clear_message(self):
        from agent.tools import _df_store
        _df_store.pop("current", None)
        from agent.plotly_tools import execute_plotly_code
        result = execute_plotly_code.invoke({"code": "fig = px.scatter(df, x='a', y='b')"})
        assert "No dataframe" in result or "load" in result.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 3. Automated EDA Report
# ══════════════════════════════════════════════════════════════════════════════

class TestEDAReport:

    def test_report_creates_html_file(self, tmp_path):
        from agent.report_generator import generate_eda_report
        path = generate_eda_report(SAMPLE_DF, title="Test Report", output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_report_contains_key_sections(self, tmp_path):
        from agent.report_generator import generate_eda_report
        path = generate_eda_report(SAMPLE_DF, output_dir=str(tmp_path))
        html = Path(path).read_text(encoding="utf-8")
        for section in ["quality score", "column schema", "missing value", "outlier"]:
            assert section.lower() in html.lower(), (
                f"Expected section '{section}' not found in report HTML"
            )

    def test_report_embeds_plotly_charts(self, tmp_path):
        from agent.report_generator import generate_eda_report
        path = generate_eda_report(SAMPLE_DF, output_dir=str(tmp_path))
        html = Path(path).read_text(encoding="utf-8")
        assert "Plotly.newPlot" in html
        assert "plotly-" in html      # CDN script tag

    def test_report_title_appears_in_html(self, tmp_path):
        from agent.report_generator import generate_eda_report
        path = generate_eda_report(
            SAMPLE_DF, title="My Custom EDA Report", output_dir=str(tmp_path)
        )
        html = Path(path).read_text(encoding="utf-8")
        assert "My Custom EDA Report" in html

    def test_data_quality_score_completeness(self):
        from agent.report_generator import _compute_quality_score
        full_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        scores = _compute_quality_score(full_df)
        assert scores["completeness"] == 100.0
        assert scores["overall"] > 0

    def test_data_quality_score_missing_reduces_completeness(self):
        from agent.report_generator import _compute_quality_score
        df_with_missing = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
        scores = _compute_quality_score(df_with_missing)
        assert scores["completeness"] < 100.0

    def test_data_quality_score_duplicates_reduce_uniqueness(self):
        from agent.report_generator import _compute_quality_score
        df_dup = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        scores = _compute_quality_score(df_dup)
        assert scores["uniqueness"] < 100.0

    def test_outlier_summary_detects_extreme_values(self):
        from agent.report_generator import _outlier_summary
        df = pd.DataFrame({"x": [10, 10, 10, 10, 10, 10, 10, 10, 10, 1000]})
        outliers = _outlier_summary(df)
        assert len(outliers) > 0
        assert outliers[0]["column"] == "x"
        assert outliers[0]["count"] >= 1

    def test_report_tool_requires_loaded_df(self):
        from agent.tools import _df_store
        _df_store.pop("current", None)
        from agent.report_generator import generate_eda_report_tool
        result = generate_eda_report_tool.invoke({"report_title": "Test"})
        assert "No dataset" in result or "load" in result.lower()

    def test_report_tool_returns_saved_marker(self, tmp_path, monkeypatch):
        from agent.tools import _df_store
        _df_store["current"] = SAMPLE_DF.copy()
        import agent.report_generator as rg
        rg.SANDBOX_DIR = str(tmp_path)
        from agent.report_generator import generate_eda_report_tool
        result = generate_eda_report_tool.invoke({"report_title": "Marker Test"})
        assert "[REPORT_SAVED:" in result

    def test_report_handles_all_numeric_dataframe(self, tmp_path):
        from agent.report_generator import generate_eda_report
        df_num = pd.DataFrame({c: range(20) for c in "abcde"})
        path = generate_eda_report(df_num, output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_report_handles_all_categorical_dataframe(self, tmp_path):
        from agent.report_generator import generate_eda_report
        df_cat = pd.DataFrame({
            "color": ["red", "blue", "green"] * 10,
            "size":  ["S", "M", "L"] * 10,
        })
        path = generate_eda_report(df_cat, output_dir=str(tmp_path))
        assert os.path.exists(path)


# ══════════════════════════════════════════════════════════════════════════════
# 4. RAG-Enhanced Dataset Understanding
# ══════════════════════════════════════════════════════════════════════════════

class TestRAG:
    """All tests use force_tfidf=True so they run without Ollama."""

    def _make_rag(self):
        from agent.rag import DatasetRAG
        return DatasetRAG(force_tfidf=True)

    def test_indexing_produces_expected_chunk_count(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF, source_path="test.csv")
        # 1 overview + n columns + 1 correlations
        expected = 1 + len(SAMPLE_DF.columns) + 1
        assert len(rag.chunks) == expected

    def test_is_indexed_false_before_indexing(self):
        rag = self._make_rag()
        assert not rag.is_indexed()

    def test_is_indexed_true_after_indexing(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        assert rag.is_indexed()

    def test_retrieve_returns_requested_k_chunks(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        results = rag.retrieve("what is the salary distribution?", top_k=3)
        assert len(results) == 3

    def test_retrieve_salary_query_returns_salary_chunk(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        results = rag.retrieve("what is the average salary?", top_k=5)
        sources = [r.source for r in results]
        assert "salary" in sources, (
            f"Expected 'salary' chunk in top-5 results, got: {sources}"
        )

    def test_retrieve_returns_fewer_chunks_than_k_when_index_is_small(self):
        rag = self._make_rag()
        tiny_df = pd.DataFrame({"x": [1, 2, 3]})
        rag.index_dataframe(tiny_df)
        # 1 overview + 1 col + 1 corr = 3 total; asking for 10 should give 3
        results = rag.retrieve("anything", top_k=10)
        assert len(results) == 3

    def test_retrieve_as_context_returns_string(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        ctx = rag.retrieve_as_context("gender distribution", top_k=3)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_retrieve_as_context_contains_retrieved_header(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        ctx = rag.retrieve_as_context("salary stats")
        assert "RAG Context" in ctx or "Retrieved" in ctx

    def test_retrieve_as_context_empty_when_not_indexed(self):
        rag = self._make_rag()
        ctx = rag.retrieve_as_context("any query")
        assert ctx == ""

    def test_retrieve_returns_empty_list_when_not_indexed(self):
        rag = self._make_rag()
        results = rag.retrieve("any query")
        assert results == []

    def test_backend_is_tfidf_when_forced(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        assert rag.backend == "tfidf"

    def test_reindex_replaces_previous_chunks(self):
        rag = self._make_rag()
        df1 = pd.DataFrame({"alpha": [1, 2], "beta": [3, 4]})
        df2 = pd.DataFrame({"gamma": [5, 6]})
        rag.index_dataframe(df1)
        n1 = len(rag.chunks)
        rag.index_dataframe(df2)
        n2 = len(rag.chunks)
        # df2 has fewer columns → fewer chunks
        assert n2 < n1
        sources = [c.source for c in rag.chunks]
        assert "alpha" not in sources
        assert "gamma" in sources

    def test_chunk_text_contains_column_stats(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        salary_chunk = next(c for c in rag.chunks if c.source == "salary")
        assert "Mean" in salary_chunk.text or "mean" in salary_chunk.text.lower()
        assert "Min" in salary_chunk.text or "min" in salary_chunk.text.lower()

    def test_overview_chunk_contains_shape(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF, source_path="my_data.csv")
        overview = next(c for c in rag.chunks if c.source == "overview")
        assert "my_data.csv" in overview.text
        assert str(SAMPLE_DF.shape[0]) in overview.text

    def test_all_chunks_have_embeddings(self):
        rag = self._make_rag()
        rag.index_dataframe(SAMPLE_DF)
        for chunk in rag.chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) > 0
