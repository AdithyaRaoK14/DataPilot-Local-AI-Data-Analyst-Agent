# agent/report_generator.py
"""
Automated EDA Report Generator.

Produces a fully self-contained, single-file HTML report that can be opened
in any browser without internet access (Plotly.js loaded from CDN at open time,
all data is embedded as JSON).

Report sections
---------------
1. Key metrics row (rows, cols, duplicates, missing total)
2. Data quality score (completeness, uniqueness, consistency)
3. Column schema table (dtype, unique count, missing %)
4. Missing value bar chart + table
5. Numeric distributions — histograms in a 2-column grid
6. Box plots for outlier visualisation
7. Correlation heatmap (numeric cols only)
8. Categorical column bar charts (top-15 values, up to 6 columns)
9. Outlier summary table (IQR method)

Public API
----------
    generate_eda_report(df, title, output_dir) -> str   # returns HTML path
    generate_eda_report_tool(report_title)              # LangChain @tool wrapper
"""

import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from agent.config import cfg

SANDBOX_DIR = cfg.sandbox_dir
os.makedirs(SANDBOX_DIR, exist_ok=True)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_quality_score(df: pd.DataFrame) -> dict:
    """Return completeness / uniqueness / consistency / overall on 0–100 scale."""
    missing_ratio = df.isnull().mean().mean()
    completeness = round((1 - missing_ratio) * 100, 1)

    dup_ratio = df.duplicated().mean()
    uniqueness = round((1 - dup_ratio) * 100, 1)

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        inf_ratio = float(np.isinf(numeric.values).mean())
    else:
        inf_ratio = 0.0
    consistency = round((1 - inf_ratio) * 100, 1)

    overall = round((completeness + uniqueness + consistency) / 3, 1)
    return {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "overall": overall,
    }


def _outlier_summary(df: pd.DataFrame) -> list:
    """IQR-based outlier count per numeric column, sorted by severity."""
    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        pct = round(n_out / len(s) * 100, 2)
        rows.append({
            "column": col,
            "count": n_out,
            "pct": pct,
            "lower": round(lower, 4),
            "upper": round(upper, 4),
        })
    return sorted(rows, key=lambda x: x["pct"], reverse=True)


# ── Core report builder ───────────────────────────────────────────────────────

def generate_eda_report(
    df: pd.DataFrame,
    title: str = "Exploratory Data Analysis Report",
    output_dir: str = SANDBOX_DIR,
) -> str:
    """
    Build a self-contained HTML EDA report for *df*.
    Returns the absolute path to the saved .html file.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.io as pio
    except ImportError as exc:
        raise ImportError(f"Plotly is required: pip install plotly  ({exc})")

    report_id = uuid.uuid4().hex[:8]
    out_path = Path(output_dir) / f"eda_report_{report_id}.html"

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    quality = _compute_quality_score(df)
    outliers = _outlier_summary(df)

    # ── Build Plotly charts as embedded JSON ──────────────────────────────────

    charts: dict[str, str] = {}

    # 1. Missing value bar chart
    miss_pct = (df.isnull().mean() * 100).round(2).reset_index()
    miss_pct.columns = ["Column", "Missing %"]
    miss_pct = miss_pct.sort_values("Missing %", ascending=False)
    fig = px.bar(
        miss_pct, x="Column", y="Missing %",
        title="Missing value % by column",
        color="Missing %", color_continuous_scale="Reds",
        template="plotly_white",
    )
    fig.update_layout(margin=dict(t=40, b=60, l=60, r=20), height=320)
    charts["missing"] = pio.to_json(fig)

    # 2. Histograms
    hist_cols = numeric_cols[: cfg.report_max_numeric_columns]
    if hist_cols:
        ncols_grid = 2
        nrows_grid = max(1, (len(hist_cols) + 1) // 2)
        fig = make_subplots(
            rows=nrows_grid, cols=ncols_grid,
            subplot_titles=hist_cols,
        )
        for i, col in enumerate(hist_cols):
            r, c = divmod(i, ncols_grid)
            fig.add_trace(
                go.Histogram(
                    x=df[col].dropna(), name=col,
                    showlegend=False, marker_color="#636EFA",
                ),
                row=r + 1, col=c + 1,
            )
        fig.update_layout(
            title="Numeric column distributions",
            height=280 * nrows_grid,
            template="plotly_white",
            margin=dict(t=50, b=40, l=60, r=20),
        )
        charts["histograms"] = pio.to_json(fig)

    # 3. Box plots
    box_cols = numeric_cols[: cfg.report_max_numeric_columns]
    if box_cols:
        fig = go.Figure()
        for col in box_cols:
            fig.add_trace(go.Box(
                y=df[col].dropna(), name=col,
                boxpoints="outliers", marker_size=3,
            ))
        fig.update_layout(
            title="Box plots — outlier detection",
            template="plotly_white", height=480,
            margin=dict(t=50, b=60, l=60, r=20),
        )
        charts["boxplots"] = pio.to_json(fig)

    # 4. Correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().round(3)
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
            text=corr.values.round(2),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Correlation heatmap",
            template="plotly_white", height=500,
            margin=dict(t=50, b=60, l=120, r=20),
        )
        charts["correlation"] = pio.to_json(fig)

    # 5. Categorical bar charts
    cat_charts: dict[str, str] = {}
    for col in cat_cols[: cfg.report_max_cat_columns]:
        vc = df[col].value_counts().head(15).reset_index()
        vc.columns = [col, "count"]
        fig = px.bar(
            vc, x=col, y="count",
            title=f"Distribution: {col}",
            template="plotly_white",
            color="count", color_continuous_scale="Blues",
        )
        fig.update_layout(
            height=320, margin=dict(t=50, b=60, l=60, r=20)
        )
        safe_key = col.replace(" ", "_").replace("-", "_")
        cat_charts[safe_key] = pio.to_json(fig)

    # ── HTML helpers ──────────────────────────────────────────────────────────

    score_color = (
        "#2ecc71" if quality["overall"] >= 80
        else "#f39c12" if quality["overall"] >= 60
        else "#e74c3c"
    )

    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" class="chart"></div>'

    def _chart_init(div_id: str, json_str: str) -> str:
        return (
            f'var _f={json_str};'
            f'Plotly.newPlot("{div_id}",_f.data,_f.layout,{{responsive:true}});'
        )

    # Build schema table rows
    schema_rows = ""
    for col in df.columns:
        n_miss = int(df[col].isnull().sum())
        miss_pct_val = round(n_miss / len(df) * 100, 1)
        schema_rows += (
            f"<tr><td>{col}</td>"
            f"<td><code>{df[col].dtype}</code></td>"
            f"<td>{df[col].nunique():,}</td>"
            f"<td>{n_miss:,} ({miss_pct_val}%)</td></tr>"
        )

    # Build missing table rows
    miss_rows = ""
    for _, row in miss_pct.iterrows():
        bw = row["Missing %"]
        color = "#e74c3c" if bw > 20 else "#f39c12" if bw > 5 else "#27ae60"
        miss_rows += (
            f"<tr><td>{row['Column']}</td><td>{bw}%</td>"
            f"<td><div class='bar-bg'><div class='bar-fill' style='width:{min(bw,100)}%;background:{color}'></div></div></td></tr>"
        )

    # Build outlier table rows
    outlier_rows = ""
    for o in outliers[:15]:
        c = "#e74c3c" if o["pct"] > 10 else "#f39c12" if o["pct"] > 5 else "#27ae60"
        outlier_rows += (
            f"<tr><td>{o['column']}</td><td>{o['count']}</td>"
            f"<td style='color:{c};font-weight:600'>{o['pct']}%</td>"
            f"<td>[{o['lower']}, {o['upper']}]</td></tr>"
        )

    # Build chart sections and init scripts
    sections = ""
    init_js = ""

    def _section(title: str, key: str, src: dict = charts) -> None:
        nonlocal sections, init_js
        if key not in src:
            return
        div_id = f"ch_{key}"
        sections += f'<div class="card"><h2>{title}</h2>{_chart_div(div_id)}</div>'
        init_js += _chart_init(div_id, src[key])

    _section("Numeric distributions", "histograms")
    _section("Box plots — outlier detection", "boxplots")
    _section("Correlation heatmap", "correlation")

    # Categorical section groups all cat charts
    if cat_charts:
        cat_inner = ""
        for safe_key, json_str in cat_charts.items():
            div_id = f"ch_cat_{safe_key}"
            cat_inner += _chart_div(div_id)
            init_js += _chart_init(div_id, json_str)
        sections += f'<div class="card"><h2>Categorical distributions</h2>{cat_inner}</div>'

    # ── Assemble final HTML ───────────────────────────────────────────────────

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#222}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);color:#fff;padding:36px 40px;text-align:center}}
.header h1{{font-size:2rem;margin-bottom:6px}}
.header p{{opacity:.75;font-size:.9rem}}
.wrap{{max-width:1360px;margin:0 auto;padding:24px 20px}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:14px;margin:24px 0}}
.metric{{background:#fff;border-radius:10px;padding:18px 16px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.metric .val{{font-size:1.9rem;font-weight:700;color:#1a1a2e}}
.metric .lbl{{font-size:.8rem;color:#666;margin-top:4px}}
.card{{background:#fff;border-radius:10px;padding:22px;margin-bottom:22px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.card h2{{font-size:1.1rem;margin-bottom:14px;color:#1a1a2e;border-left:3px solid #0f3460;padding-left:10px}}
.chart{{width:100%;min-height:300px;margin-top:8px}}
table{{width:100%;border-collapse:collapse;font-size:.88rem}}
th{{background:#f4f6f9;padding:9px 12px;text-align:left;font-weight:600;color:#444;border-bottom:2px solid #ddd}}
td{{padding:8px 12px;border-bottom:1px solid #eee}}
tr:hover td{{background:#fafafa}}
code{{background:#e8f0fe;color:#0f3460;padding:2px 6px;border-radius:3px;font-size:.83rem}}
.qual-score{{font-size:3rem;font-weight:800;text-align:center;padding:12px 0;color:{score_color}}}
.qual-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}}
.qual-item{{text-align:center;padding:14px;background:#f9f9f9;border-radius:8px}}
.qual-item .qv{{font-size:1.5rem;font-weight:700}}
.bar-bg{{background:#eee;border-radius:4px;height:14px;width:180px}}
.bar-fill{{height:14px;border-radius:4px}}
footer{{text-align:center;padding:20px;color:#999;font-size:.8rem}}
</style>
</head>
<body>
<div class="header">
  <h1>📊 {title}</h1>
  <p>Generated {datetime.now().strftime("%B %d, %Y at %H:%M")} · Autonomous Data Analyst Agent · Report {report_id}</p>
</div>
<div class="wrap">

<div class="metrics">
  <div class="metric"><div class="val">{df.shape[0]:,}</div><div class="lbl">Rows</div></div>
  <div class="metric"><div class="val">{df.shape[1]}</div><div class="lbl">Columns</div></div>
  <div class="metric"><div class="val">{len(numeric_cols)}</div><div class="lbl">Numeric</div></div>
  <div class="metric"><div class="val">{len(cat_cols)}</div><div class="lbl">Categorical</div></div>
  <div class="metric"><div class="val">{df.duplicated().sum():,}</div><div class="lbl">Duplicates</div></div>
  <div class="metric"><div class="val">{df.isnull().sum().sum():,}</div><div class="lbl">Missing cells</div></div>
</div>

<div class="card">
  <h2>🏆 Data quality score</h2>
  <div class="qual-score">{quality['overall']}/100</div>
  <div class="qual-grid">
    <div class="qual-item"><div class="qv" style="color:#27ae60">{quality['completeness']}%</div><div>Completeness</div></div>
    <div class="qual-item"><div class="qv" style="color:#2980b9">{quality['uniqueness']}%</div><div>Uniqueness</div></div>
    <div class="qual-item"><div class="qv" style="color:#8e44ad">{quality['consistency']}%</div><div>Consistency</div></div>
  </div>
</div>

<div class="card">
  <h2>📋 Column schema</h2>
  <table><thead><tr><th>Column</th><th>Type</th><th>Unique values</th><th>Missing</th></tr></thead>
  <tbody>{schema_rows}</tbody></table>
</div>

<div class="card">
  <h2>❓ Missing values</h2>
  <table><thead><tr><th>Column</th><th>Missing %</th><th>Visual</th></tr></thead>
  <tbody>{miss_rows}</tbody></table>
  {_chart_div("ch_missing")}
</div>

{sections}

<div class="card">
  <h2>⚠️ Outlier summary — IQR method</h2>
  <table><thead><tr><th>Column</th><th>Count</th><th>% of data</th><th>Safe range [lower, upper]</th></tr></thead>
  <tbody>{outlier_rows}</tbody></table>
</div>

</div>
<footer>Autonomous Data Analyst Agent · Report ID: {report_id}</footer>
<script>
{_chart_init("ch_missing", charts["missing"])}
{init_js}
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


# ── LangChain tool wrapper ────────────────────────────────────────────────────

from langchain_core.tools import tool  # noqa: E402  (after stdlib imports)


@tool
def generate_eda_report_tool(report_title: str = "Exploratory Data Analysis Report") -> str:
    """
    Generate a comprehensive, self-contained HTML EDA report for the currently
    loaded dataset and save it to the sandbox directory.
    The report includes: dataset overview, data quality score, column schema,
    missing value analysis, numeric distributions, box plots, correlation
    heatmap, categorical distributions, and an outlier summary table.
    Use this when the user asks for an 'EDA report', 'full report',
    'export analysis', or 'save analysis to file'.
    Returns [REPORT_SAVED:<path>] on success so the UI can offer a download.
    """
    from agent.tools import _df_store

    if "current" not in _df_store:
        return (
            "No dataset loaded. "
            "Call load_and_inspect_data or load_table_from_db first."
        )

    df = _df_store["current"]
    try:
        path = generate_eda_report(df, title=report_title)
        return f"EDA report generated successfully.\n[REPORT_SAVED:{path}]"
    except Exception as exc:
        import traceback as tb
        return f"Error generating report: {exc}\n{tb.format_exc()}"
