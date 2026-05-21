# agent/tools.py
import re

from langchain_core.tools import tool
import os
import uuid
import traceback
import contextlib
import io
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')           # MUST be before any other matplotlib import


# Module-level dataframe store — shared across tool calls within a session
_df_store: dict = {}
SANDBOX_DIR = "sandbox"
os.makedirs(SANDBOX_DIR, exist_ok=True)


def get_df() -> pd.DataFrame:
    if "current" not in _df_store:
        raise ValueError(
            "No dataframe loaded. Call load_and_inspect_data first.")
    return _df_store["current"]


@tool
def load_and_inspect_data(file_path: str) -> str:
    """
    Load a CSV file from the given path and return a comprehensive summary.
    Always call this first before any analysis. Returns shape, columns,
    dtypes, missing value counts, and statistical summary.
    """
    try:
        df = pd.read_csv(file_path)
        _df_store["current"] = df
        _df_store["path"] = file_path

        potential_targets = [
            c for c in df.columns
            if df[c].nunique() <= 10 and df[c].dtype in ['int64', 'object']
        ]

        summary_parts = [
            "Dataset loaded successfully.",
            f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
            f"\nColumn names: {list(df.columns)}",
            f"\nData types:\n{df.dtypes.to_string()}",
            f"\nMissing values:\n{df.isnull().sum().to_string()}",
            f"\nNumerical summary:\n{df.describe().round(3).to_string()}",
        ]

        if potential_targets:
            for col in potential_targets[:2]:
                vc = df[col].value_counts()
                summary_parts.append(
                    f"\nValue counts for '{col}':\n{vc.to_string()}")

        return "\n".join(summary_parts)

    except FileNotFoundError:
        return f"Error: File not found at path: {file_path}"
    except Exception as e:
        return f"Error loading data: {str(e)}"


@tool
def execute_python_code(code: str) -> str:
    """
    Execute Python code for data analysis. The dataframe is already available as 'df'.
    Libraries available: pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns.
    Always use print() to show results.
    To save a plot use: plt.savefig(plot_path) then plt.close(). Never use plt.show().
    The variable plot_path is pre-set to a unique sandbox filename — use it directly.
    """
    try:
        df = get_df()
    except ValueError as e:
        return str(e)

    plot_filename = f"{SANDBOX_DIR}/plot_{uuid.uuid4().hex[:8]}.png"

    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "plot_path": plot_filename,
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "sorted": sorted,
            "reversed": reversed,
            "isinstance": isinstance,
            "type": type,
        }
    }

    stdout_capture = io.StringIO()
    output = ""

    try:
        forbidden_imports = [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
        ]
        for imp in forbidden_imports:
            code = code.replace(imp, "")

            # Only clean semicolons left behind from stripped imports
            code = re.sub(r'^\s*;\s*$', '', code, flags=re.MULTILINE)

            # Prevent the model from overwriting plot_path
        lines = code.splitlines()

        filtered_lines = []

        for line in lines:
            stripped = line.strip()

            # Remove lines like:
            # plot_path = 'something.png'
            if stripped.startswith("plot_path ="):
                continue

            filtered_lines.append(line)

        code = "\n".join(filtered_lines)

        with contextlib.redirect_stdout(stdout_capture):
            exec(code, namespace)
        output = stdout_capture.getvalue()

        # Detect plots saved directly by user code
        if os.path.exists(plot_filename):
            output += f"\n[PLOT_SAVED:{plot_filename}]"

        # Otherwise auto-save active matplotlib figures
        elif plt.get_fignums():
            plt.savefig(
                plot_filename,
                bbox_inches="tight",
                dpi=150,
                facecolor='white'
            )
            plt.close('all')
            output += f"\n[PLOT_SAVED:{plot_filename}]"

        if not output.strip():
            output = "Code executed successfully. No print() output — did you forget to print() your results?"

        return output

    except Exception as e:
        plt.close('all')
        tb = traceback.format_exc()
        return f"Execution Error:\n{str(e)}\n\nTraceback:\n{tb}"


@tool
def get_column_statistics(column_name: str) -> str:
    """
    Get detailed statistics for a specific column including distribution info,
    outlier detection using IQR method, and unique value analysis.
    """
    try:
        df = get_df()
    except ValueError as e:
        return str(e)

    if column_name not in df.columns:
        return f"Column '{column_name}' not found. Available columns: {list(df.columns)}"

    col = df[column_name]
    parts = [f"Statistics for column: '{column_name}'"]
    parts.append(f"Dtype: {col.dtype}")
    parts.append(f"Non-null count: {col.notna().sum()} / {len(col)}")
    parts.append(f"Null count: {col.isna().sum()}")

    if pd.api.types.is_numeric_dtype(col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = col[(col < lower) | (col > upper)]

        parts.append(
            f"\nDescriptive stats:\n{col.describe().round(4).to_string()}")
        parts.append(f"\nOutlier analysis (IQR method):")
        parts.append(f"  Lower bound: {lower:.4f}")
        parts.append(f"  Upper bound: {upper:.4f}")
        parts.append(
            f"  Outlier count: {len(outliers)} ({len(outliers)/len(col)*100:.2f}%)")
        if len(outliers) > 0:
            parts.append(
                f"  Sample outlier values: {sorted(outliers.values)[:5]}")
        parts.append(f"\nSkewness: {col.skew():.4f}")
        parts.append(f"Kurtosis: {col.kurtosis():.4f}")
    else:
        vc = col.value_counts()
        parts.append(f"\nUnique values: {col.nunique()}")
        parts.append(f"\nTop 10 value counts:\n{vc.head(10).to_string()}")

    return "\n".join(parts)


@tool
def get_correlation_analysis() -> str:
    """
    Compute full correlation matrix for numeric columns and identify
    the strongest positive and negative correlations.
    """
    try:
        df = get_df()
    except ValueError as e:
        return str(e)

    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        return "Not enough numeric columns for correlation analysis."

    corr = numeric_df.corr().round(4)

    corr_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    parts = [
        "Correlation Analysis",
        f"Numeric columns analyzed: {list(numeric_df.columns)}",
        f"\nFull correlation matrix:\n{corr.to_string()}",
        "\nTop 10 strongest correlations (by absolute value):"
    ]

    for col1, col2, val in corr_pairs[:10]:
        direction = "positive" if val > 0 else "negative"
        strength = "strong" if abs(
            val) > 0.7 else "moderate" if abs(val) > 0.4 else "weak"
        parts.append(
            f"  {col1} <-> {col2}: {val:.4f} ({strength} {direction})")

    return "\n".join(parts)
