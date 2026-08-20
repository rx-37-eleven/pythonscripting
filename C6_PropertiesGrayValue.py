"""
Properties-vs-Gray-Value plotting and regression script.

Reads a combined properties CSV (the same kind of file produced by
C4_combine_csv_folder.py / C4: one row per Sample ID, with a "Gray Value"
column among the others). For every OTHER column that contains numeric
data (excluding X_COLUMN itself and the columns listed in
IGNORE_COLUMNS), builds FOUR scatterplots of that column (y-axis)
against X_COLUMN (x-axis) — linear/linear, log-x, log-y, and log/log —
each with the regression type appropriate to its axis scaling fit as a
line through the plotted points:
    linear/linear -> linear fit:        y  = m*x + b
    log-x         -> logarithmic fit:   y  = m*ln(x) + b
    log-y         -> exponential fit:   ln(y) = m*x + b
    log/log       -> power-law fit:     ln(y) = m*ln(x) + b
Each is a straight-line (OLS) fit of the plot's own two displayed axes
(transformed to natural log wherever that axis is log-scaled), so the
fit line is straight in every plot as drawn. A single summary CSV of
the regression results (slope, intercept, R^2, sample count, one row
per plotted variant) is also written.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependencies: pandas, numpy, matplotlib.

This script is standalone — it does not import or depend on any other
script in this repository. It merely consumes C4_combine_csv_folder.py's
output file as input.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Path to the input CSV — a combined-output file of the same type
# produced by C4_combine_csv_folder.py (C4), with a "Gray Value" column
# among its others.
INPUT_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/data_combined_20260820_141442.csv')

# Directory the plot PNGs and regression stats CSV are written into.
OUTPUT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code6_Outputs')

# Column used as the x-axis for every scatterplot/regression.
X_COLUMN = "Gray Value"

# Column names to ignore entirely as y-axis candidates, even though
# some of them may contain numeric-looking data (e.g. Date Tested is a
# numeric-looking date).
IGNORE_COLUMNS: set[str] = {"Width", "Thickness", "Date Tested"}

# Scatter point styling.
POINT_COLOR = "tab:blue"
POINT_SIZE = 20.0

# Regression line styling.
REGRESSION_LINE_COLOR = "red"
REGRESSION_LINE_WIDTH = 1.5

# Figure resolution and size (matplotlib default figsize if None).
DPI = 150
FIGSIZE = None

# The four (name, log_x, log_y, filename_suffix, title_suffix) variants
# plotted for every column — not user-configurable, since all four are
# always produced together per the brief.
REGRESSION_KINDS: tuple[tuple[str, bool, bool, str, str], ...] = (
    ("linear", False, False, "", ""),
    ("logarithmic", True, False, "_logx", " (log-x)"),
    ("exponential", False, True, "_logy", " (log-y)"),
    ("power", True, True, "_loglog", " (log-log)"),
)

# ---------------------------------------------------------------------
# Resolved answers to the brief's open questions (captured here per the
# brief's "definition of done"):
#
#  - Input format: a single flat CSV with a header row (the C4 combined
#    output shape), not the multi-row-header wide/alternating layout
#    used by the zeroing/plotting scripts earlier in this series.
#  - "Column containing numerical data": determined by attempting
#    pd.to_numeric coercion on the column — if it has at least one
#    non-blank numeric value, it's eligible. Non-numeric columns (Sample
#    ID, Gray Value ID, Notes, etc.) are excluded automatically by this
#    check; no separate ID-column exclusion list is needed beyond
#    IGNORE_COLUMNS.
#  - Ignored columns: Width, Thickness, and Date Tested are excluded by
#    exact, case-sensitive column name regardless of whether they'd
#    otherwise pass the numeric check (IGNORE_COLUMNS in CONFIG).
#  - X_COLUMN itself is never also plotted as a y-column.
#  - Per-row handling: for each candidate y-column, only rows where both
#    X_COLUMN and that column are non-blank/numeric are used (pairwise,
#    not a single global filter) — a row missing from one column's plot
#    can still appear in another column's plot.
#  - Regression: ordinary least-squares straight line (np.polyfit,
#    degree 1) fit on whichever of x/y is log-scaled for that variant
#    (natural log), over the same pairwise-valid points plotted. Rows
#    are also required to be strictly positive on any axis that's log
#    scaled for that variant (log of zero/negative is undefined) — so a
#    column can have a different valid row count, and even be skipped
#    entirely, per variant. A variant with fewer than 2 valid points is
#    skipped (logged), since a line can't be fit.
#  - R^2 is always computed in the same (possibly log-transformed)
#    space the regression was actually fit in — i.e. it describes that
#    linear fit directly, consistent across all four variants.
#  - Displayed rounding (legend text only — the CSV keeps full float
#    precision): slope/intercept to 4 significant figures (:.4g), R^2
#    to 3 decimal places (:.3f).
#  - Plot styling: one figure per (column, variant) pair, points plus a
#    fit line spanning the plotted data's x-range (sampled log-spaced
#    when x is log-scaled, so the fit renders as a straight line on
#    that axis), gridlines on (major+minor when either axis is log
#    scaled), legend with the fit equation and R^2. No fixed axis
#    limits across plots (unlike C5_plot_stress_strain.py) since each
#    column has its own unit/scale.
#  - Output: one PNG per plotted (column, variant) pair
#    (<column>_vs_<X_COLUMN><variant_suffix>_plot_<timestamp>.png,
#    variant_suffix one of "", "_logx", "_logy", "_loglog"; a literal
#    "/" in a column name is replaced with "-" so it can't be misread
#    as a path separator, otherwise column names are used as-is) plus
#    one regression_stats_<timestamp>.csv (columns: Column, Regression
#    Type, Slope, Intercept, R^2, N) covering every successfully
#    plotted (column, variant) pair. Figures are saved only — no
#    interactive plt.show() call. Never overwrites an existing file of
#    the same name.
# =====================================================================


def format_equation_label(kind: str, slope: float, intercept: float, r_squared: float) -> str:
    """Build the plot legend's fit-equation label, in the space the regression was fit in.

    Slope/intercept are rounded to 4 significant figures and R^2 to 3
    decimal places for display only — regression_stats keeps full
    precision (see build_regression_row).
    """
    slope_str = f"{slope:.4g}"
    intercept_str = f"{intercept:.4g}"
    r2_str = f"{r_squared:.3f}"

    if kind == "linear":
        equation = f"y = {slope_str}x + {intercept_str}"
    elif kind == "logarithmic":
        equation = f"y = {slope_str}·ln(x) + {intercept_str}"
    elif kind == "exponential":
        equation = f"ln(y) = {slope_str}x + {intercept_str}"
    elif kind == "power":
        equation = f"ln(y) = {slope_str}·ln(x) + {intercept_str}"
    else:
        raise ValueError(f"Unknown regression kind: {kind}")

    return f"Fit: {equation} (R^2={r2_str})"


def fit_and_plot_variant(
    column: str,
    x_all: pd.Series,
    y_all: pd.Series,
    kind: str,
    log_x: bool,
    log_y: bool,
    filename_suffix: str,
    title_suffix: str,
    timestamp: str,
    log: list[str],
) -> dict | None:
    """Fit and plot one (column, scale-variant) pair. Returns a regression_stats row, or None if skipped."""
    valid = x_all.notna() & y_all.notna()
    if log_x:
        valid &= x_all > 0
    if log_y:
        valid &= y_all > 0

    x = x_all[valid].to_numpy(dtype=float)
    y = y_all[valid].to_numpy(dtype=float)

    if len(x) < 2:
        positivity_note = " and positive (required for a log-scaled axis)" if (log_x or log_y) else ""
        log.append(
            f"SKIP '{column}'{title_suffix}: fewer than 2 rows with both "
            f"{X_COLUMN} and '{column}' present{positivity_note} "
            f"({len(x)} found) — cannot fit a regression"
        )
        return None

    x_fit = np.log(x) if log_x else x
    y_fit = np.log(y) if log_y else y

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    predicted = slope * x_fit + intercept
    residual_ss = float(np.sum((y_fit - predicted) ** 2))
    total_ss = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss != 0 else float("nan")

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.scatter(x, y, s=POINT_SIZE, color=POINT_COLOR, label=column)

    x_line = np.geomspace(x.min(), x.max(), 200) if log_x else np.linspace(x.min(), x.max(), 200)
    x_line_fit = np.log(x_line) if log_x else x_line
    y_line_fit = slope * x_line_fit + intercept
    y_line = np.exp(y_line_fit) if log_y else y_line_fit

    ax.plot(
        x_line,
        y_line,
        color=REGRESSION_LINE_COLOR,
        linewidth=REGRESSION_LINE_WIDTH,
        label=format_equation_label(kind, slope, intercept, r_squared),
    )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_title(f"{column} vs {X_COLUMN}{title_suffix}")
    ax.set_xlabel(X_COLUMN)
    ax.set_ylabel(column)
    ax.grid(True, which="both" if (log_x or log_y) else "major")
    ax.legend()
    fig.tight_layout()

    safe_column = column.replace("/", "-")
    output_path = OUTPUT_DIR / f"{safe_column}_vs_{X_COLUMN}{filename_suffix}_plot_{timestamp}.png"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "Column": column,
        "Regression Type": kind,
        "Slope": slope,
        "Intercept": intercept,
        "R^2": r_squared,
        "N": len(x),
    }


def run() -> None:
    print(f"Reading input file '{INPUT_PATH}'...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {df.shape[0]} row(s) x {df.shape[1]} column(s)")

    if X_COLUMN not in df.columns:
        raise ValueError(f"X_COLUMN '{X_COLUMN}' not found in '{INPUT_PATH}'")

    x_all = pd.to_numeric(df[X_COLUMN], errors="coerce")

    candidate_columns = [c for c in df.columns if c != X_COLUMN and c not in IGNORE_COLUMNS]

    log: list[str] = []
    regression_rows: list[dict] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column in candidate_columns:
        y_all = pd.to_numeric(df[column], errors="coerce")
        if y_all.notna().sum() == 0:
            continue  # not a numeric column (e.g. Sample ID, Notes) — skip silently

        for kind, log_x, log_y, filename_suffix, title_suffix in REGRESSION_KINDS:
            row = fit_and_plot_variant(
                column, x_all, y_all, kind, log_x, log_y, filename_suffix, title_suffix, timestamp, log
            )
            if row is not None:
                regression_rows.append(row)

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    if not regression_rows:
        print("\nNo columns had enough numeric data to plot/regress — no output written.")
        return

    regression_df = pd.DataFrame(
        regression_rows, columns=["Column", "Regression Type", "Slope", "Intercept", "R^2", "N"]
    )
    stats_path = OUTPUT_DIR / f"regression_stats_{timestamp}.csv"
    if stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {stats_path}")
    regression_df.to_csv(stats_path, index=False)

    print("\n--- Summary ---")
    print(f"  Plots generated: {len(regression_rows)}")
    print(f"  Columns covered: {sorted({r['Column'] for r in regression_rows})}")
    print(f"  Regression stats file: {stats_path}")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
