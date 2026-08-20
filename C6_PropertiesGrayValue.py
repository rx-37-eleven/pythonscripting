"""
Properties-vs-Gray-Value plotting and regression script.

Reads a combined properties CSV (the same kind of file produced by
combine_csv_folder.py / C4: one row per Sample ID, with a "Gray Value"
column among the others). For every OTHER column that contains numeric
data (excluding X_COLUMN itself and the columns listed in
IGNORE_COLUMNS), builds a scatterplot of that column (y-axis) against
X_COLUMN (x-axis), fits a linear regression through the plotted points,
draws the fit line on the plot, and saves the figure as a PNG. A single
summary CSV of the regression results (slope, intercept, R^2, sample
count) is also written, one row per plotted column.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependencies: pandas, numpy, matplotlib.

This script is standalone — it does not import or depend on any other
script in this repository. It merely consumes combine_csv_folder.py's
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
# produced by combine_csv_folder.py (C4), with a "Gray Value" column
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
#    degree 1) over the same pairwise-valid points as that column's
#    plot. A column with fewer than 2 valid points is skipped (logged),
#    since a line can't be fit.
#  - Plot styling: one figure per y-column, points plus a fit line
#    spanning the plotted data's x-range, gridlines on, legend with the
#    fit equation and R^2. No fixed axis limits across plots (unlike
#    plot_stress_strain.py) since each column has its own unit/scale.
#  - Output: one PNG per plotted column
#    (<column>_vs_<X_COLUMN>_plot_<timestamp>.png; a literal "/" in a
#    column name is replaced with "-" so it can't be misread as a path
#    separator, otherwise column names are used as-is) plus one
#    regression_stats_<timestamp>.csv (columns: Column, Slope,
#    Intercept, R^2, N) covering every successfully plotted column.
#    Figures are saved only — no interactive plt.show() call. Never
#    overwrites an existing file of the same name.
# =====================================================================


def run() -> None:
    print(f"Reading input file '{INPUT_PATH}'...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {df.shape[0]} row(s) x {df.shape[1]} column(s)")

    if X_COLUMN not in df.columns:
        raise ValueError(f"X_COLUMN '{X_COLUMN}' not found in '{INPUT_PATH}'")

    x_all = pd.to_numeric(df[X_COLUMN], errors="coerce")

    candidate_columns = [c for c in df.columns if c != X_COLUMN and c not in IGNORE_COLUMNS]

    log: list[str] = []
    plotted = 0
    regression_rows: list[dict] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column in candidate_columns:
        y_all = pd.to_numeric(df[column], errors="coerce")
        if y_all.notna().sum() == 0:
            continue  # not a numeric column (e.g. Sample ID, Notes) — skip silently

        valid = x_all.notna() & y_all.notna()
        x = x_all[valid].to_numpy(dtype=float)
        y = y_all[valid].to_numpy(dtype=float)

        if len(x) < 2:
            log.append(
                f"SKIP '{column}': fewer than 2 rows with both {X_COLUMN} and "
                f"'{column}' present ({len(x)} found) — cannot fit a regression"
            )
            continue

        slope, intercept = np.polyfit(x, y, 1)
        predicted = slope * x + intercept
        residual_ss = float(np.sum((y - predicted) ** 2))
        total_ss = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss != 0 else float("nan")

        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.scatter(x, y, s=POINT_SIZE, color=POINT_COLOR, label=column)

        x_line = np.array([x.min(), x.max()])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=REGRESSION_LINE_COLOR,
            linewidth=REGRESSION_LINE_WIDTH,
            label=f"Fit: y = {slope:.4g}x + {intercept:.4g} (R^2={r_squared:.4g})",
        )

        ax.set_title(f"{column} vs {X_COLUMN}")
        ax.set_xlabel(X_COLUMN)
        ax.set_ylabel(column)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        safe_column = column.replace("/", "-")
        output_path = OUTPUT_DIR / f"{safe_column}_vs_{X_COLUMN}_plot_{timestamp}.png"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")
        fig.savefig(output_path)
        plt.close(fig)
        plotted += 1

        regression_rows.append(
            {
                "Column": column,
                "Slope": slope,
                "Intercept": intercept,
                "R^2": r_squared,
                "N": len(x),
            }
        )

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
        regression_rows, columns=["Column", "Slope", "Intercept", "R^2", "N"]
    )
    stats_path = OUTPUT_DIR / f"regression_stats_{timestamp}.csv"
    if stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {stats_path}")
    regression_df.to_csv(stats_path, index=False)

    print("\n--- Summary ---")
    print(f"  Columns plotted: {plotted} -> {[r['Column'] for r in regression_rows]}")
    print(f"  Regression stats file: {stats_path}")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
