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
per plotted (column, variant, group) triple) is also written.

Every point is additionally color-coded by its H_V_F_COLUMN value (a
distinct color per unique value, consistent across every plot in the
run) and marker-coded by its O_C_COLUMN value ("O" -> circle, "C" ->
triangle). Regressions can optionally be fit separately per H_V_F
group, per O_C group, or per (H_V_F, O_C) group when both are enabled,
via the GROUP_REGRESSION_BY_H_V_F / GROUP_REGRESSION_BY_O_C flags
below. Each column also gets one combined 2x2-grid image with all four
scale variants side by side.

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
from matplotlib.lines import Line2D

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

# Column identifying each point's category for color-coding. Every
# unique value found in this column gets its own color, consistent
# across every plotted column/variant in the run. Rows with a blank/
# missing value are grouped under the literal category "Unknown".
H_V_F_COLUMN = "H_V_F"

# Column identifying each point's marker shape. Rows with "O" (any
# case, surrounding whitespace ignored) are plotted as circles, rows
# with "C" as triangles. Any other value (including blank) falls back
# to MARKER_FALLBACK below and is grouped under the category "Other".
O_C_COLUMN = "O_C"
MARKER_FALLBACK = "x"

# When True, fit a SEPARATE regression line for each unique H_V_F value
# within a plot instead of one line through all of the column's points.
# Independent of GROUP_REGRESSION_BY_O_C below — when BOTH are True,
# regressions are grouped by the (H_V_F, O_C) combination rather than
# by either alone.
GROUP_REGRESSION_BY_H_V_F = False

# When True, fit a SEPARATE regression line for each O_C category
# ("O"/"C"/"Other") within a plot instead of one line through all of
# the column's points. See GROUP_REGRESSION_BY_H_V_F above for the
# combined-grouping behavior when both flags are True.
GROUP_REGRESSION_BY_O_C = False

# Scatter point size (color and marker are determined per-point from
# H_V_F_COLUMN / O_C_COLUMN — see above).
POINT_SIZE = 20.0

# Regression line width. Line color/style are determined by the active
# grouping (see GROUP_REGRESSION_BY_* above): a line for a group tied
# to an H_V_F value is drawn in that value's color, and a line for a
# group tied to the "C" O_C category is dashed; REGRESSION_LINE_COLOR
# is used whenever a line isn't tied to any H_V_F group (grouping off,
# or grouped by O_C alone).
REGRESSION_LINE_COLOR = "red"
REGRESSION_LINE_WIDTH = 1.5

# Figure resolution and size (matplotlib default figsize if None).
DPI = 150
FIGSIZE = None

# Figure size for each column's combined 2x2 grid image (None ->
# matplotlib default; that default is usually too small for four
# combined plots, hence the larger explicit default here).
GRID_FIGSIZE = (10, 8)

# The four (name, log_x, log_y, filename_suffix, title_suffix) variants
# plotted for every column — not user-configurable, since all four are
# always produced together per the brief.
REGRESSION_KINDS: tuple[tuple[str, bool, bool, str, str], ...] = (
    ("linear", False, False, "", ""),
    ("logarithmic", True, False, "_logx", " (log-x)"),
    ("exponential", False, True, "_logy", " (log-y)"),
    ("power", True, True, "_loglog", " (log-log)"),
)

# Marker shape per O_C category (matplotlib marker codes).
MARKER_MAP: dict[str, str] = {"O": "o", "C": "^"}

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
#    IGNORE_COLUMNS, H_V_F_COLUMN, and O_C_COLUMN.
#  - Ignored columns: Width, Thickness, and Date Tested are excluded by
#    exact, case-sensitive column name regardless of whether they'd
#    otherwise pass the numeric check (IGNORE_COLUMNS in CONFIG).
#    H_V_F_COLUMN and O_C_COLUMN are always excluded as y-candidates
#    too, since they're the categorical color/marker columns.
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
#    entirely, per variant. A variant (or a group within it, see below)
#    with fewer than 2 valid points is skipped (logged), since a line
#    can't be fit.
#  - R^2 is always computed in the same (possibly log-transformed)
#    space the regression was actually fit in — i.e. it describes that
#    linear fit directly, consistent across all four variants.
#  - Color-coding (H_V_F_COLUMN): every unique value present in the
#    WHOLE input file (not just one column's valid rows) gets a color
#    from matplotlib's tab10 palette (tab20 if there are more than 10
#    unique values, cycling with a repeated color if there are more
#    than 20), assigned once so the same value always maps to the same
#    color across every plot in the run. Blank/missing values are
#    grouped under the category "Unknown".
#  - Marker-coding (O_C_COLUMN): "O" (case-insensitive, trimmed) ->
#    circle, "C" -> triangle, anything else (including blank) -> the
#    MARKER_FALLBACK marker under the category "Other", logged once per
#    run if any such values are found.
#  - Regression grouping (GROUP_REGRESSION_BY_H_V_F /
#    GROUP_REGRESSION_BY_O_C): when both are False (default), one
#    regression is fit per (column, variant) as before, labeled group
#    "All". When exactly one is True, one regression is fit per unique
#    value of that column instead (e.g. one line per H_V_F value).
#    When BOTH are True, one regression is fit per unique (H_V_F, O_C)
#    combination present. A group's fit line spans only that group's
#    own plotted x-range (not the whole variant's), and is skipped
#    (logged) if it has fewer than 2 valid points. Every fitted group
#    gets its own row in the regression stats CSV (Group column).
#  - Displayed rounding (legend text only — the CSV keeps full float
#    precision): slope/intercept to 4 significant figures (:.4g), R^2
#    to 3 decimal places (:.3f).
#  - Plot styling: one figure per (column, variant) pair, points plus
#    fit line(s) (see grouping above), gridlines on (major+minor when
#    either axis is log scaled). Legend lists the fit line(s) followed
#    by a color-key entry per H_V_F value and a marker-key entry per
#    O_C category present in that plot, placed outside the axes (to
#    the right) since it can get long; figures are saved with
#    bbox_inches="tight" so the external legend isn't clipped. No fixed
#    axis limits across plots (unlike C5_plot_stress_strain.py) since
#    each column has its own unit/scale.
#  - Output: one PNG per plotted (column, variant) pair
#    (<column>_vs_<X_COLUMN><variant_suffix>_plot_<timestamp>.png,
#    variant_suffix one of "", "_logx", "_logy", "_loglog"; a literal
#    "/" in a column name is replaced with "-" so it can't be misread
#    as a path separator, otherwise column names are used as-is), one
#    combined 2x2-grid image per column with all four variants that
#    were actually produced (<column>_vs_<X_COLUMN>_grid_<timestamp>.png
#    — original upper-left, log-log upper-right, log-y lower-left,
#    log-x lower-right; a variant missing for that column, e.g. skipped
#    for lack of positive values, renders as an empty "Not available"
#    quadrant rather than omitting the grid image), plus one
#    regression_stats_<timestamp>.csv (columns: Column, Regression
#    Type, Group, Slope, Intercept, R^2, N) covering every successfully
#    fitted (column, variant, group) triple. Figures are saved only —
#    no interactive plt.show() call. Never overwrites an existing file
#    of the same name.
# =====================================================================


def format_equation_label(
    kind: str, slope: float, intercept: float, r_squared: float, group_label: str | None = None
) -> str:
    """Build a plot legend's fit-equation label, in the space the regression was fit in.

    group_label is the group this fit line belongs to (e.g. "H_V_F=X"),
    or None for the single ungrouped fit. Slope/intercept are rounded
    to 4 significant figures and R^2 to 3 decimal places for display
    only — regression_stats keeps full precision (see build_regression_row).
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

    prefix = "Fit" if group_label is None else f"Fit [{group_label}]"
    return f"{prefix}: {equation} (R^2={r2_str})"


def build_color_map(values: pd.Series) -> dict[str, tuple]:
    """Assign each unique category in values a distinct, stable color."""
    uniques = sorted(values.unique())
    cmap = plt.get_cmap("tab10") if len(uniques) <= 10 else plt.get_cmap("tab20")
    return {value: cmap(i % cmap.N) for i, value in enumerate(uniques)}


def resolve_oc_categories(
    oc_raw: pd.Series, log: list[str]
) -> tuple[pd.Series, pd.Series]:
    """Map O_C_COLUMN's raw values to (category, marker) Series aligned to oc_raw's index.

    Category is "O", "C", or "Other" (blank/anything else); marker is
    the matplotlib marker code for that category. Logs one note (not
    one per row) if any "Other" values are found.
    """
    cleaned = oc_raw.apply(lambda v: "" if pd.isna(v) else str(v).strip().upper())
    category = cleaned.apply(lambda v: v if v in MARKER_MAP else "Other")
    marker = category.apply(lambda v: MARKER_MAP.get(v, MARKER_FALLBACK))

    other_values = sorted({v if v else "(blank)" for v in cleaned[category == "Other"].unique()})
    if other_values:
        log.append(
            f"NOTE: {O_C_COLUMN} has value(s) other than 'O'/'C': {other_values} — "
            f"plotted with fallback marker '{MARKER_FALLBACK}' under category 'Other'"
        )
    return category, marker


def build_grid_image(
    column: str, kind_paths: dict[str, Path], timestamp: str, log: list[str]
) -> Path | None:
    """Combine a column's four scale-variant plots into one 2x2 grid image.

    Layout: original (linear) upper-left, log-log (power) upper-right,
    log-y (exponential) lower-left, log-x (logarithmic) lower-right. A
    variant missing for this column renders as an empty "Not available"
    quadrant. Returns None (no file written) if no variant is present.
    """
    if not kind_paths:
        return None

    grid_layout = (
        ("linear", 0, 0),
        ("power", 0, 1),
        ("exponential", 1, 0),
        ("logarithmic", 1, 1),
    )

    fig, axes = plt.subplots(2, 2, figsize=GRID_FIGSIZE, dpi=DPI)
    missing = 0
    for kind, row, col in grid_layout:
        ax = axes[row, col]
        path = kind_paths.get(kind)
        if path is not None:
            # aspect="auto" fills the whole cell — the source PNGs are wide
            # (external legend) relative to a square grid cell, and letting
            # imshow preserve their exact pixel aspect ratio would letterbox
            # each cell with large blank margins instead.
            ax.imshow(plt.imread(path), aspect="auto")
        else:
            missing += 1
            ax.text(0.5, 0.5, "Not available", ha="center", va="center")
        ax.axis("off")

    if missing:
        log.append(f"NOTE grid image for '{column}': {missing} of 4 variant(s) not available")

    fig.suptitle(f"{column} vs {X_COLUMN} — all variants")
    # axis("off") quadrants need none of matplotlib's default margin
    # reserved for tick/axis labels — reclaim it so the four images fill
    # the figure instead of floating in a sea of blank border.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.02, hspace=0.06)
    safe_column = column.replace("/", "-")
    output_path = OUTPUT_DIR / f"{safe_column}_vs_{X_COLUMN}_grid_{timestamp}.png"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def fit_and_plot_variant(
    column: str,
    x_all: pd.Series,
    y_all: pd.Series,
    hvf_all: pd.Series,
    oc_category_all: pd.Series,
    marker_all: pd.Series,
    color_map: dict[str, tuple],
    grouping_mode: str,
    kind: str,
    log_x: bool,
    log_y: bool,
    filename_suffix: str,
    title_suffix: str,
    timestamp: str,
    log: list[str],
) -> tuple[list[dict], Path | None]:
    """Fit and plot one (column, scale-variant) pair.

    grouping_mode is one of "all", "hvf", "oc", "both" (see
    GROUP_REGRESSION_BY_H_V_F / GROUP_REGRESSION_BY_O_C). Returns
    (regression_stats rows, output PNG path or None if nothing was
    plotted for lack of valid points).
    """
    valid = x_all.notna() & y_all.notna()
    if log_x:
        valid &= x_all > 0
    if log_y:
        valid &= y_all > 0

    if valid.sum() < 2:
        positivity_note = " and positive (required for a log-scaled axis)" if (log_x or log_y) else ""
        log.append(
            f"SKIP '{column}'{title_suffix}: fewer than 2 rows with both "
            f"{X_COLUMN} and '{column}' present{positivity_note} "
            f"({valid.sum()} found) — cannot fit a regression"
        )
        return [], None

    x = x_all[valid]
    y = y_all[valid]
    hvf = hvf_all[valid]
    oc_category = oc_category_all[valid]
    marker = marker_all[valid]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    hvf_np = hvf.to_numpy()
    oc_np = oc_category.to_numpy()
    marker_np = marker.to_numpy()

    for hvf_val, marker_char in sorted(set(zip(hvf_np, marker_np))):
        point_mask = (hvf_np == hvf_val) & (marker_np == marker_char)
        ax.scatter(
            x_np[point_mask], y_np[point_mask], s=POINT_SIZE, color=color_map[hvf_val], marker=marker_char
        )

    if grouping_mode == "both":
        group_keys = sorted(set(zip(hvf_np, oc_np)))
    elif grouping_mode == "hvf":
        group_keys = sorted(set(hvf_np))
    elif grouping_mode == "oc":
        group_keys = sorted(set(oc_np))
    else:
        group_keys = ["All"]

    regression_rows: list[dict] = []
    legend_handles: list[object] = []
    legend_labels: list[str] = []

    for group_key in group_keys:
        if grouping_mode == "both":
            hvf_val, oc_val = group_key
            group_mask = (hvf_np == hvf_val) & (oc_np == oc_val)
            group_desc = f"H_V_F={hvf_val}, O_C={oc_val}"
            line_color = color_map[hvf_val]
            line_style = "--" if oc_val == "C" else "-"
        elif grouping_mode == "hvf":
            hvf_val = group_key
            group_mask = hvf_np == hvf_val
            group_desc = f"H_V_F={hvf_val}"
            line_color = color_map[hvf_val]
            line_style = "-"
        elif grouping_mode == "oc":
            oc_val = group_key
            group_mask = oc_np == oc_val
            group_desc = f"O_C={oc_val}"
            line_color = REGRESSION_LINE_COLOR
            line_style = "--" if oc_val == "C" else "-"
        else:
            group_mask = np.ones(len(x_np), dtype=bool)
            group_desc = "All"
            line_color = REGRESSION_LINE_COLOR
            line_style = "-"

        xs = x_np[group_mask]
        ys = y_np[group_mask]
        if len(xs) < 2:
            log.append(
                f"SKIP '{column}'{title_suffix} group [{group_desc}]: fewer than 2 "
                f"valid points ({len(xs)}) — cannot fit a regression"
            )
            continue

        x_fit = np.log(xs) if log_x else xs
        y_fit = np.log(ys) if log_y else ys

        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        predicted = slope * x_fit + intercept
        residual_ss = float(np.sum((y_fit - predicted) ** 2))
        total_ss = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r_squared = 1.0 - residual_ss / total_ss if total_ss != 0 else float("nan")

        x_line = np.geomspace(xs.min(), xs.max(), 200) if log_x else np.linspace(xs.min(), xs.max(), 200)
        x_line_fit = np.log(x_line) if log_x else x_line
        y_line_fit = slope * x_line_fit + intercept
        y_line = np.exp(y_line_fit) if log_y else y_line_fit

        (line_handle,) = ax.plot(x_line, y_line, color=line_color, linewidth=REGRESSION_LINE_WIDTH, linestyle=line_style)
        legend_handles.append(line_handle)
        legend_labels.append(
            format_equation_label(kind, slope, intercept, r_squared, None if group_desc == "All" else group_desc)
        )

        regression_rows.append(
            {
                "Column": column,
                "Regression Type": kind,
                "Group": group_desc,
                "Slope": slope,
                "Intercept": intercept,
                "R^2": r_squared,
                "N": len(xs),
            }
        )

    for hvf_val in sorted(set(hvf_np)):
        legend_handles.append(Line2D([0], [0], marker="o", linestyle="none", color=color_map[hvf_val]))
        legend_labels.append(f"{H_V_F_COLUMN}: {hvf_val}")
    for oc_val in sorted(set(oc_np)):
        legend_handles.append(
            Line2D([0], [0], marker=MARKER_MAP.get(oc_val, MARKER_FALLBACK), linestyle="none", color="black")
        )
        legend_labels.append(f"{O_C_COLUMN}: {oc_val}")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")

    ax.set_title(f"{column} vs {X_COLUMN}{title_suffix}")
    ax.set_xlabel(X_COLUMN)
    ax.set_ylabel(column)
    ax.grid(True, which="both" if (log_x or log_y) else "major")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1.0))

    safe_column = column.replace("/", "-")
    output_path = OUTPUT_DIR / f"{safe_column}_vs_{X_COLUMN}{filename_suffix}_plot_{timestamp}.png"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return regression_rows, output_path


def run() -> None:
    print(f"Reading input file '{INPUT_PATH}'...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {df.shape[0]} row(s) x {df.shape[1]} column(s)")

    if X_COLUMN not in df.columns:
        raise ValueError(f"X_COLUMN '{X_COLUMN}' not found in '{INPUT_PATH}'")
    if H_V_F_COLUMN not in df.columns:
        raise ValueError(f"H_V_F_COLUMN '{H_V_F_COLUMN}' not found in '{INPUT_PATH}'")
    if O_C_COLUMN not in df.columns:
        raise ValueError(f"O_C_COLUMN '{O_C_COLUMN}' not found in '{INPUT_PATH}'")

    x_all = pd.to_numeric(df[X_COLUMN], errors="coerce")

    log: list[str] = []

    hvf_all = df[H_V_F_COLUMN].apply(lambda v: "Unknown" if pd.isna(v) or str(v).strip() == "" else str(v).strip())
    color_map = build_color_map(hvf_all)
    oc_category_all, marker_all = resolve_oc_categories(df[O_C_COLUMN], log)

    if GROUP_REGRESSION_BY_H_V_F and GROUP_REGRESSION_BY_O_C:
        grouping_mode = "both"
    elif GROUP_REGRESSION_BY_H_V_F:
        grouping_mode = "hvf"
    elif GROUP_REGRESSION_BY_O_C:
        grouping_mode = "oc"
    else:
        grouping_mode = "all"

    excluded_columns = IGNORE_COLUMNS | {H_V_F_COLUMN, O_C_COLUMN}
    candidate_columns = [c for c in df.columns if c != X_COLUMN and c not in excluded_columns]

    regression_rows: list[dict] = []
    grid_images_written = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column in candidate_columns:
        y_all = pd.to_numeric(df[column], errors="coerce")
        if y_all.notna().sum() == 0:
            continue  # not a numeric column (e.g. Sample ID, Notes) — skip silently

        kind_paths: dict[str, Path] = {}
        for kind, log_x, log_y, filename_suffix, title_suffix in REGRESSION_KINDS:
            rows, output_path = fit_and_plot_variant(
                column,
                x_all,
                y_all,
                hvf_all,
                oc_category_all,
                marker_all,
                color_map,
                grouping_mode,
                kind,
                log_x,
                log_y,
                filename_suffix,
                title_suffix,
                timestamp,
                log,
            )
            regression_rows.extend(rows)
            if output_path is not None:
                kind_paths[kind] = output_path

        if build_grid_image(column, kind_paths, timestamp, log) is not None:
            grid_images_written += 1

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
        regression_rows, columns=["Column", "Regression Type", "Group", "Slope", "Intercept", "R^2", "N"]
    )
    stats_path = OUTPUT_DIR / f"regression_stats_{timestamp}.csv"
    if stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {stats_path}")
    regression_df.to_csv(stats_path, index=False)

    print("\n--- Summary ---")
    print(f"  Plots generated: {len(regression_rows)}")
    print(f"  Grid images generated: {grid_images_written}")
    print(f"  Columns covered: {sorted({r['Column'] for r in regression_rows})}")
    print(f"  Regression stats file: {stats_path}")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
