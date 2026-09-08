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

Every point is categorized by the columns listed in LEGEND_COLUMNS
(e.g. ["H_V_F", "O_C"]). Their values are joined with ", " into one
legend entry per combination present — "Horz, OLC", "Vert, OLC",
"Vert, CLC" — with COLOR_COLUMN driving the point/line color (hex
codes from COLOR_PALETTE) and MARKER_COLUMN driving the marker shape.
Regressions can optionally be fit separately per color-column group,
per marker-column group, or per combination when both are enabled, via
the GROUP_REGRESSION_BY_COLOR / GROUP_REGRESSION_BY_MARKER flags below.
Fit lines and their equations in the legend are each independently
switchable (SHOW_FIT_LINES / SHOW_FIT_EQUATIONS_IN_LEGEND); the
regression stats CSV is written either way. Each column also gets one
combined 2x2-grid image with all four scale variants side by side.

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
INPUT_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/data_combined_20260908_124308.csv')

# Directory the plot PNGs and regression stats CSV are written into.
OUTPUT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code6_Outputs')

# Column used as the x-axis for every scatterplot/regression.
X_COLUMN = "Gray Value"

# Column names to ignore entirely as y-axis candidates, even though
# some of them may contain numeric-looking data (e.g. Date Tested is a
# numeric-looking date).
IGNORE_COLUMNS: set[str] = {"Width", "Thickness", "Date Tested"}

# ---------------------------------------------------------------------
# Legend / categories
# ---------------------------------------------------------------------

# Columns whose values describe each point, joined IN THIS ORDER with
# ", " to make that point's legend entry. With ["H_V_F", "O_C"] a row
# whose H_V_F is "Horz" and O_C is "OLC" is labeled "Horz, OLC". The
# legend lists one entry per combination actually plotted (e.g.
# "Horz, OLC", "Vert, OLC", "Vert, CLC") — there is no longer a
# separate color key and marker key block.
#
# Add a third column here and it simply becomes a third comma-separated
# part of every label; COLOR_COLUMN / MARKER_COLUMN below decide which
# two of these columns are shown visually. Every column listed here is
# also excluded from being plotted as a y-axis column.
LEGEND_COLUMNS: list[str] = ["H_V_F", "O_C"]

# Which of LEGEND_COLUMNS drives point/line COLOR. Each of its unique
# values takes the next color from COLOR_PALETTE below, assigned once
# per run so a value keeps the same color in every plot.
COLOR_COLUMN = "H_V_F"

# Which of LEGEND_COLUMNS drives MARKER SHAPE (and fit-line dash
# style). Set to None to draw every point with DEFAULT_MARKER and every
# fit line solid.
MARKER_COLUMN = "O_C"

# Point/line colors as hex codes, handed out in sorted order of
# COLOR_COLUMN's values. Blue and red are the primaries; black and
# orange are the alternates used once a third and fourth category
# appear. Reorder or extend this list to change the assignment — with
# only Horz and Vert present, Horz takes #0000FF and Vert #FF0000.
# More categories than colors cycles the list (and is logged).
COLOR_PALETTE: list[str] = [
    "#0000FF",  # blue
    "#FF0000",  # red
    "#000000",  # black
    "#FFA500",  # orange
]

# Marker shape per MARKER_COLUMN value (matplotlib marker codes).
# Lookup is exact first, then case-insensitive. A value not listed here
# takes the next unused marker from MARKER_FALLBACK_CYCLE and is logged
# once per run.
MARKER_MAP: dict[str, str] = {"OLC": "o", "CLC": "^"}
MARKER_FALLBACK_CYCLE: tuple[str, ...] = ("s", "D", "v", "P", "X", "*")

# Marker used when MARKER_COLUMN is None.
DEFAULT_MARKER = "o"

# Fit-line dash styles, handed out in sorted order of MARKER_COLUMN's
# values (so each marker category's fit line is also distinguishable
# when two share a color). Cycles if there are more categories than
# styles.
LINE_STYLE_CYCLE: tuple[str, ...] = ("-", "--", "-.", ":")

# Text substituted for a blank/missing value in any LEGEND_COLUMNS
# column, so such rows still get a complete label (e.g. "Horz, Unknown").
UNKNOWN_LABEL = "Unknown"

# ---------------------------------------------------------------------
# Regression display / grouping
# ---------------------------------------------------------------------

# When True, DRAW the fitted regression line(s) on each plot. When
# False the plots are scatter-only — the regressions are still computed
# and still written to the regression stats CSV, they're just not drawn.
SHOW_FIT_LINES = False

# When True, each drawn fit line also gets a legend entry with its
# equation and R^2 (e.g. "Fit [Horz, OLC]: y = 1.23x + 4.56
# (R^2=0.789)"). When False those fit entries are left out of the
# legend entirely, leaving just the category entries. Independent of
# SHOW_FIT_LINES — but with no lines drawn there is nothing to caption,
# so nothing is added to the legend either way.
SHOW_FIT_EQUATIONS_IN_LEGEND = False

# When True, fit a SEPARATE regression line for each unique
# COLOR_COLUMN value within a plot instead of one line through all of
# the column's points. Independent of GROUP_REGRESSION_BY_MARKER below
# — when BOTH are True, regressions are grouped by the full legend
# combination (e.g. "Vert, CLC") rather than by either column alone.
GROUP_REGRESSION_BY_COLOR = True

# When True, fit a SEPARATE regression line for each MARKER_COLUMN
# value within a plot. See GROUP_REGRESSION_BY_COLOR above for the
# combined-grouping behavior when both flags are True.
GROUP_REGRESSION_BY_MARKER = True

# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------

# Scatter point size (color and marker are determined per-point from
# COLOR_COLUMN / MARKER_COLUMN — see above).
POINT_SIZE = 20.0

# Regression line width, and the fallback line color used only when a
# fit line isn't tied to any COLOR_COLUMN group (grouping off, or
# grouped by MARKER_COLUMN alone) — otherwise a group's line is drawn
# in that group's own palette color. Set SHOW_FIT_LINES = False above
# to hide the lines rather than shrinking the width to zero.
REGRESSION_LINE_COLOR = "#000000"
REGRESSION_LINE_WIDTH = 1.5

# Figure resolution and size (matplotlib default figsize if None).
DPI = 150
FIGSIZE = None

# Figure size for each column's combined 2x2 grid image (None ->
# matplotlib default; that default is usually too small for four
# combined plots, hence the larger explicit default here).
GRID_FIGSIZE = (24, 12)

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
#    IGNORE_COLUMNS and LEGEND_COLUMNS.
#  - Ignored columns: Width, Thickness, and Date Tested are excluded by
#    exact, case-sensitive column name regardless of whether they'd
#    otherwise pass the numeric check (IGNORE_COLUMNS in CONFIG). Every
#    column in LEGEND_COLUMNS is always excluded as a y-candidate too,
#    since those are the categorical legend columns.
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
#  - Legend entries (LEGEND_COLUMNS): each point's values from those
#    columns are joined in the configured order with ", " into a single
#    label ("Horz, OLC"), and the legend lists one entry per label that
#    is actually plotted in that figure, drawn with that label's own
#    color and marker. Blank/missing values become UNKNOWN_LABEL so a
#    partially-labeled row still reads as a complete combination.
#  - Color-coding (COLOR_COLUMN): every unique value present in the
#    WHOLE input file (not just one column's valid rows) takes a hex
#    code from COLOR_PALETTE in sorted order, assigned once so the same
#    value always maps to the same color across every plot in the run.
#    If there are more values than colors the palette cycles (logged).
#  - Marker-coding (MARKER_COLUMN): MARKER_MAP is consulted first
#    exactly, then case-insensitively; any value it doesn't cover takes
#    the next unused marker from MARKER_FALLBACK_CYCLE, logged once per
#    run. MARKER_COLUMN = None draws every point with DEFAULT_MARKER.
#  - Regression grouping (GROUP_REGRESSION_BY_COLOR /
#    GROUP_REGRESSION_BY_MARKER): when both are False, one regression
#    is fit per (column, variant), labeled group "All". When exactly
#    one is True, one regression is fit per unique value of that
#    column instead (e.g. one line per COLOR_COLUMN value). When BOTH
#    are True, one regression is fit per full legend combination
#    present ("Vert, CLC"). A group's fit line spans only that group's
#    own plotted x-range (not the whole variant's), and is skipped
#    (logged) if it has fewer than 2 valid points. Every fitted group
#    gets its own row in the regression stats CSV (Group column) even
#    when SHOW_FIT_LINES is False.
#  - Fit display: SHOW_FIT_LINES controls whether the fitted lines are
#    drawn at all; SHOW_FIT_EQUATIONS_IN_LEGEND controls whether drawn
#    lines are captioned in the legend with their equation and R^2.
#    Neither affects what is computed or written to the stats CSV.
#  - Displayed rounding (legend text only — the CSV keeps full float
#    precision): slope/intercept to 4 significant figures (:.4g), R^2
#    to 3 decimal places (:.3f).
#  - Plot styling: one figure per (column, variant) pair, points plus
#    fit line(s) (see grouping above), gridlines on (major+minor when
#    either axis is log scaled). Legend placed outside the axes (to
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

    group_label is the group this fit line belongs to (e.g. "Horz, OLC"),
    or None for the single ungrouped fit. Slope/intercept are rounded
    to 4 significant figures and R^2 to 3 decimal places for display
    only — regression_stats keeps full precision.
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


def clean_category(value: object) -> str:
    """Normalize one categorical cell: trimmed text, or UNKNOWN_LABEL if blank."""
    if pd.isna(value):
        return UNKNOWN_LABEL
    text = str(value).strip()
    return text if text else UNKNOWN_LABEL


def build_legend_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of cleaned LEGEND_COLUMNS values plus their joined "Label" column.

    The label is every LEGEND_COLUMNS value for that row joined in the
    configured order with ", " — e.g. "Horz, OLC".
    """
    cleaned = pd.DataFrame({column: df[column].apply(clean_category) for column in LEGEND_COLUMNS})
    cleaned["Label"] = cleaned[LEGEND_COLUMNS].agg(", ".join, axis=1)
    return cleaned


def build_color_map(values: pd.Series, log: list[str]) -> dict[str, str]:
    """Assign each unique COLOR_COLUMN value a hex color from COLOR_PALETTE, in sorted order."""
    uniques = sorted(values.unique())
    if len(uniques) > len(COLOR_PALETTE):
        log.append(
            f"NOTE: {COLOR_COLUMN} has {len(uniques)} unique value(s) but COLOR_PALETTE has "
            f"only {len(COLOR_PALETTE)} color(s) — colors are reused (cycled). Add more hex "
            f"codes to COLOR_PALETTE to give every value its own color."
        )
    return {value: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, value in enumerate(uniques)}


def build_marker_map(values: pd.Series, log: list[str]) -> dict[str, str]:
    """Assign each unique MARKER_COLUMN value a matplotlib marker code.

    MARKER_MAP is consulted first exactly, then case-insensitively; any
    value it doesn't cover takes the next unused marker from
    MARKER_FALLBACK_CYCLE (cycling if it runs out) and is logged once.
    """
    lowercase_map = {key.strip().lower(): marker for key, marker in MARKER_MAP.items()}
    uniques = sorted(values.unique())

    resolved: dict[str, str] = {}
    unmapped: list[str] = []
    for value in uniques:
        if value in MARKER_MAP:
            resolved[value] = MARKER_MAP[value]
        elif value.strip().lower() in lowercase_map:
            resolved[value] = lowercase_map[value.strip().lower()]
        else:
            unmapped.append(value)

    for i, value in enumerate(unmapped):
        resolved[value] = MARKER_FALLBACK_CYCLE[i % len(MARKER_FALLBACK_CYCLE)]

    if unmapped:
        assignments = ", ".join(f"{value!r} -> '{resolved[value]}'" for value in unmapped)
        log.append(
            f"NOTE: {MARKER_COLUMN} has value(s) not listed in MARKER_MAP: {assignments} "
            f"(fallback markers). Add them to MARKER_MAP to choose their shapes."
        )
    return resolved


def build_line_style_map(values: pd.Series) -> dict[str, str]:
    """Assign each unique MARKER_COLUMN value a fit-line dash style, in sorted order."""
    uniques = sorted(values.unique())
    return {value: LINE_STYLE_CYCLE[i % len(LINE_STYLE_CYCLE)] for i, value in enumerate(uniques)}


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
    categories_all: pd.DataFrame,
    color_map: dict[str, str],
    marker_map: dict[str, str],
    line_style_map: dict[str, str],
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

    categories_all holds one cleaned column per LEGEND_COLUMNS entry
    plus the joined "Label" column. grouping_mode is one of "all",
    "color", "marker", "both" (see GROUP_REGRESSION_BY_COLOR /
    GROUP_REGRESSION_BY_MARKER). Returns (regression_stats rows, output
    PNG path or None if nothing was plotted for lack of valid points).
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
    categories = categories_all[valid]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    x_np = x.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=float)
    label_np = categories["Label"].to_numpy()
    color_np = categories[COLOR_COLUMN].to_numpy()
    marker_np = (
        categories[MARKER_COLUMN].to_numpy()
        if MARKER_COLUMN is not None
        else np.full(len(label_np), "", dtype=object)
    )

    # Each label is a full LEGEND_COLUMNS combination, so its color and
    # marker keys are constant within it — take them from its first row.
    label_keys: dict[str, tuple[str, str]] = {}
    for label, color_key, marker_key in zip(label_np, color_np, marker_np):
        label_keys.setdefault(label, (color_key, marker_key))

    legend_handles: list[object] = []
    legend_labels: list[str] = []

    for label in sorted(label_keys):
        color_key, marker_key = label_keys[label]
        color = color_map[color_key]
        marker = marker_map[marker_key] if MARKER_COLUMN is not None else DEFAULT_MARKER
        point_mask = label_np == label
        ax.scatter(x_np[point_mask], y_np[point_mask], s=POINT_SIZE, color=color, marker=marker)
        legend_handles.append(Line2D([0], [0], marker=marker, linestyle="none", color=color))
        legend_labels.append(label)

    if grouping_mode == "both":
        group_keys = sorted(label_keys)
    elif grouping_mode == "color":
        group_keys = sorted(set(color_np))
    elif grouping_mode == "marker":
        group_keys = sorted(set(marker_np))
    else:
        group_keys = ["All"]

    regression_rows: list[dict] = []
    fit_handles: list[object] = []
    fit_labels: list[str] = []

    for group_key in group_keys:
        if grouping_mode == "both":
            group_mask = label_np == group_key
            group_desc = group_key
            color_key, marker_key = label_keys[group_key]
            line_color = color_map[color_key]
            line_style = line_style_map[marker_key] if MARKER_COLUMN is not None else "-"
        elif grouping_mode == "color":
            group_mask = color_np == group_key
            group_desc = group_key
            line_color = color_map[group_key]
            line_style = "-"
        elif grouping_mode == "marker":
            group_mask = marker_np == group_key
            group_desc = group_key
            line_color = REGRESSION_LINE_COLOR
            line_style = line_style_map[group_key]
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

        # The fit is still computed (and recorded below) when
        # SHOW_FIT_LINES is off — only the drawing is skipped.
        if SHOW_FIT_LINES:
            x_line = (
                np.geomspace(xs.min(), xs.max(), 200) if log_x else np.linspace(xs.min(), xs.max(), 200)
            )
            x_line_fit = np.log(x_line) if log_x else x_line
            y_line_fit = slope * x_line_fit + intercept
            y_line = np.exp(y_line_fit) if log_y else y_line_fit

            (line_handle,) = ax.plot(
                x_line, y_line, color=line_color, linewidth=REGRESSION_LINE_WIDTH, linestyle=line_style
            )
            if SHOW_FIT_EQUATIONS_IN_LEGEND:
                fit_handles.append(line_handle)
                fit_labels.append(
                    format_equation_label(
                        kind, slope, intercept, r_squared, None if group_desc == "All" else group_desc
                    )
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

    # Fit entries (when shown) lead the legend, category entries follow.
    legend_handles = fit_handles + legend_handles
    legend_labels = fit_labels + legend_labels

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


def validate_config(df: pd.DataFrame) -> None:
    """Check the CONFIG block against the input file before any plotting."""
    if X_COLUMN not in df.columns:
        raise ValueError(f"X_COLUMN '{X_COLUMN}' not found in '{INPUT_PATH}'")
    if not LEGEND_COLUMNS:
        raise ValueError("LEGEND_COLUMNS must list at least one column")

    for column in LEGEND_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"LEGEND_COLUMNS entry '{column}' not found in '{INPUT_PATH}'")
    if COLOR_COLUMN not in LEGEND_COLUMNS:
        raise ValueError(f"COLOR_COLUMN '{COLOR_COLUMN}' must be one of LEGEND_COLUMNS {LEGEND_COLUMNS}")
    if MARKER_COLUMN is not None and MARKER_COLUMN not in LEGEND_COLUMNS:
        raise ValueError(f"MARKER_COLUMN '{MARKER_COLUMN}' must be one of LEGEND_COLUMNS {LEGEND_COLUMNS}")
    if not COLOR_PALETTE:
        raise ValueError("COLOR_PALETTE must contain at least one hex color code")


def resolve_grouping_mode() -> str:
    """Translate the GROUP_REGRESSION_BY_* flags into a single grouping mode."""
    group_by_marker = GROUP_REGRESSION_BY_MARKER and MARKER_COLUMN is not None
    if GROUP_REGRESSION_BY_COLOR and group_by_marker:
        return "both"
    if GROUP_REGRESSION_BY_COLOR:
        return "color"
    if group_by_marker:
        return "marker"
    return "all"


def run() -> None:
    print(f"Reading input file '{INPUT_PATH}'...")
    df = pd.read_csv(INPUT_PATH)
    print(f"  {df.shape[0]} row(s) x {df.shape[1]} column(s)")

    validate_config(df)

    x_all = pd.to_numeric(df[X_COLUMN], errors="coerce")

    log: list[str] = []

    categories_all = build_legend_labels(df)
    color_map = build_color_map(categories_all[COLOR_COLUMN], log)
    if MARKER_COLUMN is not None:
        marker_map = build_marker_map(categories_all[MARKER_COLUMN], log)
        line_style_map = build_line_style_map(categories_all[MARKER_COLUMN])
    else:
        marker_map = {}
        line_style_map = {}

    grouping_mode = resolve_grouping_mode()

    excluded_columns = IGNORE_COLUMNS | set(LEGEND_COLUMNS)
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
                categories_all,
                color_map,
                marker_map,
                line_style_map,
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

    print("\n--- Legend categories ---")
    print(f"  Legend label = {', '.join(LEGEND_COLUMNS)} (joined with ', ')")
    for value, color in sorted(color_map.items()):
        print(f"  {COLOR_COLUMN} '{value}' -> color {color}")
    for value, marker in sorted(marker_map.items()):
        print(f"  {MARKER_COLUMN} '{value}' -> marker '{marker}'")
    print(f"  Fit lines drawn: {SHOW_FIT_LINES}; fit equations in legend: {SHOW_FIT_EQUATIONS_IN_LEGEND}")

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
