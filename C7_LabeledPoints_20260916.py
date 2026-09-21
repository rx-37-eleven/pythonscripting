"""
Properties-vs-Gray-Value plotting with NUMBERED data points (C7).

Same plots, styling and regressions as C6_PropertiesGrayValue.py, with
one addition aimed squarely at chasing down outliers: every plotted
point is annotated with its own number. By default that number is the
point's ROW NUMBER in the input CSV (the first data row is point 1,
the second is point 2, ...); turn USE_POINT_NUMBER_COLUMN on and it is
read from the file's own "Point Number" column instead, so numbers
handed out by an earlier run survive the file being re-sorted or cut
down. Spot a stray point on a chart, read its number, and
look that number up in the point key CSV this script writes alongside
the plots to get straight back to the Sample ID, the category columns,
the gray value and every property value on that row.

Three kinds of output make that round trip:
  1. The plots themselves, with each point captioned by its number.
  2. point_key_<timestamp>.csv — one row per input data row: its point
     number, the line it sits on in the input file, its Sample ID, its
     legend/category values, its designation, its X value, and its
     value in every plotted y-column. This is the "what is point 17?"
     lookup table.
  3. point_residuals_<timestamp>.csv (EXPORT_RESIDUAL_OUTLIERS) — one
     row per point per fitted regression, with that point's residual
     from its group's fit line and that residual expressed in standard
     deviations, flagged when it exceeds OUTLIER_SIGMA. This is the
     "which points ARE the outliers?" table: sort it by Abs Std
     Residual and the worst offenders, with their Sample IDs, are at
     the top.

A handful of points can be singled out for a closer look: set
HIGHLIGHT_CSV_PATH to a second CSV — in practice a copy of the input
file with all but the rows of interest deleted — and every matching
point is drawn with a thin green ring around it on every plot. The
color, thickness and diameter of the ring are all config options, and
nothing that is computed changes: the regressions and the CSVs come
out exactly as they would without it.

Everything else below is C6's behavior, reproduced here so this script
stands alone.

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
codes pinned in COLOR_MAP, else handed out from COLOR_PALETTE) and
MARKER_COLUMN driving the marker shape.
Regressions can optionally be fit separately per color-column group,
per marker-column group, or per combination when both are enabled, via
the GROUP_REGRESSION_BY_COLOR / GROUP_REGRESSION_BY_MARKER flags below.

Samples whose Sample ID carries two underscores ("VT3_1_4") are also
split into two designation groups by that ID's middle field — the "1"
samples and the "2" samples. Designation "1" (and any row without a
designation at all) is drawn with a SOLID-filled symbol and "2" with a
HOLLOW one, in that point's usual category color and marker shape. The
main regression stats CSV is unchanged by this; a SECOND CSV
(regression_stats_by_designation_<timestamp>.csv) repeats every
(column, variant) fit grouped by designation alone — one row for the
"1" samples, one for the "2" samples, plus a pooled "All" row.
Fit lines and their equations in the legend are each independently
switchable (SHOW_FIT_LINES / SHOW_FIT_EQUATIONS_IN_LEGEND); the
regression stats CSV is written either way. Each column also gets one
combined 2x2-grid image with all four scale variants side by side.

All four variants are always fitted and always reach the stats CSV.
Two independent toggles control which picture FILES are written:
SAVE_LOG_PLOTS keeps the three log variants as their own PNGs (the
base plot is always kept), and SAVE_GRID_IMAGE writes the combined
2x2 grid. With both off, the log variants are fitted but never drawn.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependencies: pandas, numpy, matplotlib.

This script is standalone — it does not import or depend on any other
script in this repository (C6 included). It merely consumes
C4_combine_csv_folder.py's output file as input, exactly as C6 does, so
the two can be run over the same file and compared point for point.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Path to the input CSV — a combined-output file of the same type
# produced by C4_combine_csv_folder.py (C4), with a "Gray Value" column
# among its others.
INPUT_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/data_combined_20260908_124308.csv')

# Directory the plot PNGs, point key and regression stats CSVs are
# written into. Kept separate from C6's output folder so the two runs'
# files never mingle.
OUTPUT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code7_Outputs')

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

# Which of LEGEND_COLUMNS drives point/line COLOR. Its values are
# colored by COLOR_MAP / COLOR_PALETTE below, assigned once per run so
# a value keeps the same color in every plot.
COLOR_COLUMN = "H_V_F"

# Which of LEGEND_COLUMNS drives MARKER SHAPE (and fit-line dash
# style). Set to None to draw every point with DEFAULT_MARKER and every
# fit line solid.
MARKER_COLUMN = "O_C"

# Hex color pinned to a specific COLOR_COLUMN value. Anything pinned
# here keeps that exact color no matter what other values exist in the
# file — this is what stops a category that happens to sort earlier
# alphabetically (e.g. "F") from pushing Horz and Vert down the
# palette. Set to {} to hand every value out of COLOR_PALETTE
# instead.
COLOR_MAP: dict[str, str] = {
    "Horz": "#0000FF",  # blue
    "Vert": "#FF0000",  # red
}

# Colors for COLOR_COLUMN values NOT pinned in COLOR_MAP, handed out in
# sorted order from the first entry not already used by a pin. With the
# pins above, a third category such as "F" takes black and a fourth
# takes orange. More unpinned values than remaining colors cycles the
# list (and is logged).
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
# Sub-sample designation (the "1" / "2" inside a two-underscore Sample ID)
# ---------------------------------------------------------------------

# Column holding the sample identifier the designation is read out of
# (the C4 combined output calls it "Sample ID"). If the file has no such
# column, every row is simply left undesignated and the script behaves
# exactly as it did before this feature existed.
SAMPLE_ID_COLUMN = "Sample ID"

# The designation is only read from Sample IDs with EXACTLY this many
# underscores — "VT3_1_4" (two) yields a designation, "VT3_4" (one) and
# "FL15" (none) do not. Rows without one are drawn solid and are left
# out of the per-designation stats CSV entirely.
DESIGNATION_UNDERSCORE_COUNT = 2

# Which underscore-separated field of such a Sample ID IS the
# designation, 0-based: "VT3_1_4" -> fields ("VT3", "1", "4"), so index
# 1 is the middle "1". That middle field is the part C4 keeps when it
# derives the Gray Value ID ("VT3_1"), i.e. the two groups here are the
# samples sharing a "_1_" scan and those sharing a "_2_" scan.
DESIGNATION_FIELD_INDEX = 1

# Designations drawn as HOLLOW (open) symbols. Every other designation,
# and every row without one, is drawn solid-filled.
HOLLOW_DESIGNATIONS: tuple[str, ...] = ("2",)

# Outline width of a hollow marker (solid markers are unaffected).
HOLLOW_MARKER_EDGE_WIDTH = 1.0

# Append the designation to that point's legend entry, so the solid and
# hollow halves of a category read as "Horz, OLC (1)" and
# "Horz, OLC (2)". Rows with no designation keep the plain label.
SHOW_DESIGNATION_IN_LEGEND = True

# Write the SECOND regression stats CSV — same (column, variant) fits,
# but grouped by designation alone (all "1" samples vs all "2"
# samples), ignoring H_V_F / O_C. The main regression stats CSV is
# unaffected by this and keeps the LEGEND_COLUMNS grouping below.
WRITE_DESIGNATION_STATS_CSV = True

# Include a pooled "All" row (every plotted point of that column and
# variant, designated or not) in that second CSV, for comparison
# against its "1" and "2" rows.
INCLUDE_ALL_ROW_IN_DESIGNATION_STATS = True

# ---------------------------------------------------------------------
# Point numbering (this script's reason for existing)
# ---------------------------------------------------------------------

# Draw each point's number next to it. Turn this off to get C6's plain
# plots while still writing the point key and residual CSVs.
SHOW_POINT_LABELS = True

# Take each point's number from a COLUMN of the input CSV instead of
# counting the rows.
#
#   False (default) -> the script numbers the points itself, counting
#                      the file's data rows from POINT_LABEL_START.
#   True            -> the number is read from POINT_NUMBER_COLUMN, so
#                      a point keeps the number it was given in an
#                      earlier run even after the file has been sorted,
#                      re-cut or had rows deleted. Feed an earlier run's
#                      point_key CSV (or the input file with that
#                      column pasted in) back through this script and
#                      the numbers on the plots stay the ones you have
#                      already been reading off them.
#
# With it on, anything the column cannot supply falls back to that
# row's counted number and is reported under "Skipped / logged items":
# the column missing from the file altogether, a blank cell, or a value
# that is not a number. Whether it is on or not, POINT_NUMBER_COLUMN is
# an identifier rather than a property, so it is never plotted as a
# y-column when the input file happens to carry it.
USE_POINT_NUMBER_COLUMN = False
POINT_NUMBER_COLUMN = "Point Number"

# Number given to the FIRST DATA ROW of the input CSV when the script
# is numbering the points itself (USE_POINT_NUMBER_COLUMN = False, or a
# row the column could not supply). With the default 1, point numbers
# are "1 = first data row", and a point's line in the file (as a
# spreadsheet shows it, header on line 1) is its number + 1 — the point
# key CSV lists both, so neither has to be worked out by hand.
POINT_LABEL_START = 1

# Point-number text size, and its offset from the point in typographic
# points (right and up by default, so the number sits clear of the
# marker).
POINT_LABEL_FONTSIZE = 7
POINT_LABEL_OFFSET: tuple[float, float] = (4.0, 4.0)

# True  -> each number takes its own point's category color (so Horz
#          numbers are blue and Vert numbers red, matching the points).
# False -> every number is drawn in POINT_LABEL_COLOR.
POINT_LABEL_COLOR_FROM_POINT = True
POINT_LABEL_COLOR = "#000000"

# ---------------------------------------------------------------------
# Outlier tracing
# ---------------------------------------------------------------------

# Write point_residuals_<timestamp>.csv: every plotted point's distance
# from its own group's fit line, in the space that regression was fit
# in, together with that point's number and Sample ID. Sort the file by
# "Abs Std Residual" (descending) to rank candidate outliers.
EXPORT_RESIDUAL_OUTLIERS = True

# Residuals for every scale variant (linear, log-x, log-y, log-log), or
# only for BASE_REGRESSION_KIND? The base plot alone keeps the file
# small and is usually what you are looking at; True is for when the
# outlier only stands out on a log plot.
RESIDUALS_FOR_ALL_VARIANTS = False

# A point is flagged ("Outlier" = True) when its residual is at least
# this many standard deviations from its group's fit line. The residual
# and its standardized value are written for EVERY point either way —
# this only sets the flag column, so raising or lowering it never hides
# a row.
OUTLIER_SIGMA = 2.0

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

# The combined 2x2 grid image has no configurable figure size: it is
# derived from the pixel dimensions of the four variant PNGs actually
# going into it, so each panel is reproduced at its true aspect ratio
# instead of being stretched to fit a guessed figure shape. GRID_TITLE_IN
# is the vertical space (inches) reserved above the panels for the
# grid's suptitle.
GRID_TITLE_IN = 0.5

# The four (name, log_x, log_y, filename_suffix, title_suffix) variants
# fitted for every column. All four are ALWAYS fitted and always appear
# in the regression stats CSV; SAVE_LOG_PLOTS / SAVE_GRID_IMAGE below
# control only which picture files are written.
REGRESSION_KINDS: tuple[tuple[str, bool, bool, str, str], ...] = (
    ("linear", False, False, "", ""),
    ("logarithmic", True, False, "_logx", " (log-x)"),
    ("exponential", False, True, "_logy", " (log-y)"),
    ("power", True, True, "_loglog", " (log-log)"),
)

# Which REGRESSION_KINDS entry is the "base" plot — the one always
# saved, and the only one saved when SAVE_LOG_PLOTS is False.
# "linear" is the untransformed plot (linear x, linear y).
BASE_REGRESSION_KIND = "linear"

# Save the combined 2x2 grid image (all four scale variants in one
# picture) for each column?
#
#   True  -> one <column>_vs_<X_COLUMN>_grid_<timestamp>.png per column.
#   False -> no grid file.
SAVE_GRID_IMAGE = False

# Save the three LOG plots (log-x, log-y, log-log) as their own
# individual PNGs?
#
#   True  -> four standalone PNGs per column: the base plot plus one
#            per log variant.
#   False -> one standalone PNG per column: the base plot only
#            (BASE_REGRESSION_KIND).
#
# The base plot is always saved; this toggle is only about the log
# variants' own files.
SAVE_LOG_PLOTS = False

# NOTE — both toggles above are about which FILES are written. All four
# variants are ALWAYS fitted and always contribute their rows to the
# regression stats CSV, whatever the toggles say.
#
# The two combine like this:
#
#   grid=True,  log=True   -> 4 plot PNGs + grid, per column
#   grid=True,  log=False  -> 1 plot PNG  + grid, per column. The log
#                             plots are still drawn (the grid is
#                             composited from the rendered PNGs) but
#                             into a temporary directory that is
#                             deleted once the grid is built.
#   grid=False, log=True   -> 4 plot PNGs, no grid
#   grid=False, log=False  -> 1 plot PNG, no grid. Nothing needs the log
#                             variants rendered, so their figures are
#                             never drawn at all — only fitted. This is
#                             the fastest combination.

# ---------------------------------------------------------------------
# Circled points (a second CSV naming the points to ring)
# ---------------------------------------------------------------------

# Optional second CSV naming points to CIRCLE on every plot. The
# intended workflow: take a copy of the input file, delete every row
# but the handful you want to look at, and point this at that copy —
# those rows' points are then drawn with a thin ring around them in
# every plot, while every other point is drawn exactly as before.
#
# Set to None (the default) to switch the feature off entirely: no file
# is read and no circles are drawn. Only HIGHLIGHT_MATCH_COLUMN is read
# out of the file, so the copy can keep all of its other columns or
# none of them — deleting rows is enough, nothing else has to be edited.
HIGHLIGHT_CSV_PATH: Path | None = None
# HIGHLIGHT_CSV_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/points_to_circle.csv')

# The column matched between the two files to decide which points get
# circled. It must exist in BOTH the input CSV and the highlight CSV.
# "Sample ID" is the natural key: it names the same specimen no matter
# how either file is sorted, filtered or regenerated. Values are
# compared trimmed and case-insensitively; a highlight row matching no
# input row is reported under "Skipped / logged items" rather than
# stopping the run.
HIGHLIGHT_MATCH_COLUMN = "Sample ID"

# Circle appearance. The diameter is in typographic points — a SCREEN
# size, like a font size, not data units — so the ring is the same
# circle on the linear plot and on the log-log one, and stays centered
# on its point whatever the axes do. Keep it comfortably bigger than
# the marker it rings: POINT_SIZE is an AREA in points^2, so its marker
# is about sqrt(POINT_SIZE) ~ 4.5 points across at the default 20.
HIGHLIGHT_CIRCLE_COLOR = "#00A000"      # green
HIGHLIGHT_CIRCLE_DIAMETER = 14.0        # points across (screen size)
HIGHLIGHT_CIRCLE_LINEWIDTH = 0.8        # ring thickness in points

# Give the circled points one legend entry of their own, drawn as the
# ring itself and listed after the category entries.
SHOW_HIGHLIGHT_IN_LEGEND = True
HIGHLIGHT_LEGEND_LABEL = "Circled (highlight CSV)"

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
#  - Color-coding (COLOR_COLUMN): colors are resolved over every unique
#    value present in the WHOLE input file (not just one column's valid
#    rows), assigned once so the same value always maps to the same
#    color across every plot in the run. COLOR_MAP pins win first
#    (exact match, then case-insensitive); the remaining values take
#    COLOR_PALETTE colors that no pin claimed, in sorted order. Pinning
#    Horz and Vert is deliberate: without it a category that merely
#    sorts earlier (e.g. "F") consumes the first color and shifts every
#    other category down the palette, even on charts where that
#    category has no points. If the unpinned values outnumber the
#    unused colors the palette cycles (logged).
#  - Sub-sample designation (SAMPLE_ID_COLUMN,
#    DESIGNATION_UNDERSCORE_COUNT, DESIGNATION_FIELD_INDEX): a Sample ID
#    is designated only when it has exactly two underscores, in which
#    case the middle underscore-separated field ("1" in "VT3_1_4") is
#    its designation — that is the field C4 keeps when deriving the Gray
#    Value ID ("VT3_1"), so the two groups are the samples sharing a
#    "_1_" scan and those sharing a "_2_" scan. IDs with any other
#    number of underscores ("VT3_4", "FL15") are undesignated: they are
#    drawn solid and contribute to the pooled "All" row of the
#    per-designation CSV but to neither the "1" nor the "2" row. A
#    designation listed in HOLLOW_DESIGNATIONS ("2") is drawn as an open
#    symbol — same color, same marker shape, no fill — and every other
#    point stays solid. SHOW_DESIGNATION_IN_LEGEND appends it to that
#    series' legend entry ("Horz, OLC (2)").
#  - Per-designation regressions (WRITE_DESIGNATION_STATS_CSV): a
#    SECOND, independent grouping, written to its own CSV
#    (regression_stats_by_designation_<timestamp>.csv: Column,
#    Regression Type, Designation, Slope, Intercept, R^2, N). It fits
#    the same (column, variant) pairs over the same plotted points,
#    grouped by designation ALONE — H_V_F and O_C are ignored there —
#    plus a pooled "All" row when
#    INCLUDE_ALL_ROW_IN_DESIGNATION_STATS is on. The main regression
#    stats CSV keeps the GROUP_REGRESSION_BY_* grouping and is not
#    affected. No designated samples in the file means no second CSV.
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
#  - Plot files kept (SAVE_LOG_PLOTS / SAVE_GRID_IMAGE): two
#    independent toggles over FILES only — every variant is fitted and
#    contributes its rows to the regression stats CSV regardless.
#    SAVE_LOG_PLOTS writes a PNG for each log variant (the
#    BASE_REGRESSION_KIND plot is always written); SAVE_GRID_IMAGE
#    writes the combined 2x2 grid. A variant is RENDERED if its own
#    file is wanted or if the grid needs it: when the grid is on but
#    the log plots are off, the log variants are drawn into a
#    per-column temporary directory that is deleted once the grid has
#    been composited from them, so they never appear in OUTPUT_DIR.
#    When neither is on, nothing needs them drawn and no figure is
#    created for them at all.
#  - Point numbers: assigned once per INPUT ROW, before any
#    per-column filtering. With USE_POINT_NUMBER_COLUMN off they are
#    counted from POINT_LABEL_START over the file's data rows in file
#    order; with it on they are read from POINT_NUMBER_COLUMN instead,
#    falling back to the counted number (and logging it) for a row the
#    column cannot supply — the column absent from the file, a blank
#    cell, or a non-numeric value. A whole number read from the file is
#    captioned as an integer ("7", not "7.0"), and a column whose
#    values repeat is logged, since two points then carry the same
#    caption. POINT_NUMBER_COLUMN is never plotted as a y-column.
#    However the number was arrived at, a row carries the SAME
#    number in every plot it appears in, and a row dropped from one
#    column's plot (blank/non-numeric there, or non-positive on a
#    log-scaled axis) leaves its number simply absent from that plot
#    rather than renumbering anything. Numbers are drawn with
#    ax.annotate at POINT_LABEL_OFFSET from the marker; overlapping
#    numbers are not de-cluttered, so on a dense plot expect to zoom in
#    on the saved PNG (raise DPI if that is a regular need).
#  - Point key CSV (always written): one row per input data row —
#    Point Number, CSV Line (its line in the input file, header = 1),
#    the Sample ID, every LEGEND_COLUMNS value, the Designation, the
#    X_COLUMN value, and that row's value in every plotted y-column. A
#    blank y-column cell means that row was not plotted for that column.
#  - Residual CSV (EXPORT_RESIDUAL_OUTLIERS): one row per point per
#    fitted (column, variant, group), holding the point's X and Y as
#    plotted, the fitted value and residual in the space the regression
#    was fit in (natural log wherever that axis is log-scaled, so for
#    the linear variant these are the data's own units), the residual
#    divided by the group's residual standard deviation, and an Outlier
#    flag for |Std Residual| >= OUTLIER_SIGMA. Groups are the same ones
#    the main regression stats CSV uses. A group of fewer than 3 points
#    has no meaningful residual spread, so its standardized residuals
#    are written as blank (the raw residuals are still there).
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
#    quadrant rather than omitting the grid image; the grid figure is
#    sized from the pixel dimensions of those PNGs and each panel drawn
#    at aspect="equal", so panels are never stretched to fit), plus one
#    regression_stats_<timestamp>.csv (columns: Column, Regression
#    Type, Group, Slope, Intercept, R^2, N) covering every successfully
#    fitted (column, variant, group) triple. Figures are saved only —
#    no interactive plt.show() call. Never overwrites an existing file
#    of the same name.
#  - Circled points (HIGHLIGHT_CSV_PATH): an optional second CSV whose
#    HIGHLIGHT_MATCH_COLUMN values ("Sample ID" by default) name the
#    input rows to ring. It is meant to be a copy of the input file
#    with all but a handful of rows deleted, but any file carrying that
#    one column will do. Values are matched trimmed and
#    case-insensitively; a key matching no input row is logged and
#    otherwise ignored, and duplicate keys simply collapse. A matched
#    row is ringed in EVERY plot it appears in (all four scale
#    variants, and the grid image built from them) — a row filtered out
#    of one column's plot is not ringed there, since it isn't drawn
#    there at all. The ring is drawn with scatter at a fixed screen
#    size (HIGHLIGHT_CIRCLE_DIAMETER points across, s = d**2 in
#    points^2) so it is the same circle on a linear and a log axis, in
#    HIGHLIGHT_CIRCLE_COLOR at HIGHLIGHT_CIRCLE_LINEWIDTH, above the
#    markers and fit lines (zorder 3). It changes nothing that is
#    computed: the regressions, the stats CSVs and every other output
#    are identical whether or not a highlight file is given.
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
    """Assign each unique COLOR_COLUMN value a hex color.

    COLOR_MAP pins take precedence (exactly first, then
    case-insensitively); every remaining value takes the next color
    from COLOR_PALETTE that no pin already claimed, in sorted order.
    Pinning is what keeps Horz blue and Vert red regardless of which
    other categories exist in the file.
    """
    lowercase_pins = {key.strip().lower(): color for key, color in COLOR_MAP.items()}
    uniques = sorted(values.unique())

    resolved: dict[str, str] = {}
    unpinned: list[str] = []
    for value in uniques:
        if value in COLOR_MAP:
            resolved[value] = COLOR_MAP[value]
        elif value.strip().lower() in lowercase_pins:
            resolved[value] = lowercase_pins[value.strip().lower()]
        else:
            unpinned.append(value)

    available = [color for color in COLOR_PALETTE if color not in resolved.values()]
    if not available:  # every palette color is pinned — fall back to the full list
        available = list(COLOR_PALETTE)

    for i, value in enumerate(unpinned):
        resolved[value] = available[i % len(available)]

    if len(unpinned) > len(available):
        log.append(
            f"NOTE: {COLOR_COLUMN} has {len(unpinned)} unpinned value(s) but only "
            f"{len(available)} unused color(s) in COLOR_PALETTE — colors are reused (cycled). "
            f"Add hex codes to COLOR_PALETTE, or pin values in COLOR_MAP, to give every "
            f"value its own color."
        )
    return resolved


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


def derive_designation(sample_id: object) -> str:
    """Return a Sample ID's "1"/"2" designation, or "" when it has none.

    Only IDs with exactly DESIGNATION_UNDERSCORE_COUNT underscores carry
    one: "VT3_1_4" -> "1" (field DESIGNATION_FIELD_INDEX), while
    "VT3_4" and "FL15" -> "". The value is returned as text, so it can
    be matched against HOLLOW_DESIGNATIONS and used as a group name
    as-is.
    """
    if pd.isna(sample_id):
        return ""
    text = str(sample_id).strip()
    if text.count("_") != DESIGNATION_UNDERSCORE_COUNT:
        return ""
    fields = text.split("_")
    if DESIGNATION_FIELD_INDEX >= len(fields):
        return ""
    return fields[DESIGNATION_FIELD_INDEX].strip()


def resolve_point_numbers(df: pd.DataFrame, log: list[str]) -> pd.Series:
    """Return each input row's point number, read from the file or counted.

    With USE_POINT_NUMBER_COLUMN off (the default) the numbers are
    simply counted over the file's data rows from POINT_LABEL_START.
    With it on they are taken from POINT_NUMBER_COLUMN instead, so a
    point keeps the number it already had in an earlier run's point key
    even after the file has been re-sorted or cut down. Anything that
    column cannot supply — the column missing altogether, a blank cell,
    a value that is not a number — falls back to that row's counted
    number and is reported in the run's log.
    """
    counted = pd.Series(range(POINT_LABEL_START, POINT_LABEL_START + len(df)), index=df.index)
    if not USE_POINT_NUMBER_COLUMN:
        return counted

    if POINT_NUMBER_COLUMN not in df.columns:
        log.append(
            f"NOTE: USE_POINT_NUMBER_COLUMN is on, but '{POINT_NUMBER_COLUMN}' is not a column "
            f"of '{INPUT_PATH}' — every point is numbered by its row in the file instead, "
            f"counting from {POINT_LABEL_START}."
        )
        return counted

    numbers = pd.to_numeric(df[POINT_NUMBER_COLUMN], errors="coerce")
    # A whole number is kept as an int so a 7 read back out of the file
    # is captioned "7" rather than "7.0".
    resolved = pd.Series(
        [
            counted[index]
            if pd.isna(value)
            else (int(value) if float(value).is_integer() else value)
            for index, value in numbers.items()
        ],
        index=df.index,
        dtype=object,
    )

    fallbacks = int(numbers.isna().sum())
    if fallbacks:
        log.append(
            f"NOTE: {fallbacks} row(s) have a blank or non-numeric '{POINT_NUMBER_COLUMN}' — "
            f"those points are numbered by their row in the file instead."
        )

    repeated = sorted({str(value) for value in resolved[resolved.duplicated(keep=False)]})
    if repeated:
        log.append(
            f"NOTE: '{POINT_NUMBER_COLUMN}' is not unique — number(s) {', '.join(repeated)} are "
            f"shared by more than one row, so more than one point on each plot is captioned "
            f"with them."
        )
    return resolved


def build_designations(df: pd.DataFrame, log: list[str]) -> pd.Series:
    """Return one designation string per row ("" where the ID has none)."""
    if SAMPLE_ID_COLUMN not in df.columns:
        log.append(
            f"NOTE: no '{SAMPLE_ID_COLUMN}' column in the input file — no sample is "
            f"designated, every point is drawn solid and no per-designation stats CSV "
            f"is written"
        )
        return pd.Series([""] * len(df), index=df.index, dtype=object)

    designations = df[SAMPLE_ID_COLUMN].apply(derive_designation)
    present = sorted({value for value in designations if value})
    undesignated = int((designations == "").sum())
    if present:
        log.append(
            f"NOTE: designation(s) found in '{SAMPLE_ID_COLUMN}': "
            f"{', '.join(repr(value) for value in present)} "
            f"(hollow: {', '.join(repr(v) for v in HOLLOW_DESIGNATIONS)}); "
            f"{undesignated} row(s) with no designation are drawn solid"
        )
    else:
        log.append(
            f"NOTE: no '{SAMPLE_ID_COLUMN}' value has exactly "
            f"{DESIGNATION_UNDERSCORE_COUNT} underscore(s) — no sample is designated"
        )
    return designations


def load_highlight_keys(log: list[str]) -> set[str]:
    """Read HIGHLIGHT_CSV_PATH and return its match values, normalized.

    Keys come back trimmed and lowercased, so matching them against the
    input file's own column ignores case and stray whitespace. Returns
    an empty set when HIGHLIGHT_CSV_PATH is None (feature off).
    """
    if HIGHLIGHT_CSV_PATH is None:
        return set()

    highlight_df = pd.read_csv(HIGHLIGHT_CSV_PATH)
    if HIGHLIGHT_MATCH_COLUMN not in highlight_df.columns:
        raise ValueError(
            f"HIGHLIGHT_MATCH_COLUMN '{HIGHLIGHT_MATCH_COLUMN}' not found in the highlight "
            f"file '{HIGHLIGHT_CSV_PATH}' (its columns are {list(highlight_df.columns)})"
        )

    keys = {
        str(value).strip().lower()
        for value in highlight_df[HIGHLIGHT_MATCH_COLUMN]
        if not pd.isna(value) and str(value).strip()
    }
    if not keys:
        log.append(
            f"NOTE: the highlight file '{HIGHLIGHT_CSV_PATH}' has no usable "
            f"'{HIGHLIGHT_MATCH_COLUMN}' value — no points are circled."
        )
    return keys


def build_highlight_mask(df: pd.DataFrame, keys: set[str], log: list[str]) -> pd.Series:
    """Flag every input row whose match value appears in the highlight file.

    Rows flagged here are ringed in every plot they appear in. A key in
    the highlight file that matches no input row is logged by name —
    that is usually a typo or a Sample ID that the input file does not
    (or no longer) carry.
    """
    if not keys:
        return pd.Series(False, index=df.index)

    normalized = df[HIGHLIGHT_MATCH_COLUMN].apply(
        lambda value: "" if pd.isna(value) else str(value).strip().lower()
    )
    mask = normalized.isin(keys)

    unmatched = sorted(keys - set(normalized))
    if unmatched:
        log.append(
            f"NOTE: {len(unmatched)} '{HIGHLIGHT_MATCH_COLUMN}' value(s) in the highlight file "
            f"match no row of the input file and are not circled: {', '.join(unmatched)}"
        )
    return mask


def fit_line(
    xs: np.ndarray, ys: np.ndarray, log_x: bool, log_y: bool
) -> tuple[float, float, float]:
    """OLS straight-line fit in the (possibly log-transformed) plotted space.

    Returns (slope, intercept, R^2), with R^2 computed in that same
    space so it describes the straight line actually drawn.
    """
    x_fit = np.log(xs) if log_x else xs
    y_fit = np.log(ys) if log_y else ys

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    predicted = slope * x_fit + intercept
    residual_ss = float(np.sum((y_fit - predicted) ** 2))
    total_ss = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss != 0 else float("nan")
    return float(slope), float(intercept), r_squared


def build_residual_rows(
    *,
    column: str,
    kind: str,
    group_desc: str,
    xs: np.ndarray,
    ys: np.ndarray,
    log_x: bool,
    log_y: bool,
    slope: float,
    intercept: float,
    point_numbers: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    designations: np.ndarray,
) -> list[dict]:
    """One row per point: how far it sits from its group's fit line.

    Residuals are taken in the space the regression was fit in (natural
    log wherever that axis is log-scaled), so they are directly
    comparable to that variant's slope/intercept; for the linear
    variant they are in the data's own units. "Std Residual" divides by
    the group's residual standard deviation, which is what makes points
    comparable ACROSS columns and variants — a group of fewer than 3
    points, or one whose residuals are all identical, has no usable
    spread, so its standardized values (and the Outlier flag) are left
    blank rather than invented.
    """
    x_fit = np.log(xs) if log_x else xs
    y_fit = np.log(ys) if log_y else ys
    predicted = slope * x_fit + intercept
    residuals = y_fit - predicted

    # ddof=2 — a straight line through n points spends two degrees of
    # freedom, so this is the usual regression residual standard error.
    spread = float(np.std(residuals, ddof=2)) if len(residuals) > 2 else 0.0
    usable_spread = spread > 0.0

    rows: list[dict] = []
    for i in range(len(residuals)):
        residual = float(residuals[i])
        standardized = residual / spread if usable_spread else None
        rows.append(
            {
                "Column": column,
                "Regression Type": kind,
                "Group": group_desc,
                "Point Number": point_numbers[i],
                "Sample ID": sample_ids[i],
                "Label": labels[i],
                "Designation": designations[i],
                X_COLUMN: float(xs[i]),
                "Y Value": float(ys[i]),
                "Fitted (fit space)": float(predicted[i]),
                "Residual (fit space)": residual,
                "Std Residual": standardized,
                "Abs Std Residual": abs(standardized) if standardized is not None else None,
                "Outlier": (
                    abs(standardized) >= OUTLIER_SIGMA if standardized is not None else None
                ),
            }
        )
    return rows


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

    # Size the grid from the source PNGs themselves rather than a guessed
    # figure size: each cell is made as large as the widest/tallest panel
    # (they differ slightly, since bbox_inches="tight" trims each one to
    # its own legend width), and every image is drawn at aspect="equal".
    # A cell that is roomier than its image gets a little blank margin;
    # nothing is stretched or skewed.
    images = {kind: plt.imread(path) for kind, path in kind_paths.items()}
    cell_height_px = max(image.shape[0] for image in images.values())
    cell_width_px = max(image.shape[1] for image in images.values())
    figure_width_in = 2 * cell_width_px / DPI
    figure_height_in = 2 * cell_height_px / DPI + GRID_TITLE_IN

    fig, axes = plt.subplots(2, 2, figsize=(figure_width_in, figure_height_in), dpi=DPI)
    missing = 0
    for kind, row, col in grid_layout:
        ax = axes[row, col]
        image = images.get(kind)
        if image is not None:
            ax.imshow(image, aspect="equal")
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
    fig.subplots_adjust(
        left=0.0,
        right=1.0,
        top=1.0 - GRID_TITLE_IN / figure_height_in,
        bottom=0.0,
        wspace=0.0,
        hspace=0.0,
    )
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
    designations_all: pd.Series,
    highlights_all: pd.Series,
    point_numbers_all: pd.Series,
    sample_ids_all: pd.Series,
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
    destination_dir: Path,
    render: bool,
    log: list[str],
) -> tuple[list[dict], Path | None]:
    """Fit one (column, scale-variant) pair, and draw it when asked.

    render=False fits the regression and returns its stats rows without
    building a figure at all — used when neither SAVE_GRID_IMAGE nor
    SAVE_LOG_PLOTS needs this variant drawn. It returns None as the path.

    destination_dir is where this variant's PNG is written when it IS
    rendered — OUTPUT_DIR for a variant being kept, or a scratch
    directory for one that is only needed long enough to be composited
    into the grid image (see SAVE_GRID_IMAGE / SAVE_LOG_PLOTS).

    categories_all holds one cleaned column per LEGEND_COLUMNS entry
    plus the joined "Label" column. designations_all holds each row's
    "1"/"2" designation ("" where its Sample ID has none): it decides
    solid vs hollow fill and drives the second, designation-only set of
    regression rows. grouping_mode is one of "all", "color", "marker",
    "both" (see GROUP_REGRESSION_BY_COLOR / GROUP_REGRESSION_BY_MARKER).
    point_numbers_all is each input row's point number (its row number
    in the input CSV) and sample_ids_all its Sample ID; they caption the
    plotted points and identify them in the residual rows.

    Returns (regression_stats rows, designation_stats rows, residual
    rows, output PNG path or None if nothing was plotted for lack of
    valid points).
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
        return [], [], [], None

    x = x_all[valid]
    y = y_all[valid]
    categories = categories_all[valid]
    designation_np = designations_all[valid].to_numpy()
    highlight_np = highlights_all[valid].to_numpy()
    point_np = point_numbers_all[valid].to_numpy()
    sample_id_np = sample_ids_all[valid].to_numpy()

    fig, ax = (plt.subplots(figsize=FIGSIZE, dpi=DPI) if render else (None, None))

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

    if render:
        # One scatter series per (legend label, designation) present, so
        # the "1" half of a category can be drawn solid and the "2" half
        # hollow while both keep that category's color and marker.
        series_keys = sorted({(label, des) for label, des in zip(label_np, designation_np)})
        for label, designation in series_keys:
            color_key, marker_key = label_keys[label]
            color = color_map[color_key]
            marker = marker_map[marker_key] if MARKER_COLUMN is not None else DEFAULT_MARKER
            point_mask = (label_np == label) & (designation_np == designation)
            hollow = designation in HOLLOW_DESIGNATIONS
            if hollow:
                ax.scatter(
                    x_np[point_mask],
                    y_np[point_mask],
                    s=POINT_SIZE,
                    marker=marker,
                    facecolors="none",
                    edgecolors=color,
                    linewidths=HOLLOW_MARKER_EDGE_WIDTH,
                )
            else:
                ax.scatter(
                    x_np[point_mask], y_np[point_mask], s=POINT_SIZE, color=color, marker=marker
                )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle="none",
                    color=color,
                    markerfacecolor="none" if hollow else color,
                    markeredgecolor=color,
                )
            )
            legend_labels.append(
                f"{label} ({designation})" if designation and SHOW_DESIGNATION_IN_LEGEND else label
            )

        if highlight_np.any():
            # One ring per highlighted point, drawn over the markers.
            # scatter's s is an AREA in points^2, so a ring
            # HIGHLIGHT_CIRCLE_DIAMETER points across is s = d**2 —
            # a screen size, identical on every scale variant and
            # independent of the axes' units.
            ax.scatter(
                x_np[highlight_np],
                y_np[highlight_np],
                s=HIGHLIGHT_CIRCLE_DIAMETER ** 2,
                marker="o",
                facecolors="none",
                edgecolors=HIGHLIGHT_CIRCLE_COLOR,
                linewidths=HIGHLIGHT_CIRCLE_LINEWIDTH,
                zorder=3,
            )
            if SHOW_HIGHLIGHT_IN_LEGEND:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="none",
                        color=HIGHLIGHT_CIRCLE_COLOR,
                        markerfacecolor="none",
                        markeredgecolor=HIGHLIGHT_CIRCLE_COLOR,
                        markeredgewidth=HIGHLIGHT_CIRCLE_LINEWIDTH,
                        markersize=HIGHLIGHT_CIRCLE_DIAMETER,
                    )
                )
                legend_labels.append(HIGHLIGHT_LEGEND_LABEL)

        if SHOW_POINT_LABELS:
            # Drawn in one pass over every point, after the series, so a
            # number is never hidden under a marker drawn later.
            for x_point, y_point, number, color_key in zip(x_np, y_np, point_np, color_np):
                ax.annotate(
                    str(number),
                    (x_point, y_point),
                    textcoords="offset points",
                    xytext=POINT_LABEL_OFFSET,
                    fontsize=POINT_LABEL_FONTSIZE,
                    color=color_map[color_key] if POINT_LABEL_COLOR_FROM_POINT else POINT_LABEL_COLOR,
                )

    if grouping_mode == "both":
        group_keys = sorted(label_keys)
    elif grouping_mode == "color":
        group_keys = sorted(set(color_np))
    elif grouping_mode == "marker":
        group_keys = sorted(set(marker_np))
    else:
        group_keys = ["All"]

    regression_rows: list[dict] = []
    residual_rows: list[dict] = []
    collect_residuals = EXPORT_RESIDUAL_OUTLIERS and (
        RESIDUALS_FOR_ALL_VARIANTS or kind == BASE_REGRESSION_KIND
    )
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

        slope, intercept, r_squared = fit_line(xs, ys, log_x, log_y)

        if collect_residuals:
            residual_rows.extend(
                build_residual_rows(
                    column=column,
                    kind=kind,
                    group_desc=group_desc,
                    xs=xs,
                    ys=ys,
                    log_x=log_x,
                    log_y=log_y,
                    slope=slope,
                    intercept=intercept,
                    point_numbers=point_np[group_mask],
                    sample_ids=sample_id_np[group_mask],
                    labels=label_np[group_mask],
                    designations=designation_np[group_mask],
                )
            )

        # The fit is still computed (and recorded below) when
        # SHOW_FIT_LINES is off — only the drawing is skipped.
        if render and SHOW_FIT_LINES:
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

    # ----- second, independent grouping: designation alone ("1" vs "2")
    designation_rows: list[dict] = []
    if WRITE_DESIGNATION_STATS_CSV:
        designation_groups = sorted({value for value in designation_np if value})
        if designation_groups and INCLUDE_ALL_ROW_IN_DESIGNATION_STATS:
            designation_groups = ["All"] + designation_groups
        for designation in designation_groups:
            if designation == "All":
                group_mask = np.ones(len(x_np), dtype=bool)
            else:
                group_mask = designation_np == designation
            xs = x_np[group_mask]
            ys = y_np[group_mask]
            if len(xs) < 2:
                log.append(
                    f"SKIP '{column}'{title_suffix} designation [{designation}]: fewer "
                    f"than 2 valid points ({len(xs)}) — cannot fit a regression"
                )
                continue
            slope, intercept, r_squared = fit_line(xs, ys, log_x, log_y)
            designation_rows.append(
                {
                    "Column": column,
                    "Regression Type": kind,
                    "Designation": designation,
                    "Slope": slope,
                    "Intercept": intercept,
                    "R^2": r_squared,
                    "N": len(xs),
                }
            )

    if not render:
        return regression_rows, designation_rows, residual_rows, None

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
    output_path = destination_dir / f"{safe_column}_vs_{X_COLUMN}{filename_suffix}_plot_{timestamp}.png"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return regression_rows, designation_rows, residual_rows, output_path


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
    for value, color in COLOR_MAP.items():
        if not mcolors.is_color_like(color):
            raise ValueError(f"COLOR_MAP['{value}'] is not a valid color: {color!r}")
    for color in COLOR_PALETTE:
        if not mcolors.is_color_like(color):
            raise ValueError(f"COLOR_PALETTE entry is not a valid color: {color!r}")

    if "SAVE_ALL_PLOT_VARIANTS" in globals():
        raise ValueError(
            "SAVE_ALL_PLOT_VARIANTS has been replaced by two separate toggles: "
            "SAVE_LOG_PLOTS (save log-x/log-y/log-log as individual PNGs) and "
            "SAVE_GRID_IMAGE (save the combined 2x2 grid). Delete the old "
            "SAVE_ALL_PLOT_VARIANTS line and set those two instead — "
            "SAVE_ALL_PLOT_VARIANTS = True is now SAVE_LOG_PLOTS = True."
        )

    if HIGHLIGHT_CSV_PATH is not None:
        if not Path(HIGHLIGHT_CSV_PATH).is_file():
            raise ValueError(f"HIGHLIGHT_CSV_PATH is not an existing file: {HIGHLIGHT_CSV_PATH}")
        if HIGHLIGHT_MATCH_COLUMN not in df.columns:
            raise ValueError(
                f"HIGHLIGHT_MATCH_COLUMN '{HIGHLIGHT_MATCH_COLUMN}' not found in the input file "
                f"'{INPUT_PATH}' — it has to exist in both the input CSV and the highlight CSV"
            )
        if not mcolors.is_color_like(HIGHLIGHT_CIRCLE_COLOR):
            raise ValueError(
                f"HIGHLIGHT_CIRCLE_COLOR is not a valid color: {HIGHLIGHT_CIRCLE_COLOR!r}"
            )
        if HIGHLIGHT_CIRCLE_DIAMETER <= 0:
            raise ValueError(
                f"HIGHLIGHT_CIRCLE_DIAMETER must be positive, got {HIGHLIGHT_CIRCLE_DIAMETER}"
            )
        if HIGHLIGHT_CIRCLE_LINEWIDTH <= 0:
            raise ValueError(
                f"HIGHLIGHT_CIRCLE_LINEWIDTH must be positive, got {HIGHLIGHT_CIRCLE_LINEWIDTH}"
            )

    if OUTLIER_SIGMA <= 0:
        raise ValueError(f"OUTLIER_SIGMA must be positive, got {OUTLIER_SIGMA}")

    if DESIGNATION_FIELD_INDEX < 0 or DESIGNATION_FIELD_INDEX > DESIGNATION_UNDERSCORE_COUNT:
        raise ValueError(
            f"DESIGNATION_FIELD_INDEX {DESIGNATION_FIELD_INDEX} is out of range for a Sample ID "
            f"with {DESIGNATION_UNDERSCORE_COUNT} underscore(s) — it must be between 0 and "
            f"{DESIGNATION_UNDERSCORE_COUNT}"
        )

    known_kinds = [kind for kind, *_ in REGRESSION_KINDS]
    if BASE_REGRESSION_KIND not in known_kinds:
        raise ValueError(
            f"BASE_REGRESSION_KIND '{BASE_REGRESSION_KIND}' is not one of the "
            f"REGRESSION_KINDS variants {known_kinds}"
        )


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
    designations_all = build_designations(df, log)
    highlights_all = build_highlight_mask(df, load_highlight_keys(log), log)

    # Point numbers are resolved once, over the file's data rows and
    # before any per-column filtering — so a row keeps the same number
    # in every plot it appears in. They are counted from
    # POINT_LABEL_START, or read from POINT_NUMBER_COLUMN when
    # USE_POINT_NUMBER_COLUMN is on. "CSV Line" is where a spreadsheet
    # shows that row, with the header occupying line 1.
    point_numbers_all = resolve_point_numbers(df, log)
    csv_lines_all = pd.Series(range(2, 2 + len(df)), index=df.index)
    if SAMPLE_ID_COLUMN in df.columns:
        sample_ids_all = df[SAMPLE_ID_COLUMN].fillna("").astype(str)
    else:
        sample_ids_all = pd.Series([""] * len(df), index=df.index, dtype=object)
    color_map = build_color_map(categories_all[COLOR_COLUMN], log)
    if MARKER_COLUMN is not None:
        marker_map = build_marker_map(categories_all[MARKER_COLUMN], log)
        line_style_map = build_line_style_map(categories_all[MARKER_COLUMN])
    else:
        marker_map = {}
        line_style_map = {}

    grouping_mode = resolve_grouping_mode()

    excluded_columns = IGNORE_COLUMNS | set(LEGEND_COLUMNS)
    if POINT_NUMBER_COLUMN in df.columns:
        # An identifier, not a property — plotting it against the gray
        # value would just draw the row order.
        excluded_columns = excluded_columns | {POINT_NUMBER_COLUMN}
        log.append(
            f"NOTE: '{POINT_NUMBER_COLUMN}' is a point identifier, not a measured property — "
            f"it is never plotted as a y-column."
        )
    candidate_columns = [c for c in df.columns if c != X_COLUMN and c not in excluded_columns]

    regression_rows: list[dict] = []
    designation_rows: list[dict] = []
    residual_rows: list[dict] = []
    plotted_y_columns: dict[str, pd.Series] = {}
    grid_images_written = 0
    plots_written = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for column in candidate_columns:
        y_all = pd.to_numeric(df[column], errors="coerce")
        if y_all.notna().sum() == 0:
            continue  # not a numeric column (e.g. Sample ID, Notes) — skip silently

        plotted_y_columns[column] = y_all

        # A variant is KEPT if its own PNG is wanted, and RENDERED if it
        # is kept or if the grid image needs it composited in. One that
        # is rendered but not kept goes to a scratch directory that is
        # deleted when this column is done; one that is neither is only
        # fitted, never drawn.
        kind_paths: dict[str, Path] = {}
        with tempfile.TemporaryDirectory(prefix="c6_variants_") as scratch:
            scratch_dir = Path(scratch)
            for kind, log_x, log_y, filename_suffix, title_suffix in REGRESSION_KINDS:
                keep = SAVE_LOG_PLOTS or kind == BASE_REGRESSION_KIND
                render = keep or SAVE_GRID_IMAGE
                (
                    rows,
                    designation_group_rows,
                    variant_residual_rows,
                    output_path,
                ) = fit_and_plot_variant(
                    column,
                    x_all,
                    y_all,
                    categories_all,
                    designations_all,
                    highlights_all,
                    point_numbers_all,
                    sample_ids_all,
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
                    OUTPUT_DIR if keep else scratch_dir,
                    render,
                    log,
                )
                regression_rows.extend(rows)
                designation_rows.extend(designation_group_rows)
                residual_rows.extend(variant_residual_rows)
                if output_path is not None:
                    kind_paths[kind] = output_path
                    if keep:
                        plots_written += 1

            # Built inside the scratch context, while the discarded
            # variants' PNGs still exist to be read back.
            if SAVE_GRID_IMAGE and build_grid_image(column, kind_paths, timestamp, log) is not None:
                grid_images_written += 1

    print("\n--- Legend categories ---")
    print(f"  Legend label = {', '.join(LEGEND_COLUMNS)} (joined with ', ')")
    for value, color in sorted(color_map.items()):
        print(f"  {COLOR_COLUMN} '{value}' -> color {color}")
    for value, marker in sorted(marker_map.items()):
        print(f"  {MARKER_COLUMN} '{value}' -> marker '{marker}'")
    print(f"  Fit lines drawn: {SHOW_FIT_LINES}; fit equations in legend: {SHOW_FIT_EQUATIONS_IN_LEGEND}")
    if SAVE_LOG_PLOTS:
        print("  Saving an individual PNG for the base plot and each log variant (SAVE_LOG_PLOTS = True)")
    else:
        print(f"  Saving only the '{BASE_REGRESSION_KIND}' plot per column (SAVE_LOG_PLOTS = False)")
    print(f"  Saving the combined 2x2 grid image: {SAVE_GRID_IMAGE} (SAVE_GRID_IMAGE)")
    if not SAVE_LOG_PLOTS and not SAVE_GRID_IMAGE:
        print("  Log variants are fitted for the stats CSV only — no figure is drawn for them")
    print("  All four variants are fitted and appear in the regression stats CSV either way")

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    if not regression_rows:
        print("\nNo columns had enough numeric data to plot/regress — no output written.")
        return

    # ----- the point key: what is point 17?
    point_key = pd.DataFrame(
        {
            "Point Number": point_numbers_all,
            "CSV Line": csv_lines_all,
            "Sample ID": sample_ids_all,
        }
    )
    for legend_column in LEGEND_COLUMNS:
        point_key[legend_column] = categories_all[legend_column]
    point_key["Designation"] = designations_all
    if HIGHLIGHT_CSV_PATH is not None:
        point_key["Circled"] = highlights_all
    point_key[X_COLUMN] = x_all
    for column, values in plotted_y_columns.items():
        point_key[column] = values

    point_key_path = OUTPUT_DIR / f"point_key_{timestamp}.csv"
    if point_key_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {point_key_path}")
    point_key.to_csv(point_key_path, index=False)

    regression_df = pd.DataFrame(
        regression_rows, columns=["Column", "Regression Type", "Group", "Slope", "Intercept", "R^2", "N"]
    )
    stats_path = OUTPUT_DIR / f"regression_stats_{timestamp}.csv"
    if stats_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {stats_path}")
    regression_df.to_csv(stats_path, index=False)

    designation_stats_path: Path | None = None
    if WRITE_DESIGNATION_STATS_CSV and designation_rows:
        designation_df = pd.DataFrame(
            designation_rows,
            columns=["Column", "Regression Type", "Designation", "Slope", "Intercept", "R^2", "N"],
        )
        designation_stats_path = OUTPUT_DIR / f"regression_stats_by_designation_{timestamp}.csv"
        if designation_stats_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output file: {designation_stats_path}"
            )
        designation_df.to_csv(designation_stats_path, index=False)

    residual_path: Path | None = None
    flagged_outliers = 0
    if EXPORT_RESIDUAL_OUTLIERS and residual_rows:
        residual_df = pd.DataFrame(
            residual_rows,
            columns=[
                "Column",
                "Regression Type",
                "Group",
                "Point Number",
                "Sample ID",
                "Label",
                "Designation",
                X_COLUMN,
                "Y Value",
                "Fitted (fit space)",
                "Residual (fit space)",
                "Std Residual",
                "Abs Std Residual",
                "Outlier",
            ],
        )
        # Worst offenders first: the top of the file is the list of
        # points to go and look at.
        residual_df = residual_df.sort_values(
            "Abs Std Residual", ascending=False, na_position="last", kind="stable"
        )
        flagged_outliers = int(residual_df["Outlier"].fillna(False).sum())
        residual_path = OUTPUT_DIR / f"point_residuals_{timestamp}.csv"
        if residual_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output file: {residual_path}")
        residual_df.to_csv(residual_path, index=False)

    print("\n--- Summary ---")
    print(f"  Regression fits computed: {len(regression_rows)}")
    print(f"  Standalone plot PNGs saved: {plots_written}")
    print(f"  Grid images generated: {grid_images_written}")
    print(f"  Columns covered: {sorted({r['Column'] for r in regression_rows})}")
    if HIGHLIGHT_CSV_PATH is not None:
        print(
            f"  Points circled: {int(highlights_all.sum())} row(s) matched by "
            f"'{HIGHLIGHT_MATCH_COLUMN}' from '{HIGHLIGHT_CSV_PATH}'"
        )
    if USE_POINT_NUMBER_COLUMN and POINT_NUMBER_COLUMN in df.columns:
        print(f"  Points numbered: {len(point_key)} (from the '{POINT_NUMBER_COLUMN}' column)")
    else:
        print(f"  Points numbered: {len(point_key)} (point {POINT_LABEL_START} = first data row)")
    print(f"  Point key file: {point_key_path}")
    if residual_path is not None:
        print(
            f"  Point residual file: {residual_path} "
            f"({flagged_outliers} row(s) flagged at |Std Residual| >= {OUTLIER_SIGMA})"
        )
    elif EXPORT_RESIDUAL_OUTLIERS:
        print("  Point residual file: not written (no regression had residuals to report)")
    print(f"  Regression stats file: {stats_path}")
    if designation_stats_path is not None:
        print(f"  Per-designation stats file: {designation_stats_path}")
    elif WRITE_DESIGNATION_STATS_CSV:
        print("  Per-designation stats file: not written (no designated samples found)")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
