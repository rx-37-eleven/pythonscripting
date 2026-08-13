"""
Per-sample stress-strain plotting script.

Reads a zeroed stress-strain dataset (the output of zero_stress_strain.py:
wide, alternating (strain, stress) column pairs per sample, Sample ID in
row 1 of each stress column, header/units row in row 2, data from row 3,
each sample's data starting at (0, 0)) plus a combined results file (the
output of combine_csv_folder.py: one row per Sample ID, columns merged
in from the earlier scripts' summary outputs — Modulus (Slope), Yield
Strain, Yield Stress, Strain at Max Stress, Max Stress, etc.).

For each sample present in both files:
  - Plots the zeroed stress-strain curve.
  - Overlays a straight modulus line through the origin with that
    sample's Modulus (Slope), clipped to wherever it exits the
    configured y-axis limits (or the x-axis limits, whichever comes
    first).
  - Marks the yield point (Yield Strain, Yield Stress) and the UTS
    point (Strain at Max Stress, Max Stress) from the combined file.
  - Saves the figure as <sampleID>_plot_<timestamp>.png, with every
    plot in the run sharing identical axis limits (from CONFIG) and one
    shared run timestamp.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependencies: pandas, matplotlib.

This script is standalone — it does not import or depend on any other
script in this repository, including zero_stress_strain.py or
combine_csv_folder.py. It merely consumes their output files as input.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Path to the zeroed dataset CSV (output of zero_stress_strain.py's
# zeroed_outputs_<timestamp>.csv).
ZEROED_INPUT_PATH = Path("zeroed_outputs.csv")

# Path to the combined results CSV (output of combine_csv_folder.py's
# DataCombined_<timestamp>.csv) — supplies Modulus (Slope), Yield
# Strain, Yield Stress, Strain at Max Stress, and Max Stress per sample.
COMBINED_INPUT_PATH = Path("DataCombined.csv")

# Directory the PNG files are written into.
OUTPUT_DIR = Path("outputs")

# Axis limits applied identically to every plot, so all samples can be
# compared side by side on the same scale.
X_MIN = 0.0
X_MAX = 0.03
Y_MIN = 0.0
Y_MAX = 2000.0

# Whether to annotate the yield and UTS points with their numeric
# (strain, stress) values, or just mark them.
SHOW_POINT_LABELS = False

# Figure resolution and size (matplotlib default figsize if None).
DPI = 150
FIGSIZE = None

# Line width (points) for the modulus line.
MODULUS_LINE_WIDTH = 1.5

# ---------------------------------------------------------------------
# Resolved answers to the brief's open questions (captured here per the
# brief's "definition of done"):
#
#  - Input structures: confirmed directly (this script's author also
#    wrote zero_stress_strain.py and combine_csv_folder.py) — zeroed
#    file is wide/alternating (strain, stress) pairs with Sample ID in
#    row 1 of the stress column, units row in row 2, data from row 3.
#    Combined file is long/tidy, one row per Sample ID, with columns
#    "Modulus (Slope)", "Yield Strain", "Yield Stress",
#    "Strain at Max Stress", "Max Stress" used here as the overlay
#    source (read directly, not re-derived from the curve).
#  - Input location: CONFIG block (explicit paths), not CLI args or
#    auto-discovery — matches this repo's existing script convention.
#  - Sample ID matching: exact string match, no trimming/normalization
#    (consistent with the rest of this script series). A sample present
#    in only one file is skipped with a warning, not plotted, not an
#    error.
#  - UTS point: read directly from the combined file's "Strain at Max
#    Stress" / "Max Stress" columns, not re-derived from the curve.
#  - Modulus line: drawn through the origin with slope = Modulus
#    (Slope), over the strain range from 0 up to wherever slope*x exits
#    the configured Y_MIN/Y_MAX (or X_MAX, whichever comes first) — not
#    drawn across the full x-axis width regardless of where it would
#    land outside the y-range.
#  - No 0.2% offset line plotted — modulus line only, per the brief's
#    literal wording.
#  - Curve style: line only, no per-point markers. A legend is included
#    identifying the curve, modulus line, yield point, and UTS point.
#  - Point markers: yield point is a green triangle ("^"), UTS point is
#    a green square ("s").
#  - Point annotation: controlled by SHOW_POINT_LABELS in CONFIG —
#    off by default (marked only), can be turned on to print numeric
#    (strain, stress) values next to the yield/UTS markers.
#  - Figure settings: gridlines on, default matplotlib figsize (None ->
#    matplotlib's own default), 150 DPI, modulus line width configurable
#    via MODULUS_LINE_WIDTH — all editable via CONFIG.
#  - Config scope: axis limits, input/output paths, DPI, figsize,
#    MODULUS_LINE_WIDTH, and SHOW_POINT_LABELS all live in the CONFIG
#    block as named constants.
#  - Output: one shared timestamp (YYYYMMDD_HHMMSS) taken once at the
#    start of the run, used in every plot's filename:
#    <sampleID>_plot_<timestamp>.png. Sample IDs are used as-is in
#    filenames, not sanitized.
#  - Figures are saved only — no interactive plt.show() call.
# =====================================================================

REQUIRED_COMBINED_COLUMNS = [
    "Sample ID",
    "Modulus (Slope)",
    "Yield Strain",
    "Yield Stress",
    "Strain at Max Stress",
    "Max Stress",
]


def discover_samples(header_row: pd.Series) -> list[tuple[str, int, int]]:
    """Scan row 1 (0-indexed header_row) for sample pairs.

    Each pair occupies two adjacent columns (strain, stress). A pair is
    recognized when its stress (second) column has a non-blank cell in
    header_row — that cell is the Sample ID. Scanning stops at the
    first column pair whose stress cell is blank, or when columns run
    out. Returns a list of (sample_id, strain_col_idx, stress_col_idx).
    """
    n_cols = len(header_row)
    samples: list[tuple[str, int, int]] = []
    col = 0
    while col + 1 < n_cols:
        strain_idx, stress_idx = col, col + 1
        sample_id = header_row.iloc[stress_idx]
        if pd.isna(sample_id) or str(sample_id).strip() == "":
            break
        samples.append((str(sample_id).strip(), strain_idx, stress_idx))
        col += 2
    return samples


def load_sample(
    df_raw: pd.DataFrame, strain_idx: int, stress_idx: int
) -> tuple[pd.Series, pd.Series]:
    """Read one sample's (strain, stress) data starting at row 3 (0-indexed row 2).

    Trims trailing rows where both cells are blank (handles ragged/
    blank-padded column lengths), then coerces the remaining cells to
    numeric.
    """
    strain_col = df_raw.iloc[2:, strain_idx]
    stress_col = df_raw.iloc[2:, stress_idx]

    both_blank = strain_col.isna() & stress_col.isna()
    if both_blank.all():
        return pd.Series(dtype=float), pd.Series(dtype=float)
    last_valid_pos = both_blank[~both_blank.to_numpy()].index[-1]
    strain_col = strain_col.loc[:last_valid_pos]
    stress_col = stress_col.loc[:last_valid_pos]

    strain = pd.to_numeric(strain_col, errors="coerce").reset_index(drop=True)
    stress = pd.to_numeric(stress_col, errors="coerce").reset_index(drop=True)
    return strain, stress


def load_combined(combined_path: Path) -> pd.DataFrame:
    """Read the combined results CSV and validate required columns are present."""
    df = pd.read_csv(combined_path, dtype=str)
    missing = [c for c in REQUIRED_COMBINED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Combined results file '{combined_path}' is missing required "
            f"column(s): {missing}"
        )
    df["Sample ID"] = df["Sample ID"].str.strip()
    return df


def modulus_line_x_range(
    slope: float, x_min: float, x_max: float, y_min: float, y_max: float
) -> tuple[float, float]:
    """Return the (x_start, x_end) span for the modulus line, from x=0.

    Drawn from strain=0 up to wherever slope*x first exits [y_min,
    y_max], clipped to [x_min, x_max] (whichever bound is reached
    first).
    """
    x_start = max(0.0, x_min)

    if slope == 0:
        return x_start, x_max

    x_at_y_min = y_min / slope
    x_at_y_max = y_max / slope
    y_bound_candidates = [x for x in (x_at_y_min, x_at_y_max) if x > x_start]
    x_end = min(y_bound_candidates) if y_bound_candidates else x_start
    x_end = min(x_end, x_max)
    x_end = max(x_end, x_start)
    return x_start, x_end


def plot_sample(
    sample_id: str,
    strain: pd.Series,
    stress: pd.Series,
    slope: float,
    yield_strain: float,
    yield_stress: float,
    uts_strain: float,
    uts_stress: float,
) -> plt.Figure:
    """Build one sample's stress-strain figure with modulus line, yield point, UTS point."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(strain, stress, linestyle="-", label="Stress-strain curve")

    x0, x1 = modulus_line_x_range(slope, X_MIN, X_MAX, Y_MIN, Y_MAX)
    ax.plot(
        [x0, x1],
        [slope * x0, slope * x1],
        linestyle="--",
        linewidth=MODULUS_LINE_WIDTH,
        label="Modulus line",
    )

    ax.plot(
        yield_strain,
        yield_stress,
        marker="^",
        color="green",
        linestyle="none",
        label="Yield point",
    )
    ax.plot(
        uts_strain,
        uts_stress,
        marker="s",
        color="green",
        linestyle="none",
        label="UTS point",
    )

    if SHOW_POINT_LABELS:
        ax.annotate(
            f"({yield_strain:.5g}, {yield_stress:.5g})",
            (yield_strain, yield_stress),
            textcoords="offset points",
            xytext=(6, 6),
        )
        ax.annotate(
            f"({uts_strain:.5g}, {uts_stress:.5g})",
            (uts_strain, uts_stress),
            textcoords="offset points",
            xytext=(6, 6),
        )

    ax.set_title(sample_id)
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress [MPa]")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    return fig


def run() -> None:
    print(f"Reading zeroed dataset from '{ZEROED_INPUT_PATH}'...")
    df_raw = pd.read_csv(ZEROED_INPUT_PATH, header=None)
    print(f"  {df_raw.shape[0]} row(s) x {df_raw.shape[1]} column(s)")

    header_row = df_raw.iloc[0]
    samples = discover_samples(header_row)
    print(f"\nDetected {len(samples)} sample column pair(s): {[s[0] for s in samples]}")

    print(f"\nReading combined results from '{COMBINED_INPUT_PATH}'...")
    combined = load_combined(COMBINED_INPUT_PATH)
    print(f"  {len(combined)} sample row(s) loaded")
    combined_by_id = {row["Sample ID"]: row for _, row in combined.iterrows()}

    log: list[str] = []
    plotted = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zeroed_ids = {sample_id for sample_id, _, _ in samples}
    combined_ids = set(combined_by_id.keys())
    for missing_id in sorted(combined_ids - zeroed_ids):
        log.append(
            f"SKIP '{missing_id}': present in combined results file but not "
            f"in the zeroed dataset"
        )

    for sample_id, strain_idx, stress_idx in samples:
        if sample_id not in combined_by_id:
            log.append(
                f"SKIP '{sample_id}': present in the zeroed dataset but not "
                f"in the combined results file"
            )
            continue

        strain, stress = load_sample(df_raw, strain_idx, stress_idx)
        if len(strain) == 0:
            log.append(f"SKIP '{sample_id}': no data rows found in zeroed dataset")
            continue
        if strain.isna().any() or stress.isna().any():
            log.append(f"SKIP '{sample_id}': non-numeric/blank cell(s) found in zeroed dataset")
            continue

        row = combined_by_id[sample_id]
        try:
            slope = float(row["Modulus (Slope)"])
            yield_strain = float(row["Yield Strain"])
            yield_stress = float(row["Yield Stress"])
            uts_strain = float(row["Strain at Max Stress"])
            uts_stress = float(row["Max Stress"])
        except (TypeError, ValueError):
            log.append(
                f"SKIP '{sample_id}': missing or non-numeric Modulus (Slope)/"
                f"Yield Strain/Yield Stress/Strain at Max Stress/Max Stress "
                f"in combined results file"
            )
            continue

        fig = plot_sample(
            sample_id, strain, stress, slope, yield_strain, yield_stress, uts_strain, uts_stress
        )
        output_path = OUTPUT_DIR / f"{sample_id}_plot_{timestamp}.png"
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")
        fig.savefig(output_path)
        plt.close(fig)
        plotted += 1

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    print("\n--- Summary ---")
    print(f"  Plots generated: {plotted}")
    print(f"  Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
