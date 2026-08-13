"""
Stress-strain zeroing and summary script.

Reads a single CSV containing an unknown number of samples, laid out as
adjacent (strain, stress) column pairs — one pair per sample. Row 1 of
each sample's stress column holds that sample's ID; row 2 is a
header/units row; data starts on row 3. Column pairs are ragged (samples
may have different numbers of data rows) and the number of sample pairs
is not known in advance — it's detected by scanning row 1.

For each sample:
  - Fits a line (linear regression, strain = x, stress = y) through the
    first FIT_POINTS data rows (or fewer, if the sample doesn't have
    that many).
  - Shifts every strain value in the sample by (intercept / slope), so
    the fitted line passes through the origin. Stress values are left
    unchanged.
  - Prepends a (0, 0) point to the shifted dataset.
  - Finds the max stress point (and its strain) and the failure point
    (the last row of the sample's original data, after zeroing).

Writes two timestamped output CSVs (same run timestamp, to the second):
  - zeroed_outputs_<timestamp>.csv — same wide/alternating layout as the
    input (Sample ID in row 1, header/units row preserved in row 2,
    shorter samples blank-padded to the longest sample's length).
  - summary_stats_<timestamp>.csv — one row per sample: Sample ID,
    Modulus (Slope), x-intercept, y-intercept, R^2, Strain at Max
    Stress, Max Stress, Strain at Failure, Stress at Failure.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependency: pandas (also used to read/write CSVs).

This script is standalone — it does not import or depend on any other
script in this repository.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Path to the input CSV (alternating strain/stress column pairs).
INPUT_PATH = Path("input.csv")

# Directory the two output files are written into.
OUTPUT_DIR = Path("outputs")

# Number of leading data points used to fit each sample's line.
FIT_POINTS = 100

# ---------------------------------------------------------------------
# Resolved answers to the brief's open questions (captured here per the
# brief's "definition of done"):
#
#  - Input format: CSV. Row 1 of each pair is blank in the strain
#    (first) column and holds the Sample ID in the stress (second)
#    column. Row 2 is a header/units row, preserved verbatim in the
#    output. Data starts on row 3.
#  - Sample-pair detection: scan row 1 across all columns for non-blank
#    cells in stress-column positions (odd 0-based index within each
#    pair) — every such cell marks one more (strain, stress) pair.
#    Columns are consumed two at a time; a pair whose stress cell is
#    blank ends the scan.
#  - Regression convention: strain = x (independent), stress = y
#    (dependent) — slope is the modulus.
#  - Fit window: the first FIT_POINTS rows positionally (rows 3..3+N-1
#    of the sample's columns), not "first N non-blank" — internal gaps
#    are not specially skipped. If a sample has fewer than FIT_POINTS
#    data rows, all available rows are used for the fit.
#  - Zeroing: new_strain = strain + intercept/slope (shifts the line's
#    x-intercept to 0); stress is left unchanged; the shift applies to
#    the sample's ENTIRE dataset, not just the fit window. A (0, 0)
#    point is then prepended as the new first row of that sample.
#  - Max stress: found by scanning the full zeroed dataset (INCLUDING
#    the prepended (0, 0) point, per instruction — it is never
#    actually the max, but is included in the search range).
#  - Failure point: the last row of the sample's own data (after
#    zeroing) — i.e. the last original data row, independent of how
#    long any other sample is.
#  - Output 1 layout: same wide/alternating layout as the input — row 1
#    Sample ID per pair, row 2 = the input's own header/units row
#    copied through unchanged, data from row 3, shorter samples
#    blank-padded (via NaN -> empty cell) to the longest sample.
#  - Output 2: CSV, columns in order: "Sample ID", "Modulus (Slope)",
#    "x-intercept", "y-intercept", "R^2", "Strain at Max Stress",
#    "Max Stress", "Strain at Failure", "Stress at Failure". x-intercept
#    is the fitted line's strain-axis crossing (-intercept/slope) — the
#    value actually used to shift strain during zeroing. y-intercept is
#    the raw fitted intercept from np.polyfit (stress value at
#    strain=0).
#  - Both outputs share one run timestamp (to the second) in their
#    filenames and never overwrite an existing file of the same name.
#  - Interface: CONFIG block (edited before running), matching this
#    repo's existing script convention — not a CLI argument.
#  - Malformed/undersized/non-numeric samples are logged and the
#    offending sample is skipped rather than silently miscomputed.
# =====================================================================


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

    Trims trailing rows where both cells are blank (handles ragged
    column lengths), then coerces the remaining cells to numeric.
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


def fit_line(
    strain: pd.Series, stress: pd.Series, n_points: int
) -> tuple[float, float, float]:
    """Fit stress = slope * strain + intercept over the first n_points rows.

    Uses all available rows if the sample has fewer than n_points.
    Returns (slope, intercept, r_squared).
    """
    x = strain.iloc[:n_points].to_numpy(dtype=float)
    y = stress.iloc[:n_points].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual_ss = float(np.sum((y - predicted) ** 2))
    total_ss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss != 0 else float("nan")
    return float(slope), float(intercept), r_squared


def zero_sample(
    strain: pd.Series, stress: pd.Series, slope: float, intercept: float
) -> tuple[pd.Series, pd.Series]:
    """Shift strain so the fitted line passes through the origin, then prepend (0, 0)."""
    shift = intercept / slope
    zeroed_strain = strain + shift
    zeroed_strain = pd.concat([pd.Series([0.0]), zeroed_strain], ignore_index=True)
    zeroed_stress = pd.concat([pd.Series([0.0]), stress], ignore_index=True)
    return zeroed_strain, zeroed_stress


def summarize_sample(
    sample_id: str,
    slope: float,
    intercept: float,
    r_squared: float,
    zeroed_strain: pd.Series,
    zeroed_stress: pd.Series,
) -> dict:
    max_idx = zeroed_stress.idxmax()
    max_stress = zeroed_stress.loc[max_idx]
    strain_at_max = zeroed_strain.loc[max_idx]

    strain_at_failure = zeroed_strain.iloc[-1]
    stress_at_failure = zeroed_stress.iloc[-1]

    x_intercept = -intercept / slope

    return {
        "Sample ID": sample_id,
        "Modulus (Slope)": slope,
        "x-intercept": x_intercept,
        "y-intercept": intercept,
        "R^2": r_squared,
        "Strain at Max Stress": strain_at_max,
        "Max Stress": max_stress,
        "Strain at Failure": strain_at_failure,
        "Stress at Failure": stress_at_failure,
    }


def build_wide_output(
    processed: list[tuple[str, pd.Series, pd.Series]], units_row: pd.Series
) -> pd.DataFrame:
    """Reassemble processed samples into the same wide/alternating layout as the input.

    Row 1 (index 0 in the returned frame) is the Sample ID per pair.
    Row 2 (index 1) is the input's own header/units row (row 2 of the
    input) for those two columns, copied through unchanged. Data
    follows from row 3. Shorter samples are blank-padded (NaN) to the
    longest sample's length.
    """
    max_len = max((len(strain) for _, strain, _ in processed), default=0)

    columns: dict[int, pd.Series] = {}
    col = 0
    for sample_id, zeroed_strain, zeroed_stress in processed:
        strain_idx, stress_idx = col, col + 1

        padded_strain = zeroed_strain.reindex(range(max_len))
        padded_stress = zeroed_stress.reindex(range(max_len))

        strain_header_val = units_row.iloc[strain_idx] if strain_idx < len(units_row) else None
        stress_header_val = units_row.iloc[stress_idx] if stress_idx < len(units_row) else None

        strain_rows = [None, strain_header_val] + list(padded_strain)
        stress_rows = [sample_id, stress_header_val] + list(padded_stress)

        columns[strain_idx] = pd.Series(strain_rows)
        columns[stress_idx] = pd.Series(stress_rows)
        col += 2

    ordered = [columns[i] for i in sorted(columns)]
    return pd.concat(ordered, axis=1, ignore_index=True)


def run() -> None:
    print(f"Reading input file '{INPUT_PATH}'...")
    df_raw = pd.read_csv(INPUT_PATH, header=None)
    print(f"  {df_raw.shape[0]} row(s) x {df_raw.shape[1]} column(s)")

    header_row = df_raw.iloc[0]
    samples = discover_samples(header_row)
    print(f"\nDetected {len(samples)} sample column pair(s): {[s[0] for s in samples]}")

    processed: list[tuple[str, pd.Series, pd.Series]] = []
    summary_rows: list[dict] = []
    log: list[str] = []

    for sample_id, strain_idx, stress_idx in samples:
        strain, stress = load_sample(df_raw, strain_idx, stress_idx)

        if len(strain) == 0:
            log.append(f"SKIP '{sample_id}': no data rows found")
            continue

        bad_strain = strain.isna().sum()
        bad_stress = stress.isna().sum()
        if bad_strain or bad_stress:
            log.append(
                f"SKIP '{sample_id}': non-numeric/blank cell(s) found "
                f"(strain: {bad_strain}, stress: {bad_stress})"
            )
            continue

        n_fit = min(FIT_POINTS, len(strain))
        if n_fit < FIT_POINTS:
            log.append(
                f"  '{sample_id}': only {n_fit} data point(s) available, "
                f"fewer than FIT_POINTS={FIT_POINTS} — using all of them for the fit"
            )

        slope, intercept, r_squared = fit_line(strain, stress, n_fit)
        if slope == 0:
            log.append(f"SKIP '{sample_id}': fitted slope is zero — cannot zero strain")
            continue

        zeroed_strain, zeroed_stress = zero_sample(strain, stress, slope, intercept)
        processed.append((sample_id, zeroed_strain, zeroed_stress))
        summary_rows.append(
            summarize_sample(sample_id, slope, intercept, r_squared, zeroed_strain, zeroed_stress)
        )

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    if not processed:
        print("\nNo samples were successfully processed — no output files written.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zeroed_path = OUTPUT_DIR / f"zeroed_outputs_{timestamp}.csv"
    if zeroed_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {zeroed_path}")
    units_row = df_raw.iloc[1]
    wide_output = build_wide_output(processed, units_row)
    wide_output.to_csv(zeroed_path, index=False, header=False)

    summary_path = OUTPUT_DIR / f"summary_stats_{timestamp}.csv"
    if summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {summary_path}")
    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "Sample ID",
            "Modulus (Slope)",
            "x-intercept",
            "y-intercept",
            "R^2",
            "Strain at Max Stress",
            "Max Stress",
            "Strain at Failure",
            "Stress at Failure",
        ],
    )
    summary_df.to_csv(summary_path, index=False)

    print("\n--- Summary ---")
    print(f"  Samples processed successfully: {len(processed)} -> {[p[0] for p in processed]}")
    print(f"  Samples skipped: {len(samples) - len(processed)}")
    print(f"  Zeroed output file: {zeroed_path}")
    print(f"  Summary stats file: {summary_path}")


if __name__ == "__main__":
    run()
