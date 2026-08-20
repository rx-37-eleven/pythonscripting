"""
0.2% offset yield point script.

Reads a zeroed stress-strain dataset (the output of zero_stress_strain.py:
wide, alternating (strain, stress) column pairs per sample, Sample ID in
row 1 of each stress column, header/units row in row 2, data from row 3,
each sample's data starting at (0, 0)) plus that same run's summary
stats CSV (for per-sample Modulus (Slope), joined on Sample ID).

For each sample:
  - Builds the 0.2% offset line: offset_stress(strain) = slope *
    (strain - OFFSET), where slope is the sample's elastic-region slope
    from the summary stats file.
  - Scans the sample's data in order of increasing strain, computing
    diff = actual_stress - offset_stress at each point, looking for
    point-to-point sign changes (the offset line crossing the actual
    curve). If more than one crossing is found, the LAST one (highest
    strain) is used.
  - Linearly interpolates the ACTUAL data curve at the intersection
    strain (found by interpolating the diff values to zero) to report
    Yield Strain and Yield Stress.
  - Computes Plastic Strain to Failure = (last strain value in the
    sample's own zeroed data) - Yield Strain.

A sample with zero sign changes (the offset line never crosses the
curve) is logged as a warning and written with blank Yield Strain /
Yield Stress / Plastic Strain to Failure values.

Writes one timestamped output CSV: yield_point_summary_<timestamp>.csv,
columns: Sample ID, Yield Strain, Yield Stress, Plastic Strain to Failure.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependency: pandas.

This script is standalone — it does not import or depend on any other
script in this repository, including zero_stress_strain.py. It merely
consumes that script's output files as input.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Path to the zeroed dataset CSV (output of zero_stress_strain.py's
# zeroed_outputs_<timestamp>.csv).
ZEROED_INPUT_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/zeroed_outputs_20260819_190935.csv')

# Path to the matching summary stats CSV (output of
# zero_stress_strain.py's summary_stats_<timestamp>.csv) — used to look
# up each sample's elastic-region Slope, joined on Sample ID.
SUMMARY_STATS_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/summary_stats_20260819_190935.csv')

# Directory the output file is written into.
OUTPUT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs')

# 0.2% offset, expressed as a decimal strain fraction.
OFFSET = 0.002

# ---------------------------------------------------------------------
# Resolved answers to the brief's open questions (captured here per the
# brief's "definition of done"):
#
#  - Input structure: confirmed directly (this script's author also
#    wrote zero_stress_strain.py) — wide/alternating (strain, stress)
#    column pairs, Sample ID in row 1 of each stress column, units row
#    in row 2, data from row 3, strain stored as decimal fractions
#    (0.002 = 0.2%), samples blank-padded to the longest sample.
#  - Input location: CONFIG block (explicit paths), not CLI args or
#    auto-discovery — matches this repo's existing script convention.
#  - Slope source: read from the prior script's summary_stats CSV
#    "Modulus (Slope)" column, joined on Sample ID — not re-derived by
#    re-fitting.
#  - Intersection method: scan data points in increasing-strain order,
#    diff = actual_stress - offset_line_stress, look for point-to-point
#    sign changes. One or more sign changes -> valid yield point, using
#    the LAST crossing (highest strain) when more than one is found,
#    resolved by linearly interpolating the diff to zero for strain,
#    then linearly interpolating the ACTUAL stress-strain curve
#    (not the offset line) at that strain for Yield Stress. Zero sign
#    changes -> logged as a warning, row written with blank Yield
#    Strain / Yield Stress / Plastic Strain to Failure.
#  - Plastic Strain to Failure = (last strain value in this script's
#    own read of the sample's zeroed data) - Yield Strain — not read
#    from the prior script's "Strain at Failure" stats column.
#  - Output: CSV, yield_point_summary_<timestamp>.csv, columns "Sample
#    ID", "Yield Strain", "Yield Stress", "Plastic Strain to Failure".
#  - No plots — console logging of progress/warnings only.
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

    Trims trailing rows where both cells are blank (handles blank-padded
    shorter samples), then coerces the remaining cells to numeric.
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


def load_slopes(summary_stats_path: Path) -> dict[str, float]:
    """Read Sample ID -> Modulus (Slope) from the prior script's summary stats CSV."""
    df = pd.read_csv(summary_stats_path)
    if "Sample ID" not in df.columns or "Modulus (Slope)" not in df.columns:
        raise ValueError(
            f"Summary stats file '{summary_stats_path}' is missing required "
            f"'Sample ID' and/or 'Modulus (Slope)' column(s)"
        )
    return {
        str(row["Sample ID"]).strip(): float(row["Modulus (Slope)"])
        for _, row in df.iterrows()
    }


def find_yield_point(
    strain: pd.Series, stress: pd.Series, slope: float, offset: float
) -> tuple[float, float] | None:
    """Find the 0.2%-offset yield point via a sign-change crossing.

    If more than one point-to-point sign change is found, the LAST one
    (highest strain) is used. Returns (yield_strain, yield_stress), or
    None if no sign change is found at all (caller should log and skip).
    """
    x = strain.to_numpy(dtype=float)
    y = stress.to_numpy(dtype=float)
    offset_stress = slope * (x - offset)
    diff = y - offset_stress

    crossings = []
    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 == 0:
            crossings.append(i)
        elif d0 * d1 < 0:
            crossings.append(i)

    if not crossings:
        return None

    i = crossings[-1]
    if diff[i] == 0:
        return float(x[i]), float(y[i])

    x0, x1 = x[i], x[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    frac = d0 / (d0 - d1)
    yield_strain = x0 + frac * (x1 - x0)

    y0, y1 = y[i], y[i + 1]
    yield_stress = y0 + frac * (y1 - y0)

    return float(yield_strain), float(yield_stress)


def run() -> None:
    print(f"Reading slope lookup from '{SUMMARY_STATS_PATH}'...")
    slopes = load_slopes(SUMMARY_STATS_PATH)
    print(f"  {len(slopes)} sample slope(s) loaded")

    print(f"\nReading zeroed dataset from '{ZEROED_INPUT_PATH}'...")
    df_raw = pd.read_csv(ZEROED_INPUT_PATH, header=None)
    print(f"  {df_raw.shape[0]} row(s) x {df_raw.shape[1]} column(s)")

    header_row = df_raw.iloc[0]
    samples = discover_samples(header_row)
    print(f"\nDetected {len(samples)} sample column pair(s): {[s[0] for s in samples]}")

    rows: list[dict] = []
    log: list[str] = []

    for sample_id, strain_idx, stress_idx in samples:
        strain, stress = load_sample(df_raw, strain_idx, stress_idx)

        if len(strain) == 0:
            log.append(f"SKIP '{sample_id}': no data rows found")
            continue

        if strain.isna().any() or stress.isna().any():
            log.append(f"SKIP '{sample_id}': non-numeric/blank cell(s) found")
            continue

        if sample_id not in slopes:
            log.append(
                f"SKIP '{sample_id}': no matching Modulus (Slope) found in "
                f"'{SUMMARY_STATS_PATH}'"
            )
            continue

        slope = slopes[sample_id]
        if slope == 0:
            log.append(f"SKIP '{sample_id}': slope is zero — cannot build offset line")
            continue

        last_strain = float(strain.iloc[-1])
        yield_point = find_yield_point(strain, stress, slope, OFFSET)

        if yield_point is None:
            log.append(
                f"WARNING '{sample_id}': offset line never crosses the actual "
                f"curve — writing blank yield values"
            )
            rows.append(
                {
                    "Sample ID": sample_id,
                    "Yield Strain": None,
                    "Yield Stress": None,
                    "Plastic Strain to Failure": None,
                }
            )
            continue

        yield_strain, yield_stress = yield_point
        rows.append(
            {
                "Sample ID": sample_id,
                "Yield Strain": yield_strain,
                "Yield Stress": yield_stress,
                "Plastic Strain to Failure": last_strain - yield_strain,
            }
        )

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    if not rows:
        print("\nNo samples were successfully processed — no output file written.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"yield_point_summary_{timestamp}.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")

    output_df = pd.DataFrame(
        rows,
        columns=["Sample ID", "Yield Strain", "Yield Stress", "Plastic Strain to Failure"],
    )
    output_df.to_csv(output_path, index=False)

    print("\n--- Summary ---")
    valid = sum(1 for r in rows if r["Yield Stress"] is not None)
    print(f"  Samples with a valid yield point: {valid} / {len(rows)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    run()
