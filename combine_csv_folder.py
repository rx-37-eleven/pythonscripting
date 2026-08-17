"""
Combine-CSV-folder script.

Reads every top-level CSV in an input folder, where the first column of
each CSV is a Sample ID (positional, regardless of its header text), and
full-outer-merges them into one wide table keyed on Sample ID — one row
per Sample ID (with fan-out rows for any ID duplicated within a single
source file), columns from every file aligned by that ID. Cells with no
value for a given Sample ID/column combination are left blank rather
than filled in.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependency: pandas.

This script is standalone — it does not import or depend on any other
script in this repository.
"""

from __future__ import annotations

import fnmatch
from datetime import datetime
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Folder to scan (top level only, not subfolders) for input CSV files.
INPUT_FOLDER = Path("input_csvs")

# Directory the combined output file is written into.
OUTPUT_DIR = Path("outputs")

# Filename pattern for this script's own output, excluded from input
# discovery so re-running on the same folder doesn't ingest a prior
# combined output as input.
OUTPUT_FILENAME_PATTERN = "DataCombined_*.csv"

# ---------------------------------------------------------------------
# Resolved answers to the brief's open questions (captured here per the
# brief's "definition of done"):
#
#  - Core combining logic: MERGE/JOIN on Sample ID (not row stacking) —
#    each file contributes columns for the same set of samples; output
#    is one wide row per Sample ID.
#  - Join type: FULL OUTER join — every Sample ID found in any file
#    appears in the output; missing values left blank (NaN -> empty
#    CSV cell), never filled in.
#  - Column-name collisions (same non-ID column name in >1 file):
#    if every sample's value agrees wherever both files have a
#    non-blank value for that column, the columns are coalesced into
#    one. If any sample's values genuinely conflict, BOTH columns are
#    kept, renamed with a "_<source file stem>" suffix so no data is
#    lost or silently overwritten.
#  - Duplicate Sample ID within a single file: NOT deduplicated — both
#    (all) rows are kept and logged as a warning. Because the output is
#    a wide join, a duplicated ID fans out into multiple output rows
#    (each duplicate row merged separately against other files' single
#    matching row for that ID) — standard join fan-out behavior.
#  - Sample ID matching: positional — the FIRST column of every file,
#    regardless of its header text, is treated as Sample ID and
#    renamed to "Sample ID" in the output. Matched by EXACT string
#    equality — no trimming/whitespace normalization, no numeric
#    coercion (e.g. "001" and "1" are distinct IDs).
#  - File discovery: top level of INPUT_FOLDER only (no subfolders).
#    Every *.csv file is treated as input EXCEPT files matching this
#    script's own OUTPUT_FILENAME_PATTERN.
#  - Header format: every input CSV has a single, standard header row
#    in row 1 (data from row 2) — not the multi-row-header convention
#    used by the zeroing/yield-point scripts earlier in this series.
#  - Output row order: sorted by Sample ID (string sort).
#  - Output filename: DataCombined_<YYYYMMDD_HHMMSS>.csv, matching the
#    earlier scripts' timestamp-to-the-second convention. Never
#    overwrites an existing file of the same name.
#  - Interface: CONFIG block (edited before running), matching the
#    prior two scripts' convention — not CLI arguments.
# =====================================================================


def discover_csv_files(input_folder: Path, output_pattern: str) -> list[Path]:
    """List top-level *.csv files in input_folder, excluding this script's own output pattern."""
    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    files = [
        p
        for p in sorted(input_folder.iterdir())
        if p.is_file()
        and p.suffix.lower() == ".csv"
        and not fnmatch.fnmatch(p.name, output_pattern)
    ]
    return files


def load_file(path: Path) -> pd.DataFrame:
    """Read one CSV; rename its first column to 'Sample ID' positionally."""
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    if df.shape[1] == 0:
        raise ValueError(f"'{path}' has no columns")
    if df.shape[0] == 0:
        raise ValueError(f"'{path}' has a header row but no data rows")

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "Sample ID"})
    return df


def report_duplicate_ids(df: pd.DataFrame, path: Path, log: list[str]) -> None:
    """Log a warning for any Sample ID that appears more than once in df. Rows are kept, not removed."""
    counts = df["Sample ID"].value_counts()
    duplicated = counts[counts > 1]
    for sample_id, count in duplicated.items():
        log.append(
            f"WARNING '{path.name}': Sample ID '{sample_id}' appears {count} "
            f"times in this file — all rows kept, will fan out in the merge"
        )


def merge_two(
    left: pd.DataFrame, left_name: str, right: pd.DataFrame, right_name: str, log: list[str]
) -> pd.DataFrame:
    """Full-outer-merge two frames on Sample ID, resolving overlapping column names.

    Overlapping non-ID columns: coalesced into one column if every row
    where both sides have a non-blank value agrees; otherwise both
    columns are kept, suffixed with "_<source file stem>".
    """
    overlap = [c for c in left.columns if c != "Sample ID" and c in right.columns]

    merged = left.merge(
        right,
        on="Sample ID",
        how="outer",
        suffixes=(f"_{left_name}", f"_{right_name}"),
    )

    for col in overlap:
        left_col = f"{col}_{left_name}"
        right_col = f"{col}_{right_name}"
        both_present = merged[left_col].notna() & merged[right_col].notna()
        conflicts = both_present & (merged[left_col] != merged[right_col])

        if conflicts.any():
            log.append(
                f"WARNING: column '{col}' has conflicting values between "
                f"'{left_name}' and '{right_name}' for {int(conflicts.sum())} "
                f"sample(s) — keeping both as '{left_col}' / '{right_col}'"
            )
        else:
            merged[col] = merged[left_col].where(merged[left_col].notna(), merged[right_col])
            merged = merged.drop(columns=[left_col, right_col])

    return merged


def merge_all(loaded: list[tuple[str, pd.DataFrame]], log: list[str]) -> pd.DataFrame:
    """Full-outer-merge every (source_name, df) pair in order, on Sample ID."""
    source_name, combined = loaded[0]
    for next_name, next_df in loaded[1:]:
        combined = merge_two(combined, source_name, next_df, next_name, log)
        source_name = f"{source_name}+{next_name}"
    return combined


def run() -> None:
    print(f"Scanning '{INPUT_FOLDER}' for CSV files...")
    csv_files = discover_csv_files(INPUT_FOLDER, OUTPUT_FILENAME_PATTERN)
    print(f"  found {len(csv_files)} CSV file(s): {[p.name for p in csv_files]}")

    if not csv_files:
        print("\nNo CSV files found in the input folder — nothing to combine.")
        return

    log: list[str] = []
    loaded: list[tuple[str, pd.DataFrame]] = []

    for path in csv_files:
        try:
            df = load_file(path)
        except ValueError as exc:
            log.append(f"SKIP '{path.name}': {exc}")
            continue

        report_duplicate_ids(df, path, log)
        loaded.append((path.stem, df))

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    if not loaded:
        print("\nNo usable CSV files — no output file written.")
        return

    combined = merge_all(loaded, log)
    combined = combined.sort_values("Sample ID", kind="stable").reset_index(drop=True)

    # Reorder columns: Sample ID first, everything else as produced by the merges.
    other_cols = [c for c in combined.columns if c != "Sample ID"]
    combined = combined[["Sample ID"] + other_cols]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"DataCombined_{timestamp}.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")

    combined.to_csv(output_path, index=False)

    print("\n--- Summary ---")
    print(f"  Files combined: {len(loaded)} -> {[name for name, _ in loaded]}")
    print(f"  Unique Sample ID rows in output: {len(combined)}")
    print(f"  Output columns: {list(combined.columns)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    run()
