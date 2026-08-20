"""
Combine-CSV-folder script.

Reads every top-level CSV in an input folder, where the first column of
each CSV is a Sample ID (positional, regardless of its header text), and
full-outer-merges them into one wide table keyed on Sample ID — one row
per Sample ID (with fan-out rows for any ID duplicated within a single
source file), columns from every file aligned by that ID. Cells with no
value for a given Sample ID/column combination are left blank rather
than filled in.

Two additional CSV inputs are merged in alongside the folder's files:
  - GEOMETRY_PATH — a geometry file (the same kind of file used as the
    static lookup in C1_paired_csv_multiplier.py / C1), merged the exact
    same way as every other input file (first column = Sample ID, full
    outer join). It typically lives outside INPUT_FOLDER, so it's given
    its own configured path rather than relying on folder discovery.
  - GRAYVALUE_PATH — a gray value file. Unlike the geometry file, this
    is NOT merged by Sample ID directly. For each sample already present
    in the combined output (e.g. "VT19_1_2"), a Gray Value ID is derived
    by dropping the sample ID's trailing "_<number>" suffix (e.g.
    "VT19_1"), then looked up in this file to pull in three columns:
    Gray Value ID, Gray Value, GV Std Dev. If the ID can't be derived
    (no trailing "_<number>") or has no match, those three columns are
    left blank for that sample — no new rows are ever added from this
    file.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependency: pandas.

This script is standalone — it does not import or depend on any other
script in this repository.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Folder to scan (top level only, not subfolders) for input CSV files.
INPUT_FOLDER = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code4_AInputs')

# Directory the combined output file is written into.
OUTPUT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs')

# Filename pattern for this script's own output, excluded from input
# discovery so re-running on the same folder doesn't ingest a prior
# combined output as input.
OUTPUT_FILENAME_PATTERN = "data_combined_*.csv"

# Path to the geometry CSV — defaults to the same file used as
# STATIC_LOOKUP_PATH in C1_paired_csv_multiplier.py (C1), but can be
# pointed anywhere. Merged in exactly like every other input file
# (first column = Sample ID, full outer join).
GEOMETRY_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/Geometry/Geometry_20260818.csv')

# Path to the gray value CSV. See the module docstring above for how
# each sample's Gray Value ID is derived and looked up.
GRAYVALUE_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/Code_Inputs,Outputs/grayvalue.csv')

# Column names expected in GRAYVALUE_PATH — used, unchanged, as the
# three new output column names.
GRAYVALUE_ID_COLUMN = "Gray Value ID"
GRAYVALUE_VALUE_COLUMN = "Gray Value"
GRAYVALUE_STDDEV_COLUMN = "GV Std Dev"

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
#  - Output filename: data_combined_<YYYYMMDD_HHMMSS>.csv, matching the
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


def drop_blank_ids(df: pd.DataFrame, path: Path, log: list[str]) -> pd.DataFrame:
    """Drop rows whose Sample ID is blank (NaN, empty, or whitespace-only) and log a warning.

    A blank Sample ID (e.g. from a stray trailing blank line in a source
    CSV) can't be merged or matched downstream — carrying it through as
    a NaN key mixes types in the Sample ID column and breaks any plain
    string sort/comparison done on it later (by this script or others
    downstream), so it's dropped here instead.
    """
    blank = df["Sample ID"].fillna("").astype(str).str.strip() == ""
    count = int(blank.sum())
    if count:
        log.append(
            f"WARNING '{path.name}': {count} row(s) with a blank Sample ID "
            f"dropped (e.g. a stray blank line) — not merged into the output"
        )
        df = df[~blank].reset_index(drop=True)
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


GRAY_VALUE_ID_PATTERN = re.compile(r"^(.+)_\d+$")


def derive_gray_value_id(sample_id: str) -> str | None:
    """Derive a sample's Gray Value ID by dropping its trailing "_<number>" suffix.

    e.g. "VT19_1_2" -> "VT19_1". Returns None if sample_id has no
    trailing underscore-number suffix to drop.
    """
    match = GRAY_VALUE_ID_PATTERN.match(sample_id)
    return match.group(1) if match else None


def load_grayvalue_lookup(path: Path) -> tuple[dict[str, tuple[object, object]], set[str]]:
    """Read the gray value CSV into {Gray Value ID: (Gray Value, GV Std Dev)}.

    Returns the lookup dict plus the set of Gray Value IDs that appear
    more than once (ambiguous — any sample matching one of these is left
    blank rather than guessing which row to use).
    """
    df = pd.read_csv(path, dtype=str)
    missing = [
        c
        for c in (GRAYVALUE_ID_COLUMN, GRAYVALUE_VALUE_COLUMN, GRAYVALUE_STDDEV_COLUMN)
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Gray value file '{path}' is missing required column(s): {missing}")

    counts: dict[str, int] = {}
    lookup: dict[str, tuple[object, object]] = {}
    for _, row in df.iterrows():
        gray_id = row[GRAYVALUE_ID_COLUMN]
        if pd.isna(gray_id) or str(gray_id).strip() == "":
            continue
        gray_id = str(gray_id).strip()
        counts[gray_id] = counts.get(gray_id, 0) + 1
        lookup[gray_id] = (row[GRAYVALUE_VALUE_COLUMN], row[GRAYVALUE_STDDEV_COLUMN])

    duplicate_ids = {gray_id for gray_id, count in counts.items() if count > 1}
    return lookup, duplicate_ids


def build_gray_value_columns(
    sample_ids: pd.Series,
    lookup: dict[str, tuple[object, object]],
    duplicate_ids: set[str],
    log: list[str],
) -> tuple[list, list, list]:
    """Derive each sample's Gray Value ID and look up its Gray Value / GV Std Dev.

    Returns three parallel lists (Gray Value ID, Gray Value, GV Std Dev),
    one entry per row in sample_ids, in the same order. A sample is left
    blank in all three lists if its Gray Value ID can't be derived, is
    ambiguous in the lookup file, or has no match there.
    """
    gray_ids: list = []
    gray_values: list = []
    gray_stddevs: list = []

    for sample_id in sample_ids:
        derived_id = derive_gray_value_id(str(sample_id))
        if derived_id is None:
            log.append(
                f"  '{sample_id}': Gray Value ID could not be derived (no "
                f"trailing \"_<number>\") — leaving Gray Value columns blank"
            )
        elif derived_id in duplicate_ids:
            log.append(
                f"  '{sample_id}': derived Gray Value ID '{derived_id}' is "
                f"ambiguous (appears more than once in the gray value file) "
                f"— leaving Gray Value columns blank"
            )
            derived_id = None
        elif derived_id not in lookup:
            log.append(
                f"  '{sample_id}': derived Gray Value ID '{derived_id}' not "
                f"found in the gray value file — leaving Gray Value columns blank"
            )
            derived_id = None

        if derived_id is None:
            gray_ids.append(None)
            gray_values.append(None)
            gray_stddevs.append(None)
        else:
            value, stddev = lookup[derived_id]
            gray_ids.append(derived_id)
            gray_values.append(value)
            gray_stddevs.append(stddev)

    return gray_ids, gray_values, gray_stddevs


def run() -> None:
    print(f"Scanning '{INPUT_FOLDER}' for CSV files...")
    csv_files = discover_csv_files(INPUT_FOLDER, OUTPUT_FILENAME_PATTERN)
    print(f"  found {len(csv_files)} CSV file(s): {[p.name for p in csv_files]}")

    log: list[str] = []
    loaded: list[tuple[str, pd.DataFrame]] = []

    for path in csv_files:
        try:
            df = load_file(path)
        except ValueError as exc:
            log.append(f"SKIP '{path.name}': {exc}")
            continue

        df = drop_blank_ids(df, path, log)
        if df.empty:
            log.append(f"SKIP '{path.name}': no rows remain after dropping blank Sample ID row(s)")
            continue

        report_duplicate_ids(df, path, log)
        loaded.append((path.stem, df))

    print(f"\nLoading geometry file '{GEOMETRY_PATH}'...")
    geometry_df = load_file(GEOMETRY_PATH)
    geometry_df = drop_blank_ids(geometry_df, GEOMETRY_PATH, log)
    report_duplicate_ids(geometry_df, GEOMETRY_PATH, log)
    loaded.append((GEOMETRY_PATH.stem, geometry_df))

    combined = merge_all(loaded, log)
    combined = combined.sort_values("Sample ID", kind="stable").reset_index(drop=True)

    print(f"\nLoading gray value file '{GRAYVALUE_PATH}'...")
    gray_lookup, gray_duplicate_ids = load_grayvalue_lookup(GRAYVALUE_PATH)
    gray_ids, gray_values, gray_stddevs = build_gray_value_columns(
        combined["Sample ID"], gray_lookup, gray_duplicate_ids, log
    )
    combined[GRAYVALUE_ID_COLUMN] = gray_ids
    combined[GRAYVALUE_VALUE_COLUMN] = gray_values
    combined[GRAYVALUE_STDDEV_COLUMN] = gray_stddevs
    matched = sum(1 for gid in gray_ids if gid is not None)

    print("\n--- Skipped / logged items ---")
    if log:
        for line in log:
            print(f"  {line}")
    else:
        print("  (none)")

    # Reorder columns: Sample ID first, everything else as produced by the merges.
    other_cols = [c for c in combined.columns if c != "Sample ID"]
    combined = combined[["Sample ID"] + other_cols]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"data_combined_{timestamp}.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output file: {output_path}")

    combined.to_csv(output_path, index=False)

    print("\n--- Summary ---")
    print(f"  Files combined: {len(loaded)} -> {[name for name, _ in loaded]}")
    print(f"  Unique Sample ID rows in output: {len(combined)}")
    print(f"  Samples with a matched Gray Value: {matched} / {len(combined)}")
    print(f"  Output columns: {list(combined.columns)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    run()
