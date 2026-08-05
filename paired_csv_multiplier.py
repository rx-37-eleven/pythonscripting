"""
Paired-CSV column extractor and multiplier.

Walks a directory tree for <base>.csv / <base>_ex.csv pairs, pulls one
column out of each file, computes (<base> * 1000) / (val2 * val3) using
per-base val2/val3 looked up from a static CSV (the _ex column is
written unchanged, no math applied), and writes the results into one
combined output CSV with two header rows: the pair name, then
"strain"/"stress" per column. Each pair's _ex column is written before
its base column. Rows whose <base>.csv source-column value is below
BASE_MIN_VALUE are dropped from the pair (see CONFIG below). A second
output CSV is also written alongside the main one, summarizing the
peak (max) stress value per pair in a compact two-row layout.

Run this from Spyder: edit the CONFIG block below, then press Run.
Non-stdlib dependency: pandas (also used to read/write CSVs).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Root directory to search recursively (all subfolders, any depth) for
# paired data CSVs.
ROOT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron')

# Path to the static lookup CSV (label, val2, val3).
STATIC_LOOKUP_PATH = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/Geometry/Geometry_20260804.csv')

# Directory + base filename for the output. A timestamp is appended on
# every run so a previous output is never silently overwritten.
OUTPUT_DIR = Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/StressStrainFiles')
OUTPUT_BASENAME = "multiplied_output"

# Basename for the second output file: a per-pair peak-stress summary,
# written into OUTPUT_DIR alongside the main output with the SAME
# timestamp (see build_output_path / run()).
MAX_OUTPUT_BASENAME = "max_stress"

# Full folder paths to exclude from the recursive search — any .csv
# file that lives inside one of these directories, or any of their
# subdirectories, is skipped entirely before pairing even runs. Useful
# for excluding the output folder (so old output files never get
# rescanned as data) or any other subtree that shouldn't be searched.
# Example paths (edit/remove to match your tree):
EXCLUDED_FOLDER_PATHS: set[Path] = {
    Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/PythonCode/StressStrainFiles'),
    Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/Geometry'),
    Path('/Users/rcaraway3/Dropbox/Research/Garmestani,Neu/TAMU,GT,EOS/Instron/Archive'),
}

# --- Column positions ---
# 0-based indexing: 0 is the FIRST column, 1 the second, 2 the third, etc.

# Column to pull out of the <base>.csv file. Independent from the _ex
# index below — the two files are not required to use the same column.
# Example: SOURCE_COL_INDEX_BASE = 2 means the THIRD column.
SOURCE_COL_INDEX_BASE = 9

# Column to pull out of the <base>_ex.csv file.
SOURCE_COL_INDEX_EX = 13

# Columns in the static lookup file:
#   STATIC_LABEL_COL_INDEX -> the label, matched against <base>
#   STATIC_VAL2_COL_INDEX  -> val2 (used with val3 in the <base> formula below)
#   STATIC_VAL3_COL_INDEX  -> val3 (used with val2 in the <base> formula below)
STATIC_LABEL_COL_INDEX = 0
STATIC_VAL2_COL_INDEX = 1
STATIC_VAL3_COL_INDEX = 2

# Suffix that marks the "_ex" half of a pair. Kept as a constant (rather
# than hardcoded inline) so other suffixes can be supported later.
PAIR_SUFFIX = "_ex"

# Second output header row, written under each pair's column names —
# ROW2_LABEL_BASE under every <base> column, ROW2_LABEL_EX under every
# <base>_ex column. Output columns are ordered <base>_ex then <base> per
# pair, so output row 2 reads "strain,stress,strain,stress,...".
ROW2_LABEL_BASE = "stress"
ROW2_LABEL_EX = "strain"

# Rows whose <base>.csv source-column value is strictly less than this
# are dropped from the pair (the aligned _ex row is dropped too). This
# applies to the RAW value read from the base CSV, before the
# (value * 1000) / (val2 * val3) math — flip to compare against the
# post-math value if that turns out to be the intent instead. Set to
# None to disable filtering.
BASE_MIN_VALUE = 1.0

# Number of extra header-like rows that appear in every _ex file
# immediately after its normal header row (before real data starts).
# These rows are skipped when reading _ex files so their data rows
# align 1:1 with the base file's data rows.
EX_EXTRA_HEADER_ROWS = 1

# ---------------------------------------------------------------------
# Resolved answers to the brief's open questions (captured here per the
# brief's "definition of done"):
#
#  1/2. Output shape/format: ONE combined CSV, all pairs as columns
#       side by side (not one file per pair).
#  3.   Row alignment: positional (row N <-> row N), not key-joined. If
#       a pair's base/_ex files have different row counts, that pair is
#       SKIPPED (logged) rather than truncated.
#  4.   Headers: every CSV — data files AND the static lookup file —
#       has a header row. _ex files additionally have ONE EXTRA
#       header-like row right after the normal header (two header
#       lines total before real data starts). That extra row is
#       skipped when reading every _ex file (see EX_EXTRA_HEADER_ROWS
#       below), so _ex data rows line up 1:1 with the base file's data
#       rows — a length mismatch after that skip is still an error.
#  5.   Missing lookup label: pair is SKIPPED (logged); not an error,
#       not treated as multiplier = 1.
#  6.   Label matching: EXACT match, case-sensitive, no whitespace
#       trimming.
#  7.   Pairing: a base file and its _ex partner may live ANYWHERE in
#       the tree (matched purely by filename stem, not by folder).
#  8.   Duplicate base names found in more than one folder: NOT
#       expected in practice -> flagged as a CONFLICT and skipped
#       (never silently merged or arbitrarily picked).
#  9.   Suffix is configurable (PAIR_SUFFIX above) so other suffixes
#       can be supported later without touching the pairing logic.
#  10.  Bad data (blank cell / non-numeric text / NaN) anywhere in the
#       source column: the whole PAIR is skipped (logged) — not
#       treated as zero, not silently dropped row by row.
#  11.  No rounding — computed values are written at full float
#       precision.
#  - Row filter: rows whose <base>.csv source-column RAW value is below
#    BASE_MIN_VALUE are dropped from the pair (both base and _ex, to
#    stay aligned); if that removes every row, the pair is SKIPPED
#    (logged).
#
# Output math (per pair):
#   <base> column    = (<base> source value * 1000) / (val2 * val3)
#   <base>_ex column = <base>_ex source value, UNCHANGED — no math applied.
#   If val2 * val3 == 0 for a pair's label, that pair is SKIPPED (logged)
#   rather than dividing by zero.
# Output column order: <base>_ex before <base> for each pair (e.g.
# "fl15_ex,fl15,fl16_ex,fl16,..."). Output header: two rows. Row 1 is
# the pair's column name. Row 2 is ROW2_LABEL_EX/ROW2_LABEL_BASE per
# column, i.e. "strain,stress,strain,stress,..." across a full row.
#  12.  Scale: expected up to hundreds of pairs / thousands of rows per
#       file — everything is loaded with pandas in memory, no
#       streaming required.
#
# Additional resolved details:
#  - Combined-output padding: pairs with fewer rows than the longest
#    pair are padded with blank cells (achieved via NaN, which pandas
#    writes as an empty cell in CSV output).
#  - Output collision: the output filename is timestamped, so re-runs
#    never overwrite a previous output file.
#  - (Extension of #8's spirit, not an explicit brief question): if the
#    static lookup file lists the same label more than once, any pair
#    matching that label is skipped as AMBIGUOUS rather than guessing
#    which row to use.
#  - Peak-stress output: a second CSV summarizing the max stress per
#    pair, built from the `combined` DataFrame already in memory (not
#    re-read from disk). Only columns whose row-2 label is
#    ROW2_LABEL_BASE are included (strain/_ex columns are ignored). Max
#    uses pandas' default NaN-skipping behavior, so blank padding cells
#    from shorter pairs don't affect the result. The file has exactly
#    two rows — a header of pair names and one row of max values — in
#    the same column order as `combined`, and shares the main output's
#    timestamp so the two files are an obviously matched set. A failure
#    while writing this file never invalidates the main output, which
#    is always written first.
# =====================================================================


def find_all_csvs(root: Path, excluded_folder_paths: set[Path]) -> tuple[list[Path], int]:
    """Recursively find every .csv file under root, at any depth.

    Any file whose path falls inside one of excluded_folder_paths (or
    any of their subdirectories) is left out. Returns
    (included_files, excluded_count).
    """
    excluded_resolved = [p.resolve() for p in excluded_folder_paths]
    included: list[Path] = []
    excluded_count = 0
    for p in sorted(root.rglob("*.csv")):
        if not p.is_file():
            continue
        resolved = p.resolve()
        if any(
            resolved == ex or ex in resolved.parents for ex in excluded_resolved
        ):
            excluded_count += 1
            continue
        included.append(p)
    return included, excluded_count


def load_static_lookup(path: Path) -> tuple[dict[str, tuple[float, float]], set[str]]:
    """Read the static lookup CSV into {label: (val2, val3)}.

    Returns the lookup dict plus the set of labels that appeared more
    than once (ambiguous — any pair matching one of these is skipped).
    """
    df = pd.read_csv(path)
    labels = df.iloc[:, STATIC_LABEL_COL_INDEX].astype(str)
    val2 = pd.to_numeric(df.iloc[:, STATIC_VAL2_COL_INDEX], errors="coerce")
    val3 = pd.to_numeric(df.iloc[:, STATIC_VAL3_COL_INDEX], errors="coerce")

    counts: dict[str, int] = {}
    lookup: dict[str, tuple[float, float]] = {}
    for label, v2, v3 in zip(labels, val2, val3):
        counts[label] = counts.get(label, 0) + 1
        lookup[label] = (v2, v3)

    duplicate_labels = {label for label, count in counts.items() if count > 1}
    return lookup, duplicate_labels


def discover_pairs(
    csv_files: list[Path], static_path: Path
) -> tuple[dict[str, tuple[Path, Path]], list[str]]:
    """Group discovered CSVs into <base>/<base>_ex pairs.

    Returns (pairs, skip_log) where pairs maps base_name -> (base_path,
    ex_path) for every valid pair, and skip_log lists every file/base
    name that was excluded and why.
    """
    static_resolved = static_path.resolve()
    base_files: dict[str, list[Path]] = {}
    ex_files: dict[str, list[Path]] = {}
    skip_log: list[str] = []

    for f in csv_files:
        if f.resolve() == static_resolved:
            continue  # ignore the static lookup file itself
        stem = f.stem
        if stem.endswith(PAIR_SUFFIX):
            base_name = stem[: -len(PAIR_SUFFIX)]
            ex_files.setdefault(base_name, []).append(f)
        else:
            base_files.setdefault(stem, []).append(f)

    pairs: dict[str, tuple[Path, Path]] = {}
    for name in sorted(set(base_files) | set(ex_files)):
        has_base = name in base_files
        has_ex = name in ex_files

        if has_base and not has_ex:
            skip_log.append(
                f"SKIP base '{name}': no matching '{name}{PAIR_SUFFIX}.csv' "
                f"partner found (base file: {base_files[name][0]})"
            )
            continue

        if has_ex and not has_base:
            for p in ex_files[name]:
                skip_log.append(
                    f"SKIP '{name}{PAIR_SUFFIX}': orphaned _ex file with no "
                    f"base '{name}.csv' partner ({p})"
                )
            continue

        if len(base_files[name]) > 1 or len(ex_files[name]) > 1:
            all_paths = base_files[name] + ex_files[name]
            skip_log.append(
                f"CONFLICT '{name}': base name appears in multiple "
                f"locations, skipping pair: {[str(p) for p in all_paths]}"
            )
            continue

        pairs[name] = (base_files[name][0], ex_files[name][0])

    return pairs, skip_log


def read_source_column(
    path: Path, col_index: int, extra_header_rows: int = 0
) -> pd.Series | None:
    """Read one column (by position) from a data CSV as numeric values.

    extra_header_rows skips that many additional rows immediately after
    the normal header row before data is read (used for _ex files,
    which have one extra header-like row — see EX_EXTRA_HEADER_ROWS).

    Blank cells, non-numeric text, and existing NaNs all become NaN via
    coercion — the caller checks for these to decide whether to skip
    the pair. Returns None if col_index is out of range for this file.
    """
    skiprows = range(1, 1 + extra_header_rows) if extra_header_rows else None
    df = pd.read_csv(path, skiprows=skiprows)
    if col_index >= df.shape[1]:
        return None
    return pd.to_numeric(df.iloc[:, col_index], errors="coerce")


def excel_column_letter(col_index: int) -> str:
    """Convert a 0-based column index to an Excel-style letter (0->A, 25->Z, 26->AA)."""
    n = col_index + 1
    letters = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def describe_bad_cells(
    col: pd.Series,
    col_index: int,
    header_rows_before_data: int,
    source_label: str,
    path: Path,
) -> list[str]:
    """List 'cell REF (row N) in <source_label> file (path)' for every NaN in col.

    header_rows_before_data is how many header lines precede the first
    data row in the actual CSV file (used to translate a 0-based data
    row position back into a real 1-based spreadsheet row number).
    """
    col_letter = excel_column_letter(col_index)
    messages = []
    for pos in col.index[col.isna()]:
        row_number = pos + header_rows_before_data + 1
        messages.append(
            f"cell {col_letter}{row_number} (row {row_number}) in {source_label} "
            f"file ({path})"
        )
    return messages


def process_pair(
    name: str,
    base_path: Path,
    ex_path: Path,
    lookup: dict[str, tuple[float, float]],
    duplicate_labels: set[str],
    log: list[str],
) -> tuple[pd.Series, pd.Series] | None:
    """Process one valid pair. Returns (base_result, ex_result) or None if skipped."""

    if name in duplicate_labels:
        log.append(
            f"SKIP '{name}': label appears more than once in the static "
            f"lookup file — ambiguous, refusing to guess"
        )
        return None

    if name not in lookup:
        log.append(f"SKIP '{name}': label not found in static lookup file")
        return None

    val2, val3 = lookup[name]
    if pd.isna(val2) or pd.isna(val3):
        log.append(
            f"SKIP '{name}': val2/val3 in static lookup file is missing or "
            f"non-numeric"
        )
        return None

    denominator = val2 * val3
    if denominator == 0:
        log.append(
            f"SKIP '{name}': val2 * val3 is zero (val2={val2}, val3={val3}) "
            f"— cannot divide"
        )
        return None

    base_col = read_source_column(base_path, SOURCE_COL_INDEX_BASE)
    ex_col = read_source_column(
        ex_path, SOURCE_COL_INDEX_EX, extra_header_rows=EX_EXTRA_HEADER_ROWS
    )

    if base_col is None:
        log.append(
            f"SKIP '{name}': base file has fewer than "
            f"{SOURCE_COL_INDEX_BASE + 1} columns ({base_path})"
        )
        return None
    if ex_col is None:
        log.append(
            f"SKIP '{name}{PAIR_SUFFIX}': _ex file has fewer than "
            f"{SOURCE_COL_INDEX_EX + 1} columns ({ex_path})"
        )
        return None

    if len(base_col) != len(ex_col):
        log.append(
            f"SKIP '{name}': row count mismatch after accounting for the "
            f"_ex file's extra header row — base has {len(base_col)} data "
            f"rows, {name}{PAIR_SUFFIX} has {len(ex_col)} data rows"
        )
        return None

    bad_cells = describe_bad_cells(
        base_col, SOURCE_COL_INDEX_BASE, header_rows_before_data=1, source_label="base", path=base_path
    ) + describe_bad_cells(
        ex_col,
        SOURCE_COL_INDEX_EX,
        header_rows_before_data=1 + EX_EXTRA_HEADER_ROWS,
        source_label=PAIR_SUFFIX,
        path=ex_path,
    )
    if bad_cells:
        log.append(
            f"SKIP '{name}': blank/non-numeric/NaN value(s) found — "
            + "; ".join(bad_cells)
        )
        return None

    # Filter on the RAW base value, before the (value * 1000) / denominator
    # math below — flip to filter on the post-math value if that turns out
    # to be the intent instead.
    if BASE_MIN_VALUE is not None:
        keep = (base_col >= BASE_MIN_VALUE).to_numpy()
        dropped = (~keep).sum()
        base_col = base_col[keep]
        ex_col = ex_col[keep]
        if dropped:
            log.append(
                f"  '{name}': dropped {dropped} row(s) with base value < "
                f"{BASE_MIN_VALUE}"
            )
        if base_col.empty:
            log.append(
                f"SKIP '{name}': no rows remain after BASE_MIN_VALUE filter"
            )
            return None

    base_result = ((base_col * 1000) / denominator).reset_index(drop=True)
    ex_result = ex_col.reset_index(drop=True)  # no math applied to _ex
    return base_result, ex_result


def build_output_path(output_dir: Path, basename: str, timestamp: str) -> Path:
    return output_dir / f"{basename}_{timestamp}.csv"


def write_max_stress_output(combined: pd.DataFrame, timestamp: str) -> Path | None:
    """Write the peak-stress-per-pair summary CSV alongside the main output.

    Selects only combined's stress columns (row-2 label == ROW2_LABEL_BASE,
    and — belt and braces — row-1 name not ending in PAIR_SUFFIX), then
    writes a two-row CSV: pair names as the header, NaN-skipping max per
    column as the single data row. Returns the output path, or None if
    there were no stress columns to summarize (main output is unaffected
    either way — this runs after the main output is already written).
    """
    try:
        stress_columns = [
            col
            for col in combined.columns
            if col[1] == ROW2_LABEL_BASE and not col[0].endswith(PAIR_SUFFIX)
        ]
        if not stress_columns:
            print("\nNo stress columns found — skipping max-stress output file.")
            return None

        stress_df = combined.loc[:, stress_columns]
        max_values = stress_df.max(axis=0, skipna=True)

        max_output_path = build_output_path(OUTPUT_DIR, MAX_OUTPUT_BASENAME, timestamp)
        if max_output_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing output file: {max_output_path}"
            )

        pair_names = [col[0] for col in stress_columns]
        max_row = pd.DataFrame([max_values.to_numpy()], columns=pair_names)
        max_row.to_csv(max_output_path, index=False)
        return max_output_path
    except Exception as exc:
        print(f"\nWARNING: failed to write max-stress output file: {exc}")
        return None


def run() -> None:
    print(f"Scanning '{ROOT_DIR}' for CSV files...")
    csv_files, excluded_count = find_all_csvs(ROOT_DIR, EXCLUDED_FOLDER_PATHS)
    print(f"  found {len(csv_files)} CSV file(s) total")
    if excluded_count:
        print(
            f"  ({excluded_count} file(s) excluded via EXCLUDED_FOLDER_PATHS "
            f"{sorted(str(p) for p in EXCLUDED_FOLDER_PATHS)})"
        )

    print(f"Loading static lookup file '{STATIC_LOOKUP_PATH}'...")
    lookup, duplicate_labels = load_static_lookup(STATIC_LOOKUP_PATH)
    print(f"  {len(lookup)} unique label(s) loaded")
    if duplicate_labels:
        print(
            f"  WARNING: {len(duplicate_labels)} label(s) appear more than "
            f"once in the lookup file and will be treated as ambiguous: "
            f"{sorted(duplicate_labels)}"
        )

    pairs, skip_log = discover_pairs(csv_files, STATIC_LOOKUP_PATH)
    print(f"\nFound {len(pairs)} valid pair(s) before processing.")

    results: dict[str, pd.Series] = {}
    processed = []
    for name, (base_path, ex_path) in sorted(pairs.items()):
        outcome = process_pair(name, base_path, ex_path, lookup, duplicate_labels, skip_log)
        if outcome is None:
            continue
        base_result, ex_result = outcome
        results[f"{name}{PAIR_SUFFIX}"] = ex_result
        results[name] = base_result
        processed.append(name)

    print("\n--- Skipped / logged items ---")
    if skip_log:
        for line in skip_log:
            print(f"  {line}")
    else:
        print("  (none)")

    if not results:
        print("\nNo pairs were successfully processed — no output file written.")
        return

    combined = pd.concat(results, axis=1)
    combined.columns = pd.MultiIndex.from_tuples(
        [
            (col, ROW2_LABEL_EX if col.endswith(PAIR_SUFFIX) else ROW2_LABEL_BASE)
            for col in combined.columns
        ]
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(OUTPUT_DIR, OUTPUT_BASENAME, timestamp)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}"
        )
    combined.to_csv(output_path, index=False)

    max_output_path = write_max_stress_output(combined, timestamp)

    print("\n--- Summary ---")
    print(f"  Pairs processed successfully: {len(processed)} -> {processed}")
    print(f"  Pairs/files skipped: {len(skip_log)}")
    print(f"  Rows written: {len(combined)}")
    print(f"  Output file: {output_path}")
    print(f"  Max-stress output file: {max_output_path if max_output_path else '(skipped)'}")


if __name__ == "__main__":
    run()
