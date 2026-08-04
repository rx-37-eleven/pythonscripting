"""
Paired-CSV column extractor and multiplier.

Walks a directory tree for <base>.csv / <base>_ex.csv pairs, pulls one
column out of each file, multiplies each by a per-base scaling factor
looked up from a static CSV, and writes the results into one combined
output CSV.

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
ROOT_DIR = Path(r"/path/to/data/root")

# Path to the static lookup CSV (label, val2, val3).
STATIC_LOOKUP_PATH = Path(r"/path/to/lookup.csv")

# Directory + base filename for the output. A timestamp is appended on
# every run so a previous output is never silently overwritten.
OUTPUT_DIR = Path(r"/path/to/output")
OUTPUT_BASENAME = "multiplied_output"

# --- Column positions ---
# 0-based indexing: 0 is the FIRST column, 1 the second, 2 the third, etc.

# Column to pull out of each data file (<base>.csv / <base>_ex.csv).
# Example: SOURCE_COL_INDEX = 2 means the THIRD column.
SOURCE_COL_INDEX = 2

# Columns in the static lookup file:
#   STATIC_LABEL_COL_INDEX -> the label, matched against <base>
#   STATIC_VAL2_COL_INDEX  -> val2 (multiplier applied to the base file's column)
#   STATIC_VAL3_COL_INDEX  -> val3 (multiplier applied to the _ex file's column)
STATIC_LABEL_COL_INDEX = 0
STATIC_VAL2_COL_INDEX = 1
STATIC_VAL3_COL_INDEX = 2

# Suffix that marks the "_ex" half of a pair. Kept as a constant (rather
# than hardcoded inline) so other suffixes can be supported later.
PAIR_SUFFIX = "_ex"

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
#       has a header row.
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
#  11.  No rounding — multiplied values are written at full float
#       precision.
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
# =====================================================================


def find_all_csvs(root: Path) -> list[Path]:
    """Recursively find every .csv file under root, at any depth."""
    return sorted(p for p in root.rglob("*.csv") if p.is_file())


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


def read_source_column(path: Path, col_index: int) -> pd.Series | None:
    """Read one column (by position) from a data CSV as numeric values.

    Blank cells, non-numeric text, and existing NaNs all become NaN via
    coercion — the caller checks for these to decide whether to skip
    the pair. Returns None if col_index is out of range for this file.
    """
    df = pd.read_csv(path)
    if col_index >= df.shape[1]:
        return None
    return pd.to_numeric(df.iloc[:, col_index], errors="coerce")


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

    base_col = read_source_column(base_path, SOURCE_COL_INDEX)
    ex_col = read_source_column(ex_path, SOURCE_COL_INDEX)

    if base_col is None:
        log.append(
            f"SKIP '{name}': base file has fewer than {SOURCE_COL_INDEX + 1} "
            f"columns ({base_path})"
        )
        return None
    if ex_col is None:
        log.append(
            f"SKIP '{name}{PAIR_SUFFIX}': _ex file has fewer than "
            f"{SOURCE_COL_INDEX + 1} columns ({ex_path})"
        )
        return None

    if len(base_col) != len(ex_col):
        log.append(
            f"SKIP '{name}': row count mismatch — base has {len(base_col)} "
            f"rows, {name}{PAIR_SUFFIX} has {len(ex_col)} rows"
        )
        return None

    if base_col.isna().any() or ex_col.isna().any():
        log.append(
            f"SKIP '{name}': blank/non-numeric/NaN value found in the "
            f"source column of the base or _ex file"
        )
        return None

    base_result = (base_col * val2).reset_index(drop=True)
    ex_result = (ex_col * val3).reset_index(drop=True)
    return base_result, ex_result


def build_output_path(output_dir: Path, basename: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return output_dir / f"{basename}_{timestamp}.csv"


def run() -> None:
    print(f"Scanning '{ROOT_DIR}' for CSV files...")
    csv_files = find_all_csvs(ROOT_DIR)
    print(f"  found {len(csv_files)} CSV file(s) total")

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
        results[name] = base_result
        results[f"{name}{PAIR_SUFFIX}"] = ex_result
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(OUTPUT_DIR, OUTPUT_BASENAME)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output file: {output_path}"
        )
    combined.to_csv(output_path, index=False)

    print("\n--- Summary ---")
    print(f"  Pairs processed successfully: {len(processed)} -> {processed}")
    print(f"  Pairs/files skipped: {len(skip_log)}")
    print(f"  Rows written: {len(combined)}")
    print(f"  Output file: {output_path}")


if __name__ == "__main__":
    run()
