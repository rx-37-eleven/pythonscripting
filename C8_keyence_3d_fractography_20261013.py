"""
Keyence 3D scan viewer + fractography roughness script.

Reads every height-map CSV (and TIFF) in a folder and:

  1. Draws interactive 3D surfaces — all samples side by side in one
     page, plus one full-size page per sample. Rotate (left-drag), pan
     (right-drag, or the Pan tool in the toolbar), zoom (scroll wheel).
     Buttons switch between isometric / top / front / side views, change
     the vertical exaggeration, and (when a colour TIFF with a matching
     name exists) swap the height colouring for the optical texture.
     With LINK_CAMERAS on, rotating one surface rotates all of them.
  2. Calculates surface roughness at FULL resolution (the 3D preview is
     downsampled, the numbers are not) and writes a summary CSV.
  3. Writes an analysis page per sample: 2D height map, height
     histogram, Abbott-Firestone (bearing area) curve, and Ra per row.

Keyence VHX-7000 series (e.g. VHX-7100) notes
  The 3D height map is stored inside the microscope's JPEG/TIFF image
  and is exported to CSV from the VHX software ("Save CSV" on the 3D
  data). Published user notes say the CSV holds heights in microns and
  carries NO X/Y information, so the pixel pitch (which depends on the
  lens and magnification) must be entered below as XY_PITCH. Unmeasured
  pixels are written as -1.000 in your files.

Input files
  CSV   a grid of heights, one row per scan line; blank cells = no
        measurement. Leading header lines are skipped automatically.
        CSVs that are not a numeric grid (e.g. load/DIC logs) are
        skipped with a message.
  TIFF  single-channel 16-bit / 32-bit TIFFs are treated as height maps
        (when no CSV with the same name exists); 8-bit grey / RGB TIFFs
        are treated as optical texture and paired to the CSV whose name
        matches (e.g. VF6_2_3_3D.csv <-> VF6_2_3_3D.tif or VF6_2_3.tif).

Where the key values occur (SHOW_EXTREMA, SHOW_MEAN_REFERENCE,
MARK_ROUGH_LINES): diamonds at the highest / lowest points on the 3D
surfaces and height map, a mean-height plane in 3D and a mean-height
contour on the map, and the roughest / smoothest scan lines on the Ra
plot. The summary CSV also lists the highest, lowest and average height
of the file and the pixel (col, row) where the extremes sit.

Roughness reported (all after levelling, in Z_LABEL units)
  Areal (ISO 25178)  Sa, Sq, Sp, Sv, Sz (= max height - min height),
                     Ssk, Sku, Sz 1-99% (outlier-robust range),
                     Sdr (developed area ratio, %), area ratio
                     (developed / projected area; "Rs" in fractography)
  Profile (ISO 4287) Ra, Rq, Rp, Rv, Rt, Rz, profile length ratio (RL),
                     averaged over every scan line, measured along X and
                     along Y separately (crack-direction matters).
  Note: no waviness (lambda-c) filter is applied by default, so these
  are raw-surface values after plane/polynomial removal. Use
  GAUSS_SIGMA_PX (needs scipy) to smooth first, and ROI to analyse a
  region of interest.

Run this from Spyder: edit the CONFIG block below, then press Run.
Dependencies: numpy, pandas, pillow, plotly   (optional: scipy)
    pip install plotly pillow            (or: conda install plotly)
Plotly pages open in your web browser (Spyder's Plots pane can't show
interactive 3D), so no Qt/matplotlib backend setup is needed.

This script is standalone — it does not import or depend on any other
script in this repository.
"""

from __future__ import annotations

import math
import re
import sys
import warnings
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import plotly.graph_objects as go
from plotly.subplots import make_subplots

Image.MAX_IMAGE_PIXELS = None  # Keyence TIFFs can be very large

# =====================================================================
# CONFIG — edit these values, then press Run in Spyder.
# =====================================================================

# Folder containing the Keyence CSV / TIFF files.
INPUT_DIR = Path('/path/to/keyence_folder')

# Where results go (None = <INPUT_DIR>/keyence_3d_output).
OUTPUT_DIR = None

# ---- Calibration -----------------------------------------------------
# The CSV carries no pixel size, so the user enters the calibration here.
#   X_STEP  size of one pixel along X, in XY_LABEL units
#   Y_STEP  size of one pixel along Y (defaults to X_STEP = square
#           pixels; set it separately if your pixels are not square)
#   Z_STEP  multiplier that turns the numbers in the file into Z_LABEL
#           units (1.0 = the file heights are already in Z_LABEL)
# All three are 1.0 for now, so XY is in pixels and Z is as in the file.
# VHX-7000 series: CSV heights are reported to be in microns (hence the
# "um" Z default below; confirm against a known step height). The XY
# step depends on lens + magnification: take it from the scale shown in
# the VHX software for that image (or measure a known distance on
# screen and divide by its pixel count).
X_STEP = 1.0
Y_STEP = X_STEP         # change only if pixels are not square
Z_STEP = 1.0
Z_STEP_TIFF = Z_STEP    # same, for height-map TIFFs (often differs)
XY_LABEL = "px"         # e.g. "um"
Z_LABEL = "um"          # unit of the heights after Z_STEP

# Per-file overrides, keyed by file name without extension. Any of:
#   x_step, y_step, z_step, roi=(x0, x1, y0, y1) in pixels of the raw grid
# FILE_OVERRIDES = {"VF6_2_3_3D": {"x_step": 0.7, "y_step": 0.7,
#                                  "z_step": 1.0,
#                                  "roi": (1500, 4000, 2000, 9000)}}
FILE_OVERRIDES = {}

# ROI applied to every file unless overridden: (x0, x1, y0, y1) or None.
DEFAULT_ROI = None

# Values in the CSV/TIFF that mean "no measurement" (become NaN).
# Keyence CSV exports mark unmeasured pixels with -1.000; the int32
# limits are the marker in some other exports. If your real surface can
# legitimately contain exactly -1.0, remove it from this list.
INVALID_VALUES = [-1.0, -2147483648, 2147483647]
# Treat exactly-zero pixels in height TIFFs as invalid?
TIFF_ZERO_IS_INVALID = False

# Does CSV row 0 sit at the bottom of the image (like your existing
# viewer, origin lower-left) or at the top (usual image convention)?
# Only affects how the 3D view and texture are oriented, not roughness.
ROW0_AT = "bottom"      # "bottom" or "top"

# ---- Preparation -----------------------------------------------------
# Form removal before any roughness is calculated:
#   0 = subtract mean only, 1 = remove best-fit plane (tilt),
#   2 = remove best-fit quadratic surface (tilt + curvature)
LEVEL_ORDER = 1
# Gaussian smoothing sigma in pixels before analysis (0 = off; scipy).
GAUSS_SIGMA_PX = 0.0

# ---- Profile roughness (Ra, Rz ...) -----------------------------------
PROFILE_LINE = "mean"   # per-line reference: "mean" or "linear"
PROFILE_SEGMENTS = 5    # Rz = mean peak-to-valley over this many segments
# Scan lines with fewer valid points than this fraction are ignored.
MIN_LINE_FILL = 0.5

# ---- Marking where the key values occur --------------------------------
# Highest / lowest points: a red / cyan diamond on the 3D surfaces and
# on the analysis height map, labelled with the value.
SHOW_EXTREMA = True
# Which highest/lowest to mark: "levelled" (after plane removal, the
# surface you are looking at, matches Sp / Sv) or "file" (the raw
# numbers in the file, before any levelling).
MARK_EXTREMA_OF = "levelled"
# Average: a translucent plane at the mean height in 3D, and a white
# contour on the height map where the surface crosses the mean height.
SHOW_MEAN_REFERENCE = True
# Roughness is spread over the whole surface, so it is marked per scan
# line: the Ra-per-line plot shows the mean Ra and flags the roughest and
# smoothest lines, and the height map draws a line at the roughest one.
MARK_ROUGH_LINES = True

# ---- 3D display ------------------------------------------------------
# The browser can't draw 60 million points, so the preview is block-
# averaged to at most this many points along the longer side. Roughness
# is NOT affected.
MAX_DISPLAY_POINTS = 400
COLUMNS = 2             # surfaces per row in the side-by-side page
COLORSCALE = "Viridis"
COMMON_COLOR_SCALE = True   # same colour range on every surface
Z_EXAGGERATION = 1.0    # initial vertical exaggeration (buttons change it)
LINK_CAMERAS = True     # rotating one surface rotates all (side-by-side)
USE_TEXTURE = True      # build optical-texture view when a colour TIFF pairs up
PLOTLY_JS = True        # True = embed plotly.js (works offline, +4 MB/file)
                        # "cdn" = smaller files, needs internet
OPEN_IN_BROWSER = True  # open side-by-side + analysis pages when done
OPEN_INDIVIDUAL = False # also open every per-sample page

# =====================================================================
# End of CONFIG
# =====================================================================

CHUNK_ROWS = 256
TIFF_SUFFIXES = {".tif", ".tiff"}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False


def sniff_csv(path: Path, min_cols: int = 20):
    """Return the number of header lines to skip, or None if the file is
    not a numeric height grid."""
    with open(path, "r", errors="replace") as fh:
        lines = [next(fh, "") for _ in range(300)]
    numeric_rows, first_numeric = 0, None
    for i, line in enumerate(lines):
        toks = [t.strip() for t in line.rstrip("\r\n").split(",")]
        filled = [t for t in toks if t != ""]
        if len(toks) >= min_cols and all(_is_number(t) for t in filled):
            numeric_rows += 1
            if first_numeric is None:
                first_numeric = i
        elif first_numeric is not None and line.strip():
            return None  # text after the grid started: not a clean grid
    if numeric_rows < 10:
        return None
    return first_numeric


def read_height_csv(path: Path) -> np.ndarray | None:
    skip = sniff_csv(path)
    if skip is None:
        return None
    try:
        df = pd.read_csv(path, header=None, skiprows=skip, dtype=np.float32,
                         engine="c", skip_blank_lines=True)
    except (ValueError, TypeError):
        df = pd.read_csv(path, header=None, skiprows=skip, dtype=str, engine="c")
        df = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    arr = df.to_numpy(dtype=np.float32)
    del df
    # Drop one trailing all-empty column (trailing comma); leading /
    # edge blanks are kept so the grid stays aligned with the TIFFs.
    if arr.shape[1] > 1 and not np.isfinite(arr[:, -1]).any():
        arr = arr[:, :-1]
    return arr


def read_tiff(path: Path):
    """Return ("height", float32 array) or ("texture", PIL RGB image)."""
    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:  # unusual TIFF flavours
        try:
            import tifffile
            arr = tifffile.imread(str(path))
            if arr.ndim == 2:
                return "height", arr.astype(np.float32)
            return "texture", Image.fromarray(arr[..., :3].astype(np.uint8))
        except Exception:
            raise RuntimeError(f"cannot read {path.name}: {exc}") from exc
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        return "height", np.array(img, dtype=np.float32)
    return "texture", img.convert("RGB")


def apply_invalid(z: np.ndarray, is_tiff: bool) -> None:
    for v in INVALID_VALUES:
        z[z == np.float32(v)] = np.nan
    if is_tiff and TIFF_ZERO_IS_INVALID:
        z[z == 0] = np.nan
    z[~np.isfinite(z)] = np.nan


def discover(folder: Path):
    """Return a list of dicts: name, kind ("csv"/"tiff"), path, texture."""
    entries, textures = [], []
    for p in sorted(folder.iterdir()):
        if p.name.startswith(".") or not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf == ".csv":
            entries.append({"name": p.stem, "kind": "csv", "path": p, "texture": None})
        elif suf in TIFF_SUFFIXES:
            try:
                kind, data = read_tiff(p)
            except RuntimeError as exc:
                print(f"  skip  {exc}")
                continue
            if kind == "height":
                entries.append({"name": p.stem, "kind": "tiff", "path": p,
                                "texture": None, "data": data})
            else:
                textures.append((p, data))
    csv_stems = {e["name"].lower() for e in entries if e["kind"] == "csv"}
    entries = [e for e in entries
               if not (e["kind"] == "tiff" and e["name"].lower() in csv_stems)]

    used = set()
    for p, img in textures:
        ts = p.stem.lower()
        match = None
        for e in entries:
            es = e["name"].lower()
            if e["texture"] is None and (ts == es or es.startswith(ts) or ts.startswith(es)):
                match = e
                break
        if match is None and len(entries) == 1 and len(textures) == 1:
            match = entries[0]
        if match is not None:
            match["texture"] = img
            match["texture_name"] = p.name
            used.add(p)
        else:
            print(f"  note  {p.name}: colour TIFF with no matching height file, ignored")
    return entries


# ---------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------
def poly_terms(u, v, order):
    terms = [np.ones_like(u * v)]
    if order >= 1:
        terms += [u + 0 * v, v + 0 * u]
    if order >= 2:
        terms += [u * u + 0 * v, u * v, v * v + 0 * u]
    return terms


def level_surface(z: np.ndarray, order: int) -> None:
    """Subtract a best-fit polynomial surface in place, then the mean."""
    H, W = z.shape
    step = max(1, int(math.sqrt(H * W / 200_000)))
    sub = z[::step, ::step]
    jj, ii = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    u = 2.0 * (ii * step) / max(W - 1, 1) - 1.0
    v = 2.0 * (jj * step) / max(H - 1, 1) - 1.0
    ok = np.isfinite(sub)
    coef = None
    if order >= 1 and ok.sum() > 10:
        A = np.stack([t[ok] for t in poly_terms(u, v, order)], axis=1)
        coef, *_ = np.linalg.lstsq(A, sub[ok].astype(np.float64), rcond=None)
    uu = 2.0 * np.arange(W)[None, :] / max(W - 1, 1) - 1.0
    for r0 in range(0, H, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, H)
        if coef is not None:
            vv = 2.0 * np.arange(r0, r1)[:, None] / max(H - 1, 1) - 1.0
            plane = sum(c * t for c, t in zip(coef, poly_terms(uu, vv, order)))
            z[r0:r1] -= plane.astype(np.float32)
    total, count = 0.0, 0
    for r0 in range(0, H, CHUNK_ROWS):
        blk = z[r0:r0 + CHUNK_ROWS]
        m = np.isfinite(blk)
        total += float(blk[m].sum(dtype=np.float64))
        count += int(m.sum())
    if count:
        z -= np.float32(total / count)


def gaussian_smooth(z: np.ndarray, sigma: float) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("  warn  GAUSS_SIGMA_PX needs scipy (pip install scipy); smoothing skipped")
        return z
    m = np.isfinite(z)
    num = gaussian_filter(np.where(m, z, 0).astype(np.float32), sigma)
    den = gaussian_filter(m.astype(np.float32), sigma)
    return np.where(m & (den > 1e-3), num / np.maximum(den, 1e-3), np.nan).astype(np.float32)


def block_mean(z: np.ndarray, f: int) -> np.ndarray:
    """NaN-aware block average by integer factor f."""
    if f <= 1:
        return z.astype(np.float32)
    H, W = (z.shape[0] // f) * f, (z.shape[1] // f) * f
    out = np.empty((H // f, W // f), np.float32)
    step = 64 * f
    for r0 in range(0, H, step):
        r1 = min(r0 + step, H)
        blk = z[r0:r1, :W].reshape((r1 - r0) // f, f, W // f, f)
        m = np.isfinite(blk)
        s = np.where(m, blk, 0).sum(axis=(1, 3), dtype=np.float64)
        c = m.sum(axis=(1, 3))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[r0 // f:r1 // f] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


# ---------------------------------------------------------------------
# Roughness
# ---------------------------------------------------------------------
def developed_area(z: np.ndarray, dx: float, dy: float):
    """Triangulated surface area and projected area over fully valid cells."""
    dev = proj = 0.0
    H = z.shape[0]
    for r0 in range(0, H - 1, CHUNK_ROWS):
        a = z[r0:min(r0 + CHUNK_ROWS + 1, H)].astype(np.float64)
        z00, z10, z01, z11 = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]
        ok = np.isfinite(z00) & np.isfinite(z10) & np.isfinite(z01) & np.isfinite(z11)
        base = (dx * dy) ** 2
        t1 = 0.5 * np.sqrt((dy * (z10 - z00)) ** 2 + (dx * (z01 - z00)) ** 2 + base)
        t2 = 0.5 * np.sqrt((dy * (z01 - z11)) ** 2 + (dx * (z10 - z11)) ** 2 + base)
        dev += float(np.where(ok, t1 + t2, 0.0).sum())
        proj += float(ok.sum()) * dx * dy
    return dev, proj


def areal_parameters(z: np.ndarray, dx: float, dy: float):
    vals = z[np.isfinite(z)]
    n = vals.size
    if n == 0:
        raise ValueError("no valid height values")
    mean = float(vals.mean(dtype=np.float64))
    d = vals - np.float32(mean)
    sq = math.sqrt(float((d * d).mean(dtype=np.float64)))
    sp, sv = float(d.max()), float(-d.min())
    p1, p99 = np.percentile(vals, [1, 99])
    out = {
        "Sa": float(np.abs(d).mean(dtype=np.float64)),
        "Sq": sq,
        "Sp": sp,
        "Sv": sv,
        "Sz (max-min)": sp + sv,
        "Sz 1-99%": float(p99 - p1),
        "Ssk": float((d * d * d).mean(dtype=np.float64)) / sq ** 3 if sq else float("nan"),
        "Sku": float((d * d * d * d).mean(dtype=np.float64)) / sq ** 4 if sq else float("nan"),
    }
    dev, proj = developed_area(z, dx, dy)
    out["Projected area"] = proj
    out["Developed area"] = dev
    out["Area ratio (dev/proj)"] = dev / proj if proj else float("nan")
    out["Sdr (%)"] = (dev / proj - 1.0) * 100.0 if proj else float("nan")
    # Plot data
    counts, edges = np.histogram(vals, bins=120)
    sub = vals if n <= 2_000_000 else vals[np.random.default_rng(0).choice(n, 2_000_000, replace=False)]
    mr = np.linspace(0, 100, 201)
    abbott = np.percentile(sub, 100 - mr) - mean
    plot = {"hist": (counts, edges - mean), "abbott": (mr, abbott)}
    return out, plot, n


def profile_parameters(z: np.ndarray, d: float, nseg: int, line: str, min_fill: float):
    """Average ISO 4287 profile parameters over every line of z (along axis 1)."""
    H, W = z.shape
    x = np.arange(W, dtype=np.float64)
    L = W // nseg
    keys = ["Ra", "Rq", "Rp", "Rv", "Rt", "Rz"]
    rows = {k: np.full(H, np.nan) for k in keys}
    length = proj = 0.0
    for r0 in range(0, H, CHUNK_ROWS):
        r1 = min(r0 + CHUNK_ROWS, H)
        a = np.asarray(z[r0:r1], dtype=np.float64)
        m = np.isfinite(a)
        nn = m.sum(axis=1)
        good = nn >= max(10, min_fill * W)
        if not good.any():
            continue
        a0 = np.where(m, a, 0.0)
        sy = a0.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            if line == "linear":
                sx = (m * x).sum(axis=1)
                sxx = (m * x * x).sum(axis=1)
                sxy = (a0 * x).sum(axis=1)
                slope = (nn * sxy - sx * sy) / (nn * sxx - sx ** 2)
                icpt = (sy - slope * sx) / nn
            else:
                slope = np.zeros(len(a))
                icpt = sy / nn
            res = a - (icpt[:, None] + slope[:, None] * x)
            r_ok = np.where(m, res, 0.0)
            ra = np.abs(r_ok).sum(axis=1) / nn
            rq = np.sqrt((r_ok ** 2).sum(axis=1) / nn)
            rp = np.where(m, res, -np.inf).max(axis=1)
            rv = -np.where(m, res, np.inf).min(axis=1)
            if L >= 2:
                seg = res[:, :L * nseg].reshape(len(a), nseg, L)
                sm = np.isfinite(seg)
                mx = np.where(sm, seg, -np.inf).max(axis=2)
                mn = np.where(sm, seg, np.inf).min(axis=2)
                sok = sm.sum(axis=2) >= 2
                rz = np.where(sok, mx - mn, 0.0).sum(axis=1) / np.maximum(sok.sum(axis=1), 1)
            else:
                rz = rp + rv
        for k, v in zip(keys, (ra, rq, rp, rv, rp + rv, rz)):
            rows[k][r0:r1] = np.where(good, v, np.nan)
        dz = np.diff(a, axis=1)
        pair = (m[:, 1:] & m[:, :-1]) & good[:, None]
        length += float(np.where(pair, np.sqrt(d * d + dz * dz), 0.0).sum())
        proj += float(pair.sum()) * d
    out = {k: float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
           for k, v in rows.items()}
    out["RL (length ratio)"] = length / proj if proj else float("nan")
    return out, rows["Ra"]


# ---------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------
def crop_roi(arr, roi):
    if not roi:
        return arr
    x0, x1, y0, y1 = roi
    return arr[y0:y1, x0:x1]


def extrema(z: np.ndarray):
    """(max, (row, col), min, (row, col), mean) of the finite values."""
    hi, hi_at, lo, lo_at = -np.inf, None, np.inf, None
    total, count = 0.0, 0
    for r0 in range(0, z.shape[0], CHUNK_ROWS):
        blk = z[r0:r0 + CHUNK_ROWS]
        m = np.isfinite(blk)
        if not m.any():
            continue
        total += float(blk[m].sum(dtype=np.float64))
        count += int(m.sum())
        i = int(np.argmax(np.where(m, blk, -np.inf)))
        r, c = divmod(i, blk.shape[1])
        if blk[r, c] > hi:
            hi, hi_at = float(blk[r, c]), (r0 + r, c)
        i = int(np.argmin(np.where(m, blk, np.inf)))
        r, c = divmod(i, blk.shape[1])
        if blk[r, c] < lo:
            lo, lo_at = float(blk[r, c]), (r0 + r, c)
    return hi, hi_at, lo, lo_at, total / count


def process(entry: dict) -> dict:
    name = entry["name"]
    ov = FILE_OVERRIDES.get(name, {})
    is_tiff = entry["kind"] == "tiff"
    dx = float(ov.get("x_step", X_STEP))
    dy = float(ov.get("y_step", Y_STEP if "x_step" not in ov else dx))
    zstep = float(ov.get("z_step", Z_STEP_TIFF if is_tiff else Z_STEP))
    roi = ov.get("roi", DEFAULT_ROI)
    off_x, off_y = (roi[0], roi[2]) if roi else (0, 0)

    z = entry.get("data") if is_tiff else read_height_csv(entry["path"])
    if z is None:
        return None
    apply_invalid(z, is_tiff)
    full_shape = z.shape
    z = np.ascontiguousarray(crop_roi(z, roi), dtype=np.float32)
    z *= np.float32(zstep)
    if z.size == 0 or not np.isfinite(z).any():
        print(f"  skip  {name}: no valid data (check ROI / INVALID_VALUES)")
        return None
    valid_pct = 100.0 * float(np.isfinite(z).mean())
    # Highest / lowest / average of the heights as they are in the file.
    f_hi, f_hi_at, f_lo, f_lo_at, f_mean = extrema(z)
    if GAUSS_SIGMA_PX > 0:
        z = gaussian_smooth(z, GAUSS_SIGMA_PX)
    level_surface(z, LEVEL_ORDER)
    l_hi, l_hi_at, l_lo, l_lo_at, _ = extrema(z)

    H, W = z.shape
    stats, plot, nvalid = areal_parameters(z, dx, dy)
    sums = {"File": name, "Rows": H, "Cols": W, "Valid (%)": round(valid_pct, 2),
            "Highest height": f_hi, "Lowest height": f_lo, "Average height": f_mean,
            "Highest at col": f_hi_at[1] + off_x, "Highest at row": f_hi_at[0] + off_y,
            "Lowest at col": f_lo_at[1] + off_x, "Lowest at row": f_lo_at[0] + off_y,
            "Peak (levelled) at col": l_hi_at[1] + off_x, "Peak (levelled) at row": l_hi_at[0] + off_y,
            "Pit (levelled) at col": l_lo_at[1] + off_x, "Pit (levelled) at row": l_lo_at[0] + off_y}
    sums.update(stats)
    px, ra_rows = profile_parameters(z, dx, PROFILE_SEGMENTS, PROFILE_LINE, MIN_LINE_FILL)
    py, _ = profile_parameters(z.T, dy, PROFILE_SEGMENTS, PROFILE_LINE, MIN_LINE_FILL)
    sums.update({f"{k} X": v for k, v in px.items()})
    sums.update({f"{k} Y": v for k, v in py.items()})

    # Display grid
    f = max(1, math.ceil(max(H, W) / MAX_DISPLAY_POINTS))
    zd = block_mean(z, f)
    shift = float(np.nanmin(zd))
    zd -= np.float32(shift)
    Hc = zd.shape[0] * f

    def y_of_row(r):
        r = min(r, Hc - 1)
        return (Hc - 1 - r) * dy if ROW0_AT == "top" else r * dy

    tex = None
    if USE_TEXTURE and entry.get("texture") is not None:
        img = entry["texture"]
        if roi and full_shape[0] and full_shape[1]:
            x0, x1, y0, y1 = roi
            sx, sy = img.width / full_shape[1], img.height / full_shape[0]
            img = img.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
        img = img.resize((zd.shape[1], zd.shape[0]), Image.BILINEAR)
        tex = np.asarray(img, dtype=np.uint8)
    # The texture TIFF shares the CSV's row order, so it is flipped
    # together with the height grid (grid row 0 is drawn at low Y).
    if ROW0_AT == "top":
        zd = zd[::-1]
        if tex is not None:
            tex = tex[::-1]
    xs = (np.arange(zd.shape[1]) * f + (f - 1) / 2.0) * dx
    ys = (np.arange(zd.shape[0]) * f + (f - 1) / 2.0) * dy

    # Where the key values occur, in plot coordinates. Heights of the
    # markers come from the full-resolution surface (the preview is
    # block-averaged, so a sharp peak can sit slightly above it).
    if MARK_EXTREMA_OF == "file":
        hi_at, lo_at, hi_val, lo_val = f_hi_at, f_lo_at, f_hi, f_lo
    else:
        hi_at, lo_at, hi_val, lo_val = l_hi_at, l_lo_at, l_hi, l_lo

    def mark(at, value):
        r, c = at
        return {"x": c * dx, "y": y_of_row(r), "z": float(z[r, c]) - shift, "value": value}

    marks = {"peak": mark(hi_at, hi_val), "pit": mark(lo_at, lo_val), "mean_level": -shift}
    if np.isfinite(ra_rows).any():
        r_hi, r_lo = int(np.nanargmax(ra_rows)), int(np.nanargmin(ra_rows))
        marks.update({"ra_hi_row": r_hi, "ra_lo_row": r_lo,
                      "ra_hi": float(ra_rows[r_hi]), "ra_lo": float(ra_rows[r_lo]),
                      "ra_mean": px["Ra"]})
    plot["ra_rows"] = (np.array([y_of_row(r) for r in range(H)]), ra_rows)
    return {"name": name, "summary": sums, "plot": plot, "marks": marks,
            "zd": zd, "xs": xs, "ys": ys, "tex": tex,
            "tex_name": entry.get("texture_name")}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
LINK_JS = """
var gd = document.getElementById('{plot_id}');
var lastCam = '';
gd.on('plotly_relayout', function(ed) {
  var src = null, cam = null;
  Object.keys(ed).forEach(function(k) {
    if (/^scene\\d*\\.camera$/.test(k)) { src = k.split('.')[0]; cam = ed[k]; }
  });
  if (!src || !cam) return;
  var s = JSON.stringify(cam);
  if (s === lastCam) return;
  lastCam = s;
  var upd = {};
  Object.keys(gd._fullLayout).forEach(function(k) {
    if (/^scene\\d*$/.test(k) && k !== src) upd[k + '.camera'] = cam;
  });
  Plotly.relayout(gd, upd);
});
"""

VIEWS = {
    "Isometric": dict(eye=dict(x=1.35, y=-1.55, z=0.95), up=dict(x=0, y=0, z=1)),
    "Top": dict(eye=dict(x=0.0, y=0.0, z=2.4), up=dict(x=0, y=1, z=0)),
    "Front": dict(eye=dict(x=0.0, y=-2.4, z=0.05), up=dict(x=0, y=0, z=1)),
    "Side": dict(eye=dict(x=2.4, y=0.0, z=0.05), up=dict(x=0, y=0, z=1)),
}
EXAG_STEPS = [1, 2, 5, 10, 20, 50]


def grid_mesh(zd, xs, ys, tex):
    h, w = zd.shape
    X, Y = np.meshgrid(xs, ys)
    idx = np.arange(h * w).reshape(h, w)
    ok = np.isfinite(zd)
    quad = ok[:-1, :-1] & ok[:-1, 1:] & ok[1:, :-1] & ok[1:, 1:]
    a, b, c, d = idx[:-1, :-1][quad], idx[:-1, 1:][quad], idx[1:, :-1][quad], idx[1:, 1:][quad]
    i = np.concatenate([a, b])
    j = np.concatenate([b, d])
    k = np.concatenate([c, c])
    return dict(x=X.ravel(), y=Y.ravel(), z=np.nan_to_num(zd.ravel()),
                i=i, j=j, k=k, vertexcolor=tex.reshape(-1, 3))


def build_3d_figure(items, title: str, link: bool):
    n = len(items)
    ncols = min(n, COLUMNS)
    nrows = math.ceil(n / ncols)
    fig = make_subplots(
        rows=nrows, cols=ncols,
        specs=[[{"type": "scene"}] * ncols for _ in range(nrows)],
        subplot_titles=[it["name"] for it in items],
        horizontal_spacing=0.01, vertical_spacing=0.08 / nrows * 2)
    scenes = ["scene"] + [f"scene{i}" for i in range(2, n + 1)]

    if COMMON_COLOR_SCALE:
        cmin = min(float(np.nanmin(it["zd"])) for it in items)
        cmax = max(float(np.nanmax(it["zd"])) for it in items)
    surf_idx, mesh_idx, ratios = [], [], []
    for n_i, it in enumerate(items):
        r, c = divmod(n_i, ncols)
        zd = it["zd"]
        lo, hi = (cmin, cmax) if COMMON_COLOR_SCALE else (float(np.nanmin(zd)), float(np.nanmax(zd)))
        fig.add_trace(go.Surface(
            x=it["xs"], y=it["ys"], z=np.round(zd, 3), colorscale=COLORSCALE,
            cmin=lo, cmax=hi, showscale=(not COMMON_COLOR_SCALE) or n_i == 0,
            colorbar=dict(title=f"Height ({Z_LABEL})", len=0.5, thickness=12),
            lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.5),
            lightposition=dict(x=2000, y=-3000, z=4000),
            hovertemplate=f"x %{{x:.1f}} {XY_LABEL}<br>y %{{y:.1f}} {XY_LABEL}"
                          f"<br>z %{{z:.3f}} {Z_LABEL}<extra>{it['name']}</extra>"),
            row=r + 1, col=c + 1)
        surf_idx.append(len(fig.data) - 1)
        if it["tex"] is not None:
            m = grid_mesh(zd, it["xs"], it["ys"], it["tex"])
            fig.add_trace(go.Mesh3d(**m, flatshading=False, visible=False, name="texture",
                                    lighting=dict(ambient=0.9, diffuse=0.3, specular=0.05),
                                    hoverinfo="skip"), row=r + 1, col=c + 1)
            mesh_idx.append(len(fig.data) - 1)
        else:
            mesh_idx.append(None)
        mk = it["marks"]
        if SHOW_EXTREMA:
            for key, label, colour in (("peak", "Highest", "#E4002B"), ("pit", "Lowest", "#00C8FF")):
                q = mk[key]
                fig.add_trace(go.Scatter3d(
                    x=[q["x"]], y=[q["y"]], z=[q["z"]], mode="markers+text",
                    marker=dict(size=6, symbol="diamond", color=colour,
                                line=dict(color="white", width=1)),
                    text=[f"{label} {q['value']:.4g}"], textposition="top center",
                    showlegend=False, name=label,
                    hovertemplate=f"{label}: {q['value']:.4g} {Z_LABEL}<br>"
                                  f"x {q['x']:.1f} {XY_LABEL}, y {q['y']:.1f} {XY_LABEL}<extra></extra>"),
                    row=r + 1, col=c + 1)
        if SHOW_MEAN_REFERENCE:
            lvl = mk["mean_level"]
            fig.add_trace(go.Surface(
                x=[it["xs"][0], it["xs"][-1]], y=[it["ys"][0], it["ys"][-1]],
                z=[[lvl, lvl], [lvl, lvl]], opacity=0.25, showscale=False,
                colorscale=[[0, "#888888"], [1, "#888888"]], name="Mean height",
                hovertemplate="Mean height plane<extra></extra>"),
                row=r + 1, col=c + 1)
        lx = float(it["xs"][-1] - it["xs"][0]) or 1.0
        ly = float(it["ys"][-1] - it["ys"][0]) or 1.0
        lz = float(np.nanmax(zd)) or 1.0
        lmax = max(lx, ly)
        ratios.append((lx / lmax, ly / lmax, lz / lmax))

    for sc, (rx, ry, rz) in zip(scenes, ratios):
        fig.layout[sc].update(
            xaxis_title=f"X ({XY_LABEL})", yaxis_title=f"Y ({XY_LABEL})",
            zaxis_title=f"Height ({Z_LABEL})", aspectmode="manual",
            aspectratio=dict(x=rx, y=ry, z=rz * Z_EXAGGERATION),
            camera=VIEWS["Isometric"], dragmode="turntable")

    view_buttons = [dict(label=k, method="relayout",
                         args=[{f"{sc}.camera": v for sc in scenes}]) for k, v in VIEWS.items()]
    exag_buttons = [dict(label=f"{e}x", method="relayout",
                         args=[{f"{sc}.aspectratio": dict(x=rx, y=ry, z=rz * e)
                                for sc, (rx, ry, rz) in zip(scenes, ratios)}])
                    for e in EXAG_STEPS]
    menus = [
        dict(type="buttons", direction="right", buttons=view_buttons, x=0.0, y=1.13,
             xanchor="left", showactive=False),
        dict(type="buttons", direction="right", buttons=exag_buttons, x=0.42, y=1.13,
             xanchor="left", showactive=False),
    ]
    if any(i is not None for i in mesh_idx):
        n_tr = len(fig.data)
        vis_h = [True] * n_tr      # markers / mean plane stay visible
        vis_t = [True] * n_tr
        for s, m in zip(surf_idx, mesh_idx):
            if m is not None:
                vis_h[m] = False
                vis_t[s] = False
        menus.append(dict(type="buttons", direction="right", x=0.9, y=1.13, xanchor="left",
                          showactive=False,
                          buttons=[dict(label="Height", method="restyle", args=[{"visible": vis_h}]),
                                   dict(label="Texture", method="restyle", args=[{"visible": vis_t}])]))
    fig.update_layout(
        title=dict(text=title, x=0.01), updatemenus=menus,
        height=max(620, 640 * nrows), margin=dict(l=0, r=0, t=110, b=0),
        paper_bgcolor="white")
    fig.add_annotation(
        text="View / vertical exaggeration" + (" / colour" if len(menus) > 2 else "")
        + " — left-drag rotate, right-drag pan, scroll zoom"
        + ("<br>Axis = height above the lowest point; Highest/Lowest labels are "
           + ("relative to the mean plane" if MARK_EXTREMA_OF != "file" else "the values in the file")
           if SHOW_EXTREMA else ""),
        x=0.0, y=1.21, xref="paper", yref="paper", showarrow=False, xanchor="left",
        font=dict(size=11, color="#555"))
    post = LINK_JS if (link and n > 1) else None
    return fig, post


def build_analysis_figure(it):
    p = it["plot"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Height map (levelled, preview resolution)", "Height distribution",
        "Abbott-Firestone curve", "Ra per scan line (along X)"))
    fig.add_trace(go.Heatmap(x=it["xs"], y=it["ys"], z=it["zd"], colorscale=COLORSCALE,
                             colorbar=dict(title=Z_LABEL, len=0.4, y=0.8, thickness=10)), 1, 1)
    mk = it["marks"]
    if SHOW_MEAN_REFERENCE:
        fig.add_trace(go.Contour(
            x=it["xs"], y=it["ys"], z=it["zd"], showscale=False, hoverinfo="skip",
            contours=dict(start=mk["mean_level"], end=mk["mean_level"], size=1,
                          coloring="lines"),
            colorscale=[[0, "white"], [1, "white"]],
            line=dict(color="white", width=1.5), name="Mean height",
            showlegend=False), 1, 1)
    if SHOW_EXTREMA:
        for key, label, colour in (("peak", "Highest", "#E4002B"), ("pit", "Lowest", "#00C8FF")):
            q = mk[key]
            fig.add_trace(go.Scatter(
                x=[q["x"]], y=[q["y"]], mode="markers+text", showlegend=False,
                marker=dict(size=11, symbol="diamond", color=colour,
                            line=dict(color="white", width=1.5)),
                text=[f"{label} {q['value']:.4g}"], textposition="top center",
                hovertemplate=f"{label}: {q['value']:.4g} {Z_LABEL}<extra></extra>"), 1, 1)
    if MARK_ROUGH_LINES and "ra_hi_row" in mk:
        fig.add_hline(y=it["plot"]["ra_rows"][0][mk["ra_hi_row"]], row=1, col=1,
                      line=dict(color="white", width=1, dash="dot"),
                      annotation_text=f"Roughest line (Ra {mk['ra_hi']:.3g})",
                      annotation_font_color="white", annotation_position="top left")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    counts, edges = p["hist"]
    fig.add_trace(go.Bar(x=0.5 * (edges[:-1] + edges[1:]), y=counts, showlegend=False,
                         marker_color="#4C78A8"), 1, 2)
    mr, h = p["abbott"]
    fig.add_trace(go.Scatter(x=mr, y=h, mode="lines", showlegend=False,
                             line=dict(color="#4C78A8")), 2, 1)
    yy, ra = p["ra_rows"]
    fig.add_trace(go.Scatter(x=yy, y=ra, mode="lines", showlegend=False,
                             line=dict(color="#4C78A8", width=1)), 2, 2)
    if MARK_ROUGH_LINES and "ra_hi_row" in mk:
        fig.add_hline(y=mk["ra_mean"], row=2, col=2, line=dict(color="#888", dash="dash"),
                      annotation_text=f"mean Ra {mk['ra_mean']:.3g}", annotation_position="top left")
        for key, label, colour in (("ra_hi", "roughest", "#E4002B"), ("ra_lo", "smoothest", "#00C8FF")):
            row_i = mk[key + "_row"]
            fig.add_trace(go.Scatter(
                x=[yy[row_i]], y=[mk[key]], mode="markers+text", showlegend=False,
                marker=dict(size=9, symbol="diamond", color=colour, line=dict(color="black", width=1)),
                text=[f"{label} {mk[key]:.3g}"], textposition="top center"), 2, 2)
    fig.update_xaxes(title_text=f"X ({XY_LABEL})", row=1, col=1)
    fig.update_yaxes(title_text=f"Y ({XY_LABEL})", row=1, col=1)
    fig.update_xaxes(title_text=f"Height about mean plane ({Z_LABEL})", row=1, col=2)
    fig.update_yaxes(title_text="Pixels", row=1, col=2)
    fig.update_xaxes(title_text="Material ratio (%)", row=2, col=1)
    fig.update_yaxes(title_text=f"Height ({Z_LABEL})", row=2, col=1)
    fig.update_xaxes(title_text=f"Y position of line ({XY_LABEL})", row=2, col=2)
    fig.update_yaxes(title_text=f"Ra ({Z_LABEL})", row=2, col=2)
    s = it["summary"]
    fig.update_layout(
        height=820, template="plotly_white", margin=dict(t=90),
        title=dict(text=(f"{it['name']} — Sa {s['Sa']:.4g}, Sq {s['Sq']:.4g}, "
                         f"Sz {s['Sz (max-min)']:.4g} {Z_LABEL}; Sdr {s['Sdr (%)']:.3g}%"), x=0.01))
    return fig


def label_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        if c in ("Highest height", "Lowest height", "Average height"):
            ren[c] = f"{c} ({Z_LABEL})"
        elif c in ("Projected area", "Developed area"):
            ren[c] = f"{c} ({XY_LABEL}^2)"
        elif re.match(r"^(S[a-zq]|Sp|Sv|Sz|R[aqpvtz])\b", c) and not c.startswith(("Ssk", "Sku")):
            ren[c] = f"{c} ({Z_LABEL})"
    return df.rename(columns=ren)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    folder = Path(INPUT_DIR)
    if not folder.is_dir():
        sys.exit(f"INPUT_DIR does not exist: {folder}")
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else folder / "keyence_3d_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Scanning {folder} ...")
    entries = discover(folder)
    results = []
    for e in entries:
        print(f"  load  {e['path'].name}" + (f"  (+ texture {e['texture_name']})" if e["texture"] is not None else ""))
        try:
            r = process(e)
        except Exception as exc:
            print(f"  FAIL  {e['name']}: {exc}")
            continue
        if r is None:
            print(f"  skip  {e['path'].name}: not a numeric height grid")
            continue
        results.append(r)
    if not results:
        sys.exit("No usable height maps found.")

    summary = label_columns(pd.DataFrame([r["summary"] for r in results]))
    csv_path = out_dir / f"roughness_summary_{stamp}.csv"
    summary.to_csv(csv_path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print("\n" + summary.set_index("File").T.to_string())
    print(f"\nSummary written: {csv_path}")

    paths = []
    fig, post = build_3d_figure(results, f"Keyence 3D scans — {folder.name}", LINK_CAMERAS)
    side = out_dir / f"side_by_side_{stamp}.html"
    cfg = {"scrollZoom": True, "displaylogo": False,
           "toImageButtonOptions": {"format": "png", "scale": 2}}
    fig.write_html(side, include_plotlyjs=PLOTLY_JS, post_script=post, config=cfg)
    paths.append(side)
    individual = []
    for r in results:
        f1, _ = build_3d_figure([r], r["name"], False)
        p = out_dir / f"{r['name']}_3d_{stamp}.html"
        f1.write_html(p, include_plotlyjs=PLOTLY_JS, config=cfg)
        individual.append(p)

    # One analysis page: every sample's plots stacked.
    ana = out_dir / f"analysis_{stamp}.html"
    parts = []
    for n_i, r in enumerate(results):
        parts.append(build_analysis_figure(r).to_html(
            full_html=False, include_plotlyjs=PLOTLY_JS if n_i == 0 else False, config=cfg))
    ana.write_text("<html><head><meta charset='utf-8'><title>Roughness analysis</title></head>"
                   "<body style='font-family:sans-serif'>" + "<hr>".join(parts) + "</body></html>",
                   encoding="utf-8")
    print(f"3D pages  : {side}\n            {len(individual)} individual pages in {out_dir}")
    print(f"Analysis  : {ana}")

    if OPEN_IN_BROWSER:
        for p in [side, ana] + (individual if OPEN_INDIVIDUAL else []):
            webbrowser.open(p.resolve().as_uri())
    return summary


if __name__ == "__main__":
    summary = main()
