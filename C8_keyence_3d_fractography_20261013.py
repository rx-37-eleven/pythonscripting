"""
Keyence 3D scan viewer + fractography roughness script.

Reads every height-map CSV (and TIFF) in a folder and:

  1. Draws interactive 3D surfaces (matplotlib) — all samples side by
     side in one window, plus one window per sample. Rotate with a
     left-drag, zoom with a right-drag, and use the toolbar's Pan / Zoom
     tools. Buttons switch between isometric / top / front / side
     views, a slider sets the vertical exaggeration, and (when a colour
     TIFF with a matching name exists) a button swaps the height
     colouring for the optical texture. With LINK_CAMERAS on, rotating
     one surface rotates all of them.
  2. Calculates surface roughness at FULL resolution (the 3D preview is
     downsampled, the numbers are not) and writes a summary CSV.
  3. Shows an analysis window per sample: 2D height map, height
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
Dependencies: numpy, pandas, pillow, matplotlib   (optional: scipy)
All of these ship with Spyder, so nothing needs installing.
In Spyder the script switches the console to the Qt graphics backend so
the figures open in their own rotatable windows (the inline Plots pane
can't rotate 3D). If that fails: Preferences > IPython console >
Graphics > Backend: Qt, restart the kernel, and run again.

This script is standalone — it does not import or depend on any other
script in this repository.
"""

from __future__ import annotations

import json
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector, Slider

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

# ---- Excluding background (crop) --------------------------------------
# Everything outside the crop is left out of ALL calculations, the 3D
# views and the texture. Crops are in pixels of the raw file (the CSV
# grid; a colour TIFF is cropped to the same region automatically).
#
# 1) Interactive (recommended): the script opens each file's height map
#    (and its colour TIFF when there is one). Drag a box around the
#    sample, then click "Accept crop" (or press Enter). The box is saved
#    to crop_regions.json in the output folder and reused next run, so
#    you only draw it once per file.
CROP_INTERACTIVE = True
CROP_REDO = False        # True = ignore saved crops and draw them again
CROP_PREVIEW_POINTS = 900   # size of the crop preview along its long side
# 2) Fixed numbers: (x0, x1, y0, y1) for every file, or per file with
#    FILE_OVERRIDES["name"]["roi"]. These win over saved/drawn crops.
DEFAULT_ROI = None
# 3) After cropping, also trim any fully-empty margin rows/columns.
AUTO_TRIM_INVALID = True
# 4) Height masks, in Z_LABEL units: pixels below / above these heights
#    are treated as background (no measurement). Handy for a flat
#    background plane that surrounds the sample. None = off.
MASK_BELOW_Z = None
MASK_ABOVE_Z = None

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
# matplotlib redraws every polygon while you rotate, so the preview is
# block-averaged to at most this many points along the longer side
# (roughness is NOT affected). 200 stays smooth; raise it for more
# detail if rotating still feels fast enough.
MAX_DISPLAY_POINTS = 200
COLUMNS = 2             # surfaces per row in the side-by-side window
COLORMAP = "viridis"
COMMON_COLOR_SCALE = True   # same colour range on every surface
Z_EXAGGERATION = 1.0    # initial vertical exaggeration (slider changes it)
LINK_CAMERAS = True     # rotating one surface rotates all (side-by-side)
USE_TEXTURE = True      # build optical-texture view when a colour TIFF pairs up
SHOW_SIDE_BY_SIDE = True    # one window with every sample
SHOW_INDIVIDUAL = True      # one 3D window per sample
SHOW_ANALYSIS = True        # one analysis window per sample
SAVE_PNG = True             # also save the windows as PNG files

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
_CROP_PATH = None        # set in main(): <output folder>/crop_regions.json


def load_saved_crop(name: str, shape):
    """Return (found, roi) from crop_regions.json; roi None = full image."""
    try:
        data = json.loads(Path(_CROP_PATH).read_text())
        rec = data[name]
        if list(rec.get("shape", [])) != list(shape):
            return False, None
        return True, (tuple(rec["roi"]) if rec.get("roi") else None)
    except Exception:
        return False, None


def save_crop(name: str, roi, shape) -> None:
    try:
        data = json.loads(Path(_CROP_PATH).read_text())
    except Exception:
        data = {}
    data[name] = {"roi": list(roi) if roi else None, "shape": list(shape)}
    Path(_CROP_PATH).write_text(json.dumps(data, indent=2))


def _wait_for(state: dict, fig) -> None:
    """Block until the crop window is accepted or closed.

    Uses the canvas event loop directly. plt.pause() would re-raise the
    window ten times a second and steal focus (on macOS that makes the
    desktop bounce between the Spyder and figure screens)."""
    while not state["done"] and plt.fignum_exists(fig.number):
        fig.canvas.start_event_loop(0.2)


def select_crop(name: str, z: np.ndarray, texture):
    """Let the user drag a box on the height map / colour image.
    Returns ((x0, x1, y0, y1) or None for the full image, decided). `decided`
    is False when the window was closed without a choice (nothing is saved)."""
    H, W = z.shape
    f = max(1, math.ceil(max(H, W) / CROP_PREVIEW_POINTS))
    prev = block_mean(z, f)
    ph, pw = prev.shape
    lower = ROW0_AT == "bottom"
    ext = [0, pw * f, 0, ph * f] if lower else [0, pw * f, ph * f, 0]
    origin = "lower" if lower else "upper"
    ncols = 2 if texture is not None else 1
    fig, axs = plt.subplots(1, ncols, figsize=(7 * ncols, 7.5), sharex=True, sharey=True,
                            squeeze=False)
    axs = axs[0]
    fig.suptitle(f"{name}: drag a box around the sample, then Accept crop (or press Enter)",
                 fontsize=11)
    fig.subplots_adjust(bottom=0.14, top=0.92)
    lo, hi = (np.nanpercentile(prev, [2, 98]) if np.isfinite(prev).any() else (0, 1))
    axs[0].imshow(prev, origin=origin, extent=ext, cmap=COLORMAP, aspect="equal",
                  vmin=lo, vmax=hi)
    axs[0].set_title("Height map (blank = no measurement)")
    if texture is not None:
        timg = texture.resize((pw, ph), Image.BILINEAR)
        axs[1].imshow(np.asarray(timg), origin=origin, extent=ext, aspect="equal")
        axs[1].set_title("Colour image")
    for a in axs:
        a.set_xlabel("X (file pixels)")
    axs[0].set_ylabel("Y (file pixels)")

    state = {"roi": None, "done": False, "decided": False}
    sels = []

    def on_select(src):
        def cb(eclick, erelease):
            x0, x1 = sorted([eclick.xdata, erelease.xdata])
            y0, y1 = sorted([eclick.ydata, erelease.ydata])
            state["roi"] = (x0, x1, y0, y1)
            for sel in sels:
                if sel is not src[0]:
                    sel.extents = (x0, x1, y0, y1)
            fig.canvas.draw_idle()
        return cb
    for a in axs:
        holder = [None]
        sel = RectangleSelector(a, on_select(holder), useblit=True, interactive=True,
                                button=[1], minspanx=3, minspany=3, spancoords="pixels",
                                props=dict(facecolor="red", edgecolor="white", alpha=0.25))
        holder[0] = sel
        sels.append(sel)

    def accept(_e=None):
        if state["roi"] is None:        # nothing drawn yet: keep the window open
            fig.suptitle(f"{name}: draw a box first (or click 'Use full image')",
                         fontsize=11, color="red")
            fig.canvas.draw_idle()
            return
        state["decided"] = True
        state["done"] = True

    def full(_e=None):
        state["roi"] = None
        state["decided"] = True
        state["done"] = True
    b1 = Button(fig.add_axes([0.30, 0.02, 0.18, 0.06]), "Accept crop")
    b2 = Button(fig.add_axes([0.52, 0.02, 0.18, 0.06]), "Use full image")
    b1.on_clicked(accept)
    b2.on_clicked(full)
    fig.canvas.mpl_connect("key_press_event",
                           lambda e: accept() if e.key == "enter" else None)
    fig._keyence_widgets = (b1, b2, sels)
    plt.show(block=False)
    _wait_for(state, fig)
    if plt.fignum_exists(fig.number):
        plt.close(fig)
    r = state["roi"]
    if not state["decided"]:
        print(f"  crop  {name}: window closed without accepting - using the full image "
              "this run (not saved, you will be asked again)")
        return None, False
    if r is None:
        return None, True
    x0, x1, y0, y1 = int(math.floor(r[0])), int(math.ceil(r[1])), int(math.floor(r[2])), int(math.ceil(r[3]))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, True
    return (x0, x1, y0, y1), True


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

    z = entry.get("data") if is_tiff else read_height_csv(entry["path"])
    if z is None:
        return None
    apply_invalid(z, is_tiff)
    z *= np.float32(zstep)
    if MASK_BELOW_Z is not None:
        z[z < MASK_BELOW_Z] = np.nan
    if MASK_ABOVE_Z is not None:
        z[z > MASK_ABOVE_Z] = np.nan
    full_shape = z.shape

    # Crop: fixed numbers > saved/drawn crop > ask.
    roi = ov.get("roi", DEFAULT_ROI)
    if roi is None and (CROP_INTERACTIVE or _CROP_PATH):
        found, saved = (False, None) if CROP_REDO else load_saved_crop(name, full_shape)
        if found:
            roi = saved
            print(f"  crop  {name}: using saved crop {roi}" if roi else
                  f"  crop  {name}: saved choice is the FULL image")
            print(f"        (to draw it again: CROP_REDO = True, or delete {_CROP_PATH})")
        elif CROP_INTERACTIVE:
            roi, decided = select_crop(name, z, entry.get("texture"))
            if decided:
                save_crop(name, roi, full_shape)
                print(f"  crop  {name}: {roi if roi else 'full image'} (saved to {_CROP_PATH})")
    if roi:
        x0, x1, y0, y1 = roi
        z = np.ascontiguousarray(z[y0:y1, x0:x1], dtype=np.float32)
    else:
        x0, x1, y0, y1 = 0, full_shape[1], 0, full_shape[0]
    if AUTO_TRIM_INVALID and z.size and np.isfinite(z).any():
        ok = np.isfinite(z)
        rows, cols = np.where(ok.any(axis=1))[0], np.where(ok.any(axis=0))[0]
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        if (r1 - r0, c1 - c0) != z.shape:
            z = np.ascontiguousarray(z[r0:r1, c0:c1])
            x0, x1, y0, y1 = x0 + c0, x0 + c1, y0 + r0, y0 + r1
    roi = (x0, x1, y0, y1) if (x0, x1, y0, y1) != (0, full_shape[1], 0, full_shape[0]) else None
    off_x, off_y = x0, y0
    if z.size == 0 or not np.isfinite(z).any():
        print(f"  skip  {name}: no valid data (check crop / masks / INVALID_VALUES)")
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
            "Crop x0": x0, "Crop x1": x1, "Crop y0": y0, "Crop y1": y1,
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
# Plotting (matplotlib)
# ---------------------------------------------------------------------
VIEWS = {            # (elevation, azimuth) in degrees
    "Isometric": (25, -60),
    "Top": (90, -90),
    "Front": (0, -90),
    "Side": (0, 0),
}
RED, CYAN = "#E4002B", "#00C8FF"


def setup_backend() -> bool:
    """Switch to an interactive backend. Returns True when running inside
    IPython/Spyder (figures then stay open after the script ends)."""
    try:
        ip = get_ipython()  # noqa: F821  (defined inside IPython/Spyder)
    except NameError:
        ip = None
    if ip is not None:
        try:
            ip.run_line_magic("matplotlib", "qt")   # separate, rotatable windows
        except Exception as exc:
            print(f"  warn  could not switch Spyder to the Qt backend: {exc}\n"
                  "        In Spyder: Preferences > IPython console > Graphics > "
                  "Backend: Qt, then restart the kernel.")
        return True
    if matplotlib.get_backend().lower() == "agg":
        for name in ("QtAgg", "MacOSX", "TkAgg"):
            try:
                plt.switch_backend(name)
                break
            except Exception:
                continue
    return False


def apply_aspect(ax, it, exag: float):
    lx = float(it["xs"][-1] - it["xs"][0]) or 1.0
    ly = float(it["ys"][-1] - it["ys"][0]) or 1.0
    lz = float(np.nanmax(it["zd"])) or 1.0
    ax.set_box_aspect((lx, ly, lz * exag))


def draw_surface_axes(ax, it, vmin, vmax):
    """Draw one sample on a 3D axes; returns (height_surface, texture_surface)."""
    zd = it["zd"]
    X, Y = np.meshgrid(it["xs"], it["ys"])
    h, w = zd.shape
    kw = dict(rcount=h, ccount=w, linewidth=0, antialiased=False, shade=False)
    surf = ax.plot_surface(X, Y, zd, cmap=COLORMAP, vmin=vmin, vmax=vmax, **kw)
    tex_surf = None
    if it["tex"] is not None:
        tex_surf = ax.plot_surface(X, Y, np.nan_to_num(zd), facecolors=it["tex"] / 255.0, **kw)
        tex_surf.set_visible(False)
    mk = it["marks"]
    if SHOW_MEAN_REFERENCE:
        lvl = mk["mean_level"]
        ax.plot_surface(np.array([[it["xs"][0], it["xs"][-1]]] * 2),
                        np.array([[it["ys"][0]] * 2, [it["ys"][-1]] * 2]),
                        np.full((2, 2), lvl), color="grey", alpha=0.25, linewidth=0,
                        shade=False)
    if SHOW_EXTREMA:
        for key, label, colour in (("peak", "Highest", RED), ("pit", "Lowest", CYAN)):
            q = mk[key]
            ax.scatter([q["x"]], [q["y"]], [q["z"]], marker="D", s=70, c=colour,
                       edgecolors="white", depthshade=False, zorder=10)
            ax.text(q["x"], q["y"], q["z"], f"  {label} {q['value']:.4g}", fontsize=8,
                    zorder=11)
    ax.set_xlabel(f"X ({XY_LABEL})")
    ax.set_ylabel(f"Y ({XY_LABEL})")
    ax.set_zlabel(f"Height ({Z_LABEL})")
    ax.set_title(it["name"], fontsize=10)
    return surf, tex_surf


def build_3d_figure(items, title: str, link: bool):
    n = len(items)
    ncols = min(n, COLUMNS)
    nrows = math.ceil(n / ncols)
    fig = plt.figure(figsize=(6.2 * ncols, 5.4 * nrows + 0.8))
    if n > 1:
        fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.88, top=0.92, bottom=0.14, wspace=0.05, hspace=0.12)
    if COMMON_COLOR_SCALE:
        gmin = min(float(np.nanmin(it["zd"])) for it in items)
        gmax = max(float(np.nanmax(it["zd"])) for it in items)
    axes, surfs, texs = [], [], []
    for k, it in enumerate(items):
        ax = fig.add_subplot(nrows, ncols, k + 1, projection="3d")
        lo, hi = (gmin, gmax) if COMMON_COLOR_SCALE else (float(np.nanmin(it["zd"])), float(np.nanmax(it["zd"])))
        s, t = draw_surface_axes(ax, it, lo, hi)
        apply_aspect(ax, it, Z_EXAGGERATION)
        ax.view_init(*VIEWS["Isometric"])
        axes.append(ax)
        surfs.append(s)
        texs.append(t)
        if not COMMON_COLOR_SCALE:
            fig.colorbar(s, ax=ax, shrink=0.5, pad=0.10, label=f"Height ({Z_LABEL})")
    if COMMON_COLOR_SCALE:
        fig.colorbar(surfs[0], ax=axes, shrink=0.5, pad=0.10, label=f"Height ({Z_LABEL})")

    widgets = []
    state = {"exag": Z_EXAGGERATION}

    def set_view(elev, azim):
        for a in axes:
            a.view_init(elev, azim)
        fig.canvas.draw_idle()

    for i, (name, (el, az)) in enumerate(VIEWS.items()):
        bax = fig.add_axes([0.02 + i * 0.105, 0.015, 0.10, 0.04])
        b = Button(bax, name)
        b.on_clicked(lambda _e, el=el, az=az: set_view(el, az))
        widgets.append(b)
    sax = fig.add_axes([0.52, 0.025, 0.20, 0.025])
    slider = Slider(sax, "Z exag.", 0.5, 50, valinit=Z_EXAGGERATION, valstep=0.5)

    def on_exag(v):
        state["exag"] = v
        for a, it in zip(axes, items):
            apply_aspect(a, it, v)
        fig.canvas.draw_idle()
    slider.on_changed(on_exag)
    widgets.append(slider)
    if any(t is not None for t in texs):
        tax = fig.add_axes([0.80, 0.015, 0.17, 0.04])
        tb = Button(tax, "Texture / Height")
        mode = {"texture": False}

        def toggle(_e):
            mode["texture"] = not mode["texture"]
            for s, t in zip(surfs, texs):
                if t is not None:
                    s.set_visible(not mode["texture"])
                    t.set_visible(mode["texture"])
            fig.canvas.draw_idle()
        tb.on_clicked(toggle)
        widgets.append(tb)

    if link and n > 1:
        def sync(event):
            if event.inaxes in axes and event.button == 1:
                src = event.inaxes
                for a in axes:
                    if a is not src:
                        a.view_init(src.elev, src.azim, getattr(src, "roll", 0))
                fig.canvas.draw_idle()
        fig.canvas.mpl_connect("motion_notify_event", sync)
    fig._keyence_widgets = widgets      # keep widgets alive
    return fig


def build_analysis_figure(it):
    p, mk, s = it["plot"], it["marks"], it["summary"]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(f"{it['name']} — Sa {s['Sa']:.4g}, Sq {s['Sq']:.4g}, "
                 f"Sz {s['Sz (max-min)']:.4g} {Z_LABEL}; Sdr {s['Sdr (%)']:.3g}%", fontsize=12)
    fig.subplots_adjust(hspace=0.32, wspace=0.28)
    xs, ys = it["xs"], it["ys"]

    ax = axs[0, 0]
    im = ax.imshow(it["zd"], origin="lower", cmap=COLORMAP,
                   extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"Height ({Z_LABEL})")
    ax.set_title("Height map (levelled; zero = lowest point)")
    ax.set_xlabel(f"X ({XY_LABEL})")
    ax.set_ylabel(f"Y ({XY_LABEL})")
    if SHOW_MEAN_REFERENCE:
        try:
            ax.contour(xs, ys, it["zd"], levels=[mk["mean_level"]], colors="white", linewidths=0.8)
        except Exception:
            pass
    if SHOW_EXTREMA:
        for key, label, colour in (("peak", "Highest", RED), ("pit", "Lowest", CYAN)):
            q = mk[key]
            ax.scatter([q["x"]], [q["y"]], marker="D", s=60, c=colour, edgecolors="white", zorder=5)
            ax.annotate(f"{label} {q['value']:.4g}", (q["x"], q["y"]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8, color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, lw=0))
    yy, ra = p["ra_rows"]
    if MARK_ROUGH_LINES and "ra_hi_row" in mk:
        ax.axhline(yy[mk["ra_hi_row"]], color="white", ls=":", lw=1)
        ax.annotate(f"Roughest line (Ra {mk['ra_hi']:.3g})", (xs[0], yy[mk["ra_hi_row"]]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8, color="white")

    counts, edges = p["hist"]
    axs[0, 1].stairs(counts, edges, fill=True, color="#4C78A8")
    axs[0, 1].set_title("Height distribution")
    axs[0, 1].set_xlabel(f"Height about mean plane ({Z_LABEL})")
    axs[0, 1].set_ylabel("Pixels")

    mr, h = p["abbott"]
    axs[1, 0].plot(mr, h, color="#4C78A8")
    axs[1, 0].set_title("Abbott-Firestone curve")
    axs[1, 0].set_xlabel("Material ratio (%)")
    axs[1, 0].set_ylabel(f"Height about mean plane ({Z_LABEL})")
    axs[1, 0].grid(alpha=0.3)

    ax = axs[1, 1]
    ax.plot(yy, ra, color="#4C78A8", lw=0.8)
    ax.set_title("Ra per scan line (along X)")
    ax.set_xlabel(f"Y position of line ({XY_LABEL})")
    ax.set_ylabel(f"Ra ({Z_LABEL})")
    ax.grid(alpha=0.3)
    if MARK_ROUGH_LINES and "ra_hi_row" in mk:
        ax.axhline(mk["ra_mean"], color="grey", ls="--", lw=1)
        ax.annotate(f"mean Ra {mk['ra_mean']:.3g}", (yy.min(), mk["ra_mean"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=8, color="grey")
        for key, label, colour in (("ra_hi", "roughest", RED), ("ra_lo", "smoothest", CYAN)):
            r = mk[key + "_row"]
            ax.scatter([yy[r]], [mk[key]], marker="D", s=50, c=colour, edgecolors="black", zorder=5)
            ax.annotate(f"{label} {mk[key]:.3g}", (yy[r], mk[key]), xytext=(6, 6),
                        textcoords="offset points", fontsize=8)
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

    global _CROP_PATH
    _CROP_PATH = out_dir / "crop_regions.json"
    in_ipython = setup_backend()
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

    figs = []
    if SHOW_SIDE_BY_SIDE:
        figs.append((f"side_by_side_{stamp}",
                     build_3d_figure(results, f"Keyence 3D scans — {folder.name}", LINK_CAMERAS)))
    for r in results:
        if SHOW_INDIVIDUAL:
            figs.append((f"{r['name']}_3d_{stamp}", build_3d_figure([r], r["name"], False)))
        if SHOW_ANALYSIS:
            figs.append((f"{r['name']}_analysis_{stamp}", build_analysis_figure(r)))
    if SAVE_PNG:
        for stem, fig in figs:
            fig.savefig(out_dir / f"{stem}.png", dpi=150)
        print(f"PNG files written to {out_dir}")
    if figs:
        plt.show(block=not in_ipython)
    return summary


if __name__ == "__main__":
    summary = main()
