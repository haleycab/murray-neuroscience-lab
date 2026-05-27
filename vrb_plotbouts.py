from __future__ import annotations
from typing import Iterable, Tuple, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_sheets_for_abfs(sheets_abfs: dict, abf_names: list[str]) -> list[str]:
    """
    Return a unique list of sheets that contain any of the given abf_names.

    Parameters
    ----------
    sheets_abfs : dict
        Dictionary of sheets, each with 'abfs' and/or 'annotations'.
    abf_names : list[str]
        List of ABF trace names to search for.

    Returns
    -------
    list[str]
        Unique list of sheet names containing at least one abf_name.
    """
    matching_sheets = set()

    for sheet, content in sheets_abfs.items():
        # Collect ABF names from "abfs"
        try:
            traces_abfs = set(content["abfs"].keys())
        except (KeyError, AttributeError):
            traces_abfs = set()

        # Collect ABF names from "annotations"
        try:
            annotations = content["annotations"]
            traces_annotated = set(annotations["Trace name"].unique().tolist())
        except (KeyError, AttributeError):
            traces_annotated = set()

        # Check if any overlap exists
        all_traces = traces_abfs.union(traces_annotated)
        if any(name in all_traces for name in abf_names):
            matching_sheets.add(sheet)

    return sorted(matching_sheets)



# ---------------------------------------------------------------------
# 1) Light-weight helpers (pure / easy to unit-test)
# ---------------------------------------------------------------------

def is_kw(tag: str | None, kw: str) -> bool:
    """Case-insensitive substring match with None-guard."""
    return isinstance(tag, str) and (kw in tag.lower())

def label_from_tag(tag: str, *, start_kw: str, end_kw: str) -> str:
    """Map annotation tag -> one of {'start','end','vrb'}."""
    t = (tag or "").lower()
    if start_kw in t: return "start"
    if end_kw   in t: return "end"
    return "vrb"

def adaptive_last_cross(
    t0: float,
    t1: float,
    bin_centers: np.ndarray,
    variances: np.ndarray,
    between_std_val: float,
    sigma0: float,
    step: float,
    sigma_floor: float,
) -> Tuple[float, float, float]:
    """
    Find the last bin center in [t0, t1] where variance exceeds (sigma * between_std)^2,
    scanning sigma downward; returns (t_cross, sigma_used, var_threshold).
    If between_std <= 0, returns the last available bin center in the interval.
    On failure, returns (nan, nan, nan).
    """
    # input sanity
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.nan, np.nan, np.nan

    lo = max(t0, np.nanmin(bin_centers))
    hi = min(t1, np.nanmax(bin_centers))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan, np.nan, np.nan

    interval = (bin_centers >= lo) & (bin_centers <= hi) & np.isfinite(variances)
    if not np.any(interval):
        return np.nan, np.nan, np.nan

    # No between-std: return last available bin center
    if not (np.isfinite(between_std_val) and between_std_val > 0):
        idxs = np.where(interval)[0]
        return (float(bin_centers[idxs[-1]]), np.nan, np.nan) if idxs.size else (np.nan, np.nan, np.nan)

    # Try decreasing sigma thresholds
    sigma = float(sigma0)
    while sigma >= sigma_floor - 1e-12:
        thr = (sigma * between_std_val) ** 2
        idxs = np.where(interval & (variances > thr))[0]
        if idxs.size > 0:
            return float(bin_centers[idxs[-1]]), sigma, thr
        sigma -= step

    # Fallback: any last bin in interval
    idxs = np.where(interval)[0]
    return (float(bin_centers[idxs[-1]]), np.nan, np.nan) if idxs.size else (np.nan, np.nan, np.nan)

def pick_end_offset(
    labels: np.ndarray,
    ann_times: np.ndarray,
    ann_midpoints: np.ndarray,
    strategy: str,
    fallback: float,
) -> float:
    """Choose an average offset to use for 'end' annotations."""
    s = (strategy or "auto").lower()
    vrb_mask = (labels == "vrb")

    # distances from each vrb to its following midpoint
    vrb_to_mid = []
    if ann_midpoints.size:
        for k in range(len(ann_times) - 1):
            if vrb_mask[k]:
                vrb_to_mid.append(ann_midpoints[k] - ann_times[k])
    vrb_to_mid = np.asarray(vrb_to_mid, dtype=float)
    vrb_to_mid = vrb_to_mid[np.isfinite(vrb_to_mid) & (vrb_to_mid > 0)]

    inter_vrb = np.diff(ann_times[vrb_mask]) if np.sum(vrb_mask) >= 2 else np.array([], dtype=float)
    inter_vrb = inter_vrb[np.isfinite(inter_vrb) & (inter_vrb > 0)]
    half_inter_vrb = 0.5 * inter_vrb if inter_vrb.size else np.array([], dtype=float)

    inter_ann = np.diff(ann_times) if ann_times.size >= 2 else np.array([], dtype=float)
    inter_ann = inter_ann[np.isfinite(inter_ann) & (inter_ann > 0)]
    half_inter_ann = 0.5 * inter_ann if inter_ann.size else np.array([], dtype=float)

    if s == "vrb_to_mid":
        return float(np.median(vrb_to_mid)) if vrb_to_mid.size else fallback
    if s == "half_inter_vrb":
        return float(np.median(half_inter_vrb)) if half_inter_vrb.size else fallback
    if s == "half_inter_ann":
        return float(np.median(half_inter_ann)) if half_inter_ann.size else fallback

    # auto preference ordering
    if vrb_to_mid.size:   return float(np.median(vrb_to_mid))
    if half_inter_vrb.size: return float(np.median(half_inter_vrb))
    if half_inter_ann.size: return float(np.median(half_inter_ann))
    return fallback

# ---------------------------------------------------------------------
# 2) Data-prep helpers (one responsibility each)
# ---------------------------------------------------------------------

def between_bout_stats(
    abf,
    annotations: pd.DataFrame,
    *,
    channel: int = 1,
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,
    bins: int = 50,   # kept for API compatibility (not used here)
):
    """
    Compute mean and std of the ABF signal *between* swimming bouts, using ALL sweeps.

    Returns
    -------
    (mean_val, std_val) as floats
    """

    # ---- build full continuous trace across all sweeps ----
    full_t, full_y = [], []
    for s in range(abf.sweepCount):
        abf.setSweep(sweepNumber=s, channel=channel)
        t = abf.sweepX + s * abf.sweepLengthSec
        y = abf.sweepY
        full_t.append(t)
        full_y.append(y)

    if not full_t:
        raise ValueError("ABF has no sweeps.")

    x = np.concatenate(full_t)
    y = np.concatenate(full_y)

    # ---- get bout windows (absolute seconds, padded) ----
    windows = extract_bout_windows(
        annotations,
        seconds_col=seconds_col,
        tags_col=tags_col,
        start_kw=start_kw.lower(),
        end_kw=end_kw.lower(),
        pad_s=pad_s,
    )

    # ---- mask out all samples that fall inside any window ----
    mask = np.ones_like(x, dtype=bool)
    for (a, b) in windows:
        mask &= ~((x >= a) & (x <= b))

    between_vals = y[mask]

    if between_vals.size == 0:
        print("⚠️ No between-bout data available across all sweeps.")
        return np.nan, np.nan

    mean_val = float(np.mean(between_vals))
    std_val  = float(np.std(between_vals))
    return mean_val, std_val



def concat_one_channel(abf, ch=0):
    sr = float(abf.dataRate)  # Hz

    # Gather per-sweep lengths (in samples)
    sweep_lengths = []
    for s in range(abf.sweepCount):
        abf.setSweep(s, channel=ch)
        sweep_lengths.append(len(abf.sweepY))
    sweep_lengths = np.asarray(sweep_lengths, dtype=int)

    # Offsets in seconds based on actual lengths
    offsets_sec = np.cumsum(np.concatenate([[0], sweep_lengths[:-1]])) / sr

    full_time_chunks = []
    full_current_chunks = []

    for s, n in enumerate(sweep_lengths):
        abf.setSweep(s, channel=ch)
        # copy() is IMPORTANT so arrays don't mutate on the next setSweep()
        tx = abf.sweepX.copy()           # (n,) seconds, starts at 0
        y  = abf.sweepY.copy().astype(np.float32, copy=False)
        full_time_chunks.append(tx + offsets_sec[s])
        full_current_chunks.append(y)

    full_time    = np.concatenate(full_time_chunks)
    full_current = np.concatenate(full_current_chunks)
    return full_time, full_current


def extract_bout_windows(
    annotations: pd.DataFrame,
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,  # 10 ms padding
):
    """Return merged list of (start, end) bout windows in seconds, padded by pad_s."""
    df = annotations.copy()

    def _has_kw(x, kw):
        return isinstance(x, str) and kw in x.lower()

    start_kw = start_kw.lower()
    end_kw   = end_kw.lower()

    df["_is_start"] = df[tags_col].apply(lambda x: _has_kw(x, start_kw))
    df["_is_end"]   = df[tags_col].apply(lambda x: _has_kw(x, end_kw))
    df = df.sort_values(seconds_col, kind="mergesort")

    starts = df.loc[df["_is_start"], seconds_col].astype(float).to_list()
    ends   = df.loc[df["_is_end"],   seconds_col].astype(float).to_list()

    windows, si, ei = [], 0, 0
    while si < len(starts) and ei < len(ends):
        s = starts[si]
        while ei < len(ends) and ends[ei] < s:
            ei += 1
        if ei >= len(ends):
            break
        e = ends[ei]
        windows.append((max(0.0, s - pad_s), e + pad_s))
        si += 1
        ei += 1

    while si < len(starts):  # unmatched starts
        s = starts[si]
        windows.append((max(0.0, s - pad_s), s + pad_s))
        si += 1

    while ei < len(ends):  # unmatched ends
        e = ends[ei]
        windows.append((max(0.0, e - pad_s), e + pad_s))
        ei += 1

    if not windows and len(annotations) > 0 and seconds_col in annotations.columns:
        for t in annotations[seconds_col].astype(float).to_list():
            windows.append((max(0.0, t - pad_s), t + pad_s))

    # merge overlapping/touching
    if not windows:
        return []
    windows = sorted(windows)
    merged = [windows[0]]
    for a, b in windows[1:]:
        ca, cb = merged[-1]
        if a <= cb:
            merged[-1] = (ca, max(cb, b))
        else:
            merged.append((a, b))
    return merged

# ---------------------------------------------------------------------
# 3) Per-bout computation (variances, last-crossings, table rows)
# ---------------------------------------------------------------------

def per_bout_variances(
    x_plot: np.ndarray,
    y_bout: np.ndarray,
    *,
    between_mean: float,
    bin_width_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute fixed-width bins on x_plot and variance of |y - between_mean| inside each bin.
    Returns (bins, bin_centers, variances).
    """
    y_dev = np.abs(y_bout - between_mean)
    tmin, tmax = float(np.nanmin(x_plot)), float(np.nanmax(x_plot))
    bins = np.arange(tmin, tmax + bin_width_s * 0.5, bin_width_s, dtype=float)
    if bins.size < 2:
        bins = np.array([tmin, tmax], dtype=float)
    n_bins_eff = bins.size - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    variances = np.empty(n_bins_eff, dtype=float)
    for k in range(n_bins_eff):
        if k < n_bins_eff - 1:
            sel = (x_plot >= bins[k]) & (x_plot <  bins[k + 1])
        else:
            sel = (x_plot >= bins[k]) & (x_plot <= bins[k + 1])
        variances[k] = np.var(y_dev[sel]) if np.any(sel) else np.nan

    return bins, bin_centers, variances

def annotations_in_bout(
    annotations: pd.DataFrame,
    a: float, b: float,
    *,
    seconds_col: str,
    tags_col: str,
    start_kw: str,
    end_kw: str,
    relative_time: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Slice annotations to [a, b], build label array and sorted times (relative or absolute).
    Returns (ann_times_all, labels_all, ann_midpoints, ann_slice_sorted).
    """
    ann_in = annotations[
        (annotations[seconds_col].astype(float) >= a) &
        (annotations[seconds_col].astype(float) <= b)
    ].copy()

    ann_in["_is_start"] = ann_in[tags_col].apply(lambda s: is_kw(s, start_kw))
    ann_in["_is_end"]   = ann_in[tags_col].apply(lambda s: is_kw(s, end_kw))
    ann_in["_is_vrb"]   = ~(ann_in["_is_start"] | ann_in["_is_end"])

    ann_times_abs = ann_in[seconds_col].astype(float).to_numpy()
    labels_all = np.where(ann_in["_is_start"].to_numpy(), "start",
                   np.where(ann_in["_is_end"].to_numpy(), "end", "vrb"))

    ann_times_all = ann_times_abs - a if relative_time else ann_times_abs
    order = np.argsort(ann_times_all)
    ann_times_all = ann_times_all[order]
    labels_all = labels_all[order]

    ann_midpoints = 0.5 * (ann_times_all[:-1] + ann_times_all[1:]) if len(ann_times_all) >= 2 else np.array([], dtype=float)

    return ann_times_all, labels_all, ann_midpoints, ann_in.iloc[order].reset_index(drop=True)

def compute_last_crossings(
    ann_times_all: np.ndarray,
    labels_all: np.ndarray,
    ann_midpoints: np.ndarray,
    *,
    bin_centers: np.ndarray,
    variances: np.ndarray,
    between_std: float,
    sigma_multiplier: float,
    sigma_step: float,
    min_sigma: float,
    end_offset: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each annotation time, compute last-crossing stats.
    Returns arrays for (last_cross_times, sigma_used_arr, varthr_used_arr).
    """
    last_cross_times = np.full_like(ann_times_all, np.nan, dtype=float)
    sigma_used_arr   = np.full_like(ann_times_all, np.nan, dtype=float)
    varthr_used_arr  = np.full_like(ann_times_all, np.nan, dtype=float)

    for k, t_ann in enumerate(ann_times_all):
        label = labels_all[k]
        if label == "end":
            t0, t1 = t_ann, t_ann + end_offset
        else:
            if k >= len(ann_times_all) - 1:
                t0 = t1 = np.nan
            else:
                t0 = t_ann
                t1 = ann_midpoints[k]

        t_cross, sigma_used, thr_used = adaptive_last_cross(
            t0, t1, bin_centers, variances,
            between_std_val=between_std,
            sigma0=sigma_multiplier, step=sigma_step, sigma_floor=min_sigma
        )
        last_cross_times[k] = t_cross
        sigma_used_arr[k]   = sigma_used
        varthr_used_arr[k]  = thr_used

    return last_cross_times, sigma_used_arr, varthr_used_arr

def build_per_bout_table(
    i: int, a: float, b: float,
    ann_times_all: np.ndarray,
    last_cross_times: np.ndarray,
    black_dot_times: np.ndarray,
    labels_all: np.ndarray,
    *,
    relative_time: bool
) -> pd.DataFrame:
    """Create the per-bout rows; converts back to absolute times if needed."""
    if relative_time:
        abs_ann  = ann_times_all + a
        abs_last = np.where(np.isfinite(last_cross_times), last_cross_times + a, np.nan)
        abs_black= np.where(np.isfinite(black_dot_times), black_dot_times + a,  np.nan)
    else:
        abs_ann, abs_last, abs_black = ann_times_all.copy(), last_cross_times.copy(), black_dot_times.copy()

    return pd.DataFrame({
        "BoutIndex": i,
        "BoutStart": a,
        "BoutEnd": b,
        "AnnotationType": labels_all,
        "AnnotationTime": abs_ann,
        "LastCrossTime": abs_last,
        "BlackDotTime": abs_black,
    })

# ---------------------------------------------------------------------
# 4) Plotting (kept minimal; accepts precomputed arrays)
# ---------------------------------------------------------------------

def plot_one_bout_axis(
    ax,
    x_plot: np.ndarray,
    y_bout: np.ndarray,
    *,
    between_mean: float,
    dot_times: np.ndarray,
    dot_size: float,
    ylim: Optional[Tuple[float, float]],
    title: str,
    xlabel: str,
    ylabel: str,
):
    ax.plot(x_plot, y_bout, linewidth=0.9)
    finite_mask = np.isfinite(dot_times)
    if np.any(finite_mask):
        ax.scatter(
            dot_times[finite_mask],
            np.full(np.sum(finite_mask), between_mean),
            s=dot_size, color="black", zorder=5
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(x_plot.min(), x_plot.max())
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------
# 6) Public entry point
# ---------------------------------------------------------------------

def plot_bouts_like_lastcross2(
    abf,
    annotations: pd.DataFrame,
    *,
    channel: int = 1,
    sweep: int = 0,   # kept for API compatibility; not used when concatenating
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,
    bin_width_s: float = 0.010,
    relative_time: bool = False,
    sigma_multiplier: float = 1.0,
    sigma_step: float = 0.1,
    min_sigma: float = 0.0,
    shade_exceeding: bool = True,  # retained for API compatibility
    dot_size: float = 60.0,
    ylim = None,
    ncols: int = 2,
    max_bouts: int | None = None,
    title_prefix: str = "Bout",
    verbose: bool = False,
    end_offset_strategy: str = "auto",
    return_tables: bool = False,
    merge_tolerance_s: float = 0.0,  # 0 => exact-time merge
):
    """
    Same behavior as before, now refactored into small helpers.
    If return_tables=True, returns (windows, merged_table).
    """
    # Compute stats & signal
    between_mean, between_std = between_bout_stats(
        abf, annotations, channel=channel,
        seconds_col=seconds_col, tags_col=tags_col, pad_s=pad_s
    )
    x, y = concat_one_channel(abf, ch=channel)

    # Windows
    windows = extract_bout_windows(
        annotations, seconds_col=seconds_col, tags_col=tags_col,
        start_kw=start_kw, end_kw=end_kw, pad_s=pad_s
    )
    if not windows:
        if return_tables:
            return [], annotations.head(0).copy()
        return []

    if max_bouts is not None:
        windows = windows[:max_bouts]

    # Figure grid
    n = len(windows)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9*ncols, 3.2*nrows), squeeze=False)
    axes = axes.ravel()

    per_bout_tables: List[pd.DataFrame] = []

    # Main loop
    for i, (a, b) in enumerate(windows):
        ax = axes[i]

        in_bout = (x >= a) & (x <= b)
        if not np.any(in_bout):
            if verbose: print(f"[warn] bout {i} has no samples.")
            ax.set_title(f"{title_prefix} {i+1} [empty]")
            ax.axis("off")
            continue

        x_bout, y_bout = x[in_bout], y[in_bout]
        x_plot = (x_bout - a) if relative_time else x_bout

        # degenerate guard
        if x_plot.size == 0 or np.nanmin(x_plot) == np.nanmax(x_plot):
            if verbose: print(f"[warn] bout {i} degenerate time span.")
            ax.set_title(f"{title_prefix} {i+1} [degenerate]")
            ax.axis("off")
            continue

        # bins/variances
        bins, bin_centers, variances = per_bout_variances(
            x_plot, y_bout, between_mean=between_mean, bin_width_s=bin_width_s
        )

        # annotations slice & labeling
        ann_times_all, labels_all, ann_midpoints, ann_in_sorted = annotations_in_bout(
            annotations, a, b,
            seconds_col=seconds_col, tags_col=tags_col,
            start_kw=start_kw, end_kw=end_kw,
            relative_time=relative_time
        )

        # end offset choice
        bout_span = float(np.nanmax(x_plot) - np.nanmin(x_plot)) if x_plot.size else 0.0
        fallback = 0.05 * bout_span if bout_span > 0 else np.nan
        avg_offset = pick_end_offset(labels_all, ann_times_all, ann_midpoints, end_offset_strategy, fallback)

        # last-crossings
        last_cross_times, sigma_used_arr, varthr_used_arr = compute_last_crossings(
            ann_times_all, labels_all, ann_midpoints,
            bin_centers=bin_centers, variances=variances, between_std=between_std,
            sigma_multiplier=sigma_multiplier, sigma_step=sigma_step, min_sigma=min_sigma,
            end_offset=avg_offset
        )
        black_dot_times = np.where(np.isfinite(last_cross_times),
                                   0.5 * (ann_times_all + last_cross_times),
                                   np.nan)

        # plot
        a_disp, b_disp = (a, b) if not relative_time else (0.0, b - a)
        title = f"{title_prefix} {i+1}  [{a_disp:.3f}s–{b_disp:.3f}s]  (bins {bin_width_s*1e3:.0f} ms)"
        plot_one_bout_axis(
            ax, x_plot, y_bout,
            between_mean=between_mean,
            dot_times=black_dot_times,
            dot_size=dot_size,
            ylim=ylim,
            title=title,
            xlabel=abf.sweepLabelX,
            ylabel=abf.sweepLabelY,
        )

        # table rows (always build; merge happens later if requested)
        tbl = build_per_bout_table(
            i, a, b, ann_times_all, last_cross_times, black_dot_times, labels_all,
            relative_time=relative_time
        )
        # add sigma/threshold columns
        tbl["SigmaUsed"] = sigma_used_arr
        tbl["VarThresholdUsed"] = varthr_used_arr
        per_bout_tables.append(tbl)
    vrb_timings = pd.concat(per_bout_tables, ignore_index=True)

    # tidy figure
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    plt.show()

    # merged = merge_annotations(
    #     per_bout_df, annotations,
    #     seconds_col=seconds_col,
    #     merge_tolerance_s=merge_tolerance_s
    # )
    return vrb_timings
