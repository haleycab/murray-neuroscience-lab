
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
import os
from waveforms_helpers2 import *
import pickle

parent_folder_path = "/Users/Haley/Desktop/" # work on local computer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def _extract_bout_windows(
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
def plot_between_bout_histogram(
    abf,
    annotations: pd.DataFrame,
    *,
    channel: int = 1,
    sweep: int = 0,
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,
    bins: int = 50
):
    """
    Plot histogram of ABF values in between swimming bouts.
    Marks the mean and ±1 std dev.
    """
    # get data
    abf.setSweep(sweepNumber=sweep, channel=channel)
    x, y = abf.sweepX, abf.sweepY

    # reuse bout window extraction
    windows = _extract_bout_windows(
        annotations,
        seconds_col=seconds_col,
        tags_col=tags_col,
        start_kw=start_kw.lower(),
        end_kw=end_kw.lower(),
        pad_s=pad_s,
    )

    mask = np.ones_like(y, dtype=bool)
    for (a, b) in windows:
        mask &= ~((x >= a) & (x <= b))

    between_vals = y[mask]

    if between_vals.size == 0:
        print("⚠️ No between-bout data available.")
        return

    mean_val = float(np.mean(between_vals))
    std_val  = float(np.std(between_vals))

    # plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(between_vals, bins=bins, color="lightblue", edgecolor="black", alpha=0.7)
    
    # mean line
    plt.axvline(mean_val, color="red", linestyle="-", linewidth=2, label=f"Mean = {mean_val:.3f}")
    # std dev lines
    plt.axvline(mean_val - std_val, color="green", linestyle="--", linewidth=2, label=f"Mean - 1 SD = {mean_val-std_val:.3f}")
    plt.axvline(mean_val + std_val, color="green", linestyle="--", linewidth=2, label=f"Mean + 1 SD = {mean_val+std_val:.3f}")

    plt.title("Histogram of Between-Bout Signal Values")
    plt.xlabel("Signal Amplitude")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

    return mean_val, std_val
def get_abf_and_annotations(abf_name, sheets_abfs):
    """
    Given an abf_name, return the abf file object and its corresponding
    annotations DataFrame subset.

    Parameters
    ----------
    abf_name : str
        The abf filename to look up (e.g., '2012_08_01_0009').
    sheets_abfs : dict
        Dictionary with structure:
        {sheet_name: {"abfs": {abf_name: abf_obj}, "annotations": DataFrame}}

    Returns
    -------
    tuple
        (abf_obj, annotations_subset) if found,
        otherwise (None, None).
    """
    for sheet, content in sheets_abfs.items():
        if abf_name in content["abfs"]:
            abf_obj = content["abfs"][abf_name]
            annotations_df = content["annotations"]
            annotations_subset = annotations_df[annotations_df["Trace name"] == abf_name]
            return abf_obj, annotations_subset
    return None, None

def plot_bout_raw_with_vrb_lastcross_and_midpoint_dots(
    abf,
    annotations: pd.DataFrame,
    *,
    bout_index: int = 0,
    channel: int = 1,
    sweep: int = 0,
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,
    n_bins: int = 500,
    relative_time: bool = False,
    between_std: float | None = None,     # pass std_val from between-bout histogram
    sigma_multiplier: float = 1.0,
    sigma_step: float = 0.1,              # per-annotation sigma decrement
    min_sigma: float = 0.0,               # sigma floor
    shade_exceeding: bool = True,
    dot_size: float = 60.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Unified annotation pipeline:
      - One ordered stream of annotations (start, vrb, end).
      - For each annotation k, find 'last-crossing' (orange) within its interval:
          * start/vrb: [t_k, midpoint(t_k, t_{k+1})]
          * end:       [t_k - avg_offset, t_k], where avg_offset comes from bout rhythm
        Adaptive threshold lowering by 0.1 σ per-annotation until a crossing is found.
      - Black dot at midpoint between annotation time and its own last-crossing.
    Returns a DataFrame with one row per annotation in the bout.
    """

    def _is_kw(tag, kw):
        return isinstance(tag, str) and (kw in tag.lower())

    def _adaptive_last_cross(t0: float, t1: float,
                             bin_centers: np.ndarray,
                             variances: np.ndarray,
                             between_std: float,
                             sigma0: float,
                             step: float,
                             sigma_floor: float):
        """Find last bin center in [t0,t1] whose variance exceeds (sigma*between_std)^2,
        lowering sigma in `step` increments until found. Fallback: last finite-variance
        bin in interval. Returns (t_cross, sigma_used, var_threshold_used)."""
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            return np.nan, np.nan, np.nan

        lo = max(t0, np.nanmin(bin_centers))
        hi = min(t1, np.nanmax(bin_centers))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.nan, np.nan, np.nan

        interval_mask = (bin_centers >= lo) & (bin_centers <= hi) & np.isfinite(variances)
        if not np.any(interval_mask):
            return np.nan, np.nan, np.nan

        if not (np.isfinite(between_std) and between_std > 0):
            idxs = np.where(interval_mask)[0]
            return (float(bin_centers[idxs[-1]]), np.nan, np.nan) if idxs.size else (np.nan, np.nan, np.nan)

        sigma = float(sigma0)
        while sigma >= sigma_floor - 1e-12:
            thr = (sigma * between_std) ** 2
            idxs = np.where(interval_mask & (variances > thr))[0]
            if idxs.size > 0:
                return float(bin_centers[idxs[-1]]), sigma, thr
            sigma -= step

        # Fallback
        idxs = np.where(interval_mask)[0]
        return (float(bin_centers[idxs[-1]]), np.nan, np.nan) if idxs.size else (np.nan, np.nan, np.nan)

    # --- sweep data ---
    abf.setSweep(sweepNumber=sweep, channel=channel)
    x, y = abf.sweepX, abf.sweepY

    # --- bout windows ---
    windows = _extract_bout_windows(
        annotations,
        seconds_col=seconds_col,
        tags_col=tags_col,
        start_kw=start_kw.lower(),
        end_kw=end_kw.lower(),
        pad_s=pad_s,
    )
    if not windows or bout_index >= len(windows):
        if verbose: print("⚠️ No bout found with given index.")
        return pd.DataFrame(columns=[
            "BoutIndex","BoutStart","BoutEnd",
            "AnnotationTime","AnnotationType","LastCrossTime","BlackDotTime",
            "SigmaUsed","VarThresholdUsed"
        ])

    start, end = windows[bout_index]
    # slice to bout
    in_bout = (x >= start) & (x <= end)
    if not np.any(in_bout):
        if verbose: print("⚠️ Bout window has no samples.")
        return pd.DataFrame(columns=[
            "BoutIndex","BoutStart","BoutEnd",
            "AnnotationTime","AnnotationType","LastCrossTime","BlackDotTime",
            "SigmaUsed","VarThresholdUsed"
        ])

    x_bout, y_bout = x[in_bout], y[in_bout]
    x_plot = (x_bout - start) if relative_time else x_bout

    # guard degenerate time span
    if x_plot.size == 0 or np.nanmin(x_plot) == np.nanmax(x_plot):
        if verbose: print("⚠️ Degenerate time span inside bout.")
        return pd.DataFrame(columns=[
            "BoutIndex","BoutStart","BoutEnd",
            "AnnotationTime","AnnotationType","LastCrossTime","BlackDotTime",
            "SigmaUsed","VarThresholdUsed"
        ])

    # variance of |signal - mean_bout|
    mean_bout = float(np.mean(y_bout)) if y_bout.size else np.nan
    y_dev = np.abs(y_bout - mean_bout)

    # binning (last bin inclusive)
    bins = np.linspace(x_plot.min(), x_plot.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    variances = np.empty(n_bins, dtype=float)
    for i in range(n_bins):
        if i < n_bins - 1:
            sel = (x_plot >= bins[i]) & (x_plot <  bins[i + 1])
        else:
            sel = (x_plot >= bins[i]) & (x_plot <= bins[i + 1])
        variances[i] = np.var(y_dev[sel]) if np.any(sel) else np.nan

    # global variance threshold (for shading only)
    var_thresh_global = None
    if between_std is not None and np.isfinite(between_std):
        var_thresh_global = float((sigma_multiplier * between_std) ** 2)

    # --- annotations in bout (ordered) ---
    ann_in_bout = annotations[
        (annotations[seconds_col].astype(float) >= start) &
        (annotations[seconds_col].astype(float) <= end)
    ].copy()

    # tag classification
    ann_in_bout["_is_start"] = ann_in_bout[tags_col].apply(lambda s: _is_kw(s, start_kw))
    ann_in_bout["_is_end"]   = ann_in_bout[tags_col].apply(lambda s: _is_kw(s, end_kw))
    ann_in_bout["_is_vrb"]   = ~(ann_in_bout["_is_start"] | ann_in_bout["_is_end"])

    # times (relative if requested) and labels
    ann_times_abs = ann_in_bout[seconds_col].astype(float).to_numpy()
    ann_times_all = ann_times_abs - start if relative_time else ann_times_abs
    labels_all = np.where(ann_in_bout["_is_start"].to_numpy(), "start",
                   np.where(ann_in_bout["_is_end"].to_numpy(), "end", "vrb"))

    # order by time
    order = np.argsort(ann_times_all)
    ann_times_all = ann_times_all[order]
    labels_all = labels_all[order]

    # midpoints between successive *all* annotations
    ann_midpoints = 0.5 * (ann_times_all[:-1] + ann_times_all[1:]) if len(ann_times_all) >= 2 else np.array([], dtype=float)

    # ---- Average offset for END intervals (bout rhythm) ----
    # prefer median of (VRB -> its next midpoint)
    vrb_mask = (labels_all == "vrb")
    vrb_to_mid_offsets = []
    if ann_midpoints.size:
        for i in range(len(ann_times_all) - 1):
            if vrb_mask[i]:
                vrb_to_mid_offsets.append(ann_midpoints[i] - ann_times_all[i])
    vrb_to_mid_offsets = np.array(vrb_to_mid_offsets, dtype=float)
    vrb_to_mid_offsets = vrb_to_mid_offsets[np.isfinite(vrb_to_mid_offsets) & (vrb_to_mid_offsets > 0)]

    # next preference: 0.5 * median inter-VRB spacing
    inter_vrb = np.diff(ann_times_all[vrb_mask]) if np.sum(vrb_mask) >= 2 else np.array([], dtype=float)
    inter_vrb = inter_vrb[np.isfinite(inter_vrb) & (inter_vrb > 0)]
    half_inter_vrb = 0.5 * inter_vrb if inter_vrb.size else np.array([], dtype=float)

    # final structured fallback: 0.5 * median inter-annotation spacing
    inter_ann = np.diff(ann_times_all) if ann_times_all.size >= 2 else np.array([], dtype=float)
    inter_ann = inter_ann[np.isfinite(inter_ann) & (inter_ann > 0)]
    half_inter_ann = 0.5 * inter_ann if inter_ann.size else np.array([], dtype=float)

    if vrb_to_mid_offsets.size:
        avg_offset = float(np.median(vrb_to_mid_offsets))
    elif half_inter_vrb.size:
        avg_offset = float(np.median(half_inter_vrb))
    elif half_inter_ann.size:
        avg_offset = float(np.median(half_inter_ann))
    else:
        avg_offset = 0.05 * (np.nanmax(x_plot) - np.nanmin(x_plot))

    # ---- Per-annotation intervals and last-crossings (unified arrays) ----
    last_cross_times = np.full_like(ann_times_all, np.nan, dtype=float)
    sigma_used_arr   = np.full_like(ann_times_all, np.nan, dtype=float)
    varthr_used_arr  = np.full_like(ann_times_all, np.nan, dtype=float)

    for k, t_ann in enumerate(ann_times_all):
        label = labels_all[k]
        if label == "end":
            t0, t1 = t_ann , t_ann + avg_offset
        else:
            # need next midpoint
            if k >= len(ann_times_all) - 1:
                t0 = t1 = np.nan  # no interval
            else:
                t0 = t_ann
                t1 = ann_midpoints[k]
        t_cross, sigma_used, thr_used = _adaptive_last_cross(
            t0, t1, bin_centers, variances,
            between_std=between_std if between_std is not None else np.nan,
            sigma0=sigma_multiplier, step=sigma_step, sigma_floor=min_sigma
        )
        last_cross_times[k] = t_cross
        sigma_used_arr[k]   = sigma_used
        varthr_used_arr[k]  = thr_used

    # black dots = midpoint between annotation and its own last-crossing
    black_dot_times = np.where(np.isfinite(last_cross_times),
                               0.5 * (ann_times_all + last_cross_times),
                               np.nan)

    # ===================== PLOTTING =====================
    plt.figure(figsize=(12, 4))

    # raw signal
    plt.plot(x_plot, y_bout, linewidth=0.9, label="Raw signal")

    # shaded bins above global threshold
    if var_thresh_global is not None and shade_exceeding:
        for i, v in enumerate(variances):
            if np.isfinite(v) and v > var_thresh_global:
                plt.axvspan(bins[i], bins[i+1], alpha=0.12)

    # annotation lines by type (but using unified arrays)
    for t, lab in zip(ann_times_all, labels_all):
        if lab == "vrb":
            plt.axvline(t, linestyle="--", color="blue", linewidth=1.2)
        elif lab == "start":
            plt.axvline(t, linestyle="--", color="tab:green", linewidth=1.2)
        else:  # end
            plt.axvline(t, linestyle="--", color="tab:red", linewidth=1.2)

    # orange last-crossing lines (unified)
    for t in last_cross_times:
        if np.isfinite(t):
            plt.axvline(t, linestyle="-", color="orange", linewidth=1.6)

    # black dots (unified)
    finite_mask = np.isfinite(black_dot_times)
    if np.any(finite_mask):
        plt.scatter(black_dot_times[finite_mask],
                    np.zeros(np.sum(finite_mask)),
                    s=dot_size, color="black", zorder=5)

    title_suffix = " (relative)" if relative_time else ""
    plt.title(f"Bout {bout_index}: vrb annotations original start, detected end (orange), and middle (black) {title_suffix}")
    plt.xlabel("Time (s)" + (" since bout start" if relative_time else ""))
    plt.ylabel(abf.sweepLabelY)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.show()

    # ===================== OUTPUT DATAFRAME =====================
    # Include absolute times in the DF for convenience
    if relative_time:
        abs_ann_times = ann_times_all + start
        abs_last_cross = np.where(np.isfinite(last_cross_times), last_cross_times + start, np.nan)
        abs_black_dots = np.where(np.isfinite(black_dot_times), black_dot_times + start, np.nan)
    else:
        abs_ann_times = ann_times_all.copy()
        abs_last_cross = last_cross_times.copy()
        abs_black_dots = black_dot_times.copy()

    df = pd.DataFrame({
        "BoutIndex": bout_index,
        "BoutStart": start,
        "BoutEnd": end,
        "AnnotationType": labels_all,         # 'start' | 'vrb' | 'end'
        "AnnotationTime": abs_ann_times,      # same length vector
        "LastCrossTime": abs_last_cross,      # same length vector
        "BlackDotTime": abs_black_dots,       # same length vector
        "SigmaUsed": sigma_used_arr,
        "VarThresholdUsed": varthr_used_arr,
    })

    return df
def center_vrbs(
    abf,
    annotations: pd.DataFrame,
    *,
    bout_index: int = 0,
    channel: int = 1,
    sweep: int = 0,
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,
    n_bins: int = 500,
    relative_time: bool = False,
    between_std: float | None = None,     # pass std_val from between-bout histogram
    sigma_multiplier: float = 1.0,
    sigma_step: float = 0.1,              # per-annotation sigma decrement
    min_sigma: float = 0.0,               # sigma floor
    shade_exceeding: bool = True,
    dot_size: float = 60.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Unified annotation pipeline:
      - One ordered stream of annotations (start, vrb, end).
      - For each annotation k, find 'last-crossing' (orange) within its interval:
          * start/vrb: [t_k, midpoint(t_k, t_{k+1})]
          * end:       [t_k - avg_offset, t_k], where avg_offset comes from bout rhythm
        Adaptive threshold lowering by 0.1 σ per-annotation until a crossing is found.
      - Black dot at midpoint between annotation time and its own last-crossing.
    Returns a DataFrame with one row per annotation in the bout.
    """

    def _is_kw(tag, kw):
        return isinstance(tag, str) and (kw in tag.lower())

    def _adaptive_last_cross(t0: float, t1: float,
                             bin_centers: np.ndarray,
                             variances: np.ndarray,
                             between_std: float,
                             sigma0: float,
                             step: float,
                             sigma_floor: float):
        """Find last bin center in [t0,t1] whose variance exceeds (sigma*between_std)^2,
        lowering sigma in `step` increments until found. Fallback: last finite-variance
        bin in interval. Returns (t_cross, sigma_used, var_threshold_used)."""
        if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
            return np.nan, np.nan, np.nan

        lo = max(t0, np.nanmin(bin_centers))
        hi = min(t1, np.nanmax(bin_centers))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.nan, np.nan, np.nan

        interval_mask = (bin_centers >= lo) & (bin_centers <= hi) & np.isfinite(variances)
        if not np.any(interval_mask):
            return np.nan, np.nan, np.nan

        if not (np.isfinite(between_std) and between_std > 0):
            idxs = np.where(interval_mask)[0]
            return (float(bin_centers[idxs[-1]]), np.nan, np.nan) if idxs.size else (np.nan, np.nan, np.nan)

        sigma = float(sigma0)
        while sigma >= sigma_floor - 1e-12:
            thr = (sigma * between_std) ** 2
            idxs = np.where(interval_mask & (variances > thr))[0]
            if idxs.size > 0:
                return float(bin_centers[idxs[-1]]), sigma, thr
            sigma -= step

        # Fallback
        idxs = np.where(interval_mask)[0]
        return (float(bin_centers[idxs[-1]]), np.nan, np.nan) if idxs.size else (np.nan, np.nan, np.nan)

    # --- sweep data ---
    abf.setSweep(sweepNumber=sweep, channel=channel)
    x, y = abf.sweepX, abf.sweepY

    # --- bout windows ---
    windows = _extract_bout_windows(
        annotations,
        seconds_col=seconds_col,
        tags_col=tags_col,
        start_kw=start_kw.lower(),
        end_kw=end_kw.lower(),
        pad_s=pad_s,
    )
    if not windows or bout_index >= len(windows):
        if verbose: print("⚠️ No bout found with given index.")
        return pd.DataFrame(columns=[
            "BoutIndex","BoutStart","BoutEnd",
            "AnnotationTime","AnnotationType","LastCrossTime","BlackDotTime",
            "SigmaUsed","VarThresholdUsed"
        ])

    start, end = windows[bout_index]
    # slice to bout
    in_bout = (x >= start) & (x <= end)
    if not np.any(in_bout):
        if verbose: print("⚠️ Bout window has no samples.")
        return pd.DataFrame(columns=[
            "BoutIndex","BoutStart","BoutEnd",
            "AnnotationTime","AnnotationType","LastCrossTime","BlackDotTime",
            "SigmaUsed","VarThresholdUsed"
        ])

    x_bout, y_bout = x[in_bout], y[in_bout]
    x_plot = (x_bout - start) if relative_time else x_bout

    # guard degenerate time span
    if x_plot.size == 0 or np.nanmin(x_plot) == np.nanmax(x_plot):
        if verbose: print("⚠️ Degenerate time span inside bout.")
        return pd.DataFrame(columns=[
            "BoutIndex","BoutStart","BoutEnd",
            "AnnotationTime","AnnotationType","LastCrossTime","BlackDotTime",
            "SigmaUsed","VarThresholdUsed"
        ])

    # variance of |signal - mean_bout|
    mean_bout = float(np.mean(y_bout)) if y_bout.size else np.nan
    y_dev = np.abs(y_bout - mean_bout)

    # binning (last bin inclusive)
    bins = np.linspace(x_plot.min(), x_plot.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    variances = np.empty(n_bins, dtype=float)
    for i in range(n_bins):
        if i < n_bins - 1:
            sel = (x_plot >= bins[i]) & (x_plot <  bins[i + 1])
        else:
            sel = (x_plot >= bins[i]) & (x_plot <= bins[i + 1])
        variances[i] = np.var(y_dev[sel]) if np.any(sel) else np.nan

    # global variance threshold (for shading only)
    var_thresh_global = None
    if between_std is not None and np.isfinite(between_std):
        var_thresh_global = float((sigma_multiplier * between_std) ** 2)

    # --- annotations in bout (ordered) ---
    ann_in_bout = annotations[
        (annotations[seconds_col].astype(float) >= start) &
        (annotations[seconds_col].astype(float) <= end)
    ].copy()

    # tag classification
    ann_in_bout["_is_start"] = ann_in_bout[tags_col].apply(lambda s: _is_kw(s, start_kw))
    ann_in_bout["_is_end"]   = ann_in_bout[tags_col].apply(lambda s: _is_kw(s, end_kw))
    ann_in_bout["_is_vrb"]   = ~(ann_in_bout["_is_start"] | ann_in_bout["_is_end"])

    # times (relative if requested) and labels
    ann_times_abs = ann_in_bout[seconds_col].astype(float).to_numpy()
    ann_times_all = ann_times_abs - start if relative_time else ann_times_abs
    labels_all = np.where(ann_in_bout["_is_start"].to_numpy(), "start",
                   np.where(ann_in_bout["_is_end"].to_numpy(), "end", "vrb"))

    # order by time
    order = np.argsort(ann_times_all)
    ann_times_all = ann_times_all[order]
    labels_all = labels_all[order]

    # midpoints between successive *all* annotations
    ann_midpoints = 0.5 * (ann_times_all[:-1] + ann_times_all[1:]) if len(ann_times_all) >= 2 else np.array([], dtype=float)

    # ---- Average offset for END intervals (bout rhythm) ----
    # prefer median of (VRB -> its next midpoint)
    vrb_mask = (labels_all == "vrb")
    vrb_to_mid_offsets = []
    if ann_midpoints.size:
        for i in range(len(ann_times_all) - 1):
            if vrb_mask[i]:
                vrb_to_mid_offsets.append(ann_midpoints[i] - ann_times_all[i])
    vrb_to_mid_offsets = np.array(vrb_to_mid_offsets, dtype=float)
    vrb_to_mid_offsets = vrb_to_mid_offsets[np.isfinite(vrb_to_mid_offsets) & (vrb_to_mid_offsets > 0)]

    # next preference: 0.5 * median inter-VRB spacing
    inter_vrb = np.diff(ann_times_all[vrb_mask]) if np.sum(vrb_mask) >= 2 else np.array([], dtype=float)
    inter_vrb = inter_vrb[np.isfinite(inter_vrb) & (inter_vrb > 0)]
    half_inter_vrb = 0.5 * inter_vrb if inter_vrb.size else np.array([], dtype=float)

    # final structured fallback: 0.5 * median inter-annotation spacing
    inter_ann = np.diff(ann_times_all) if ann_times_all.size >= 2 else np.array([], dtype=float)
    inter_ann = inter_ann[np.isfinite(inter_ann) & (inter_ann > 0)]
    half_inter_ann = 0.5 * inter_ann if inter_ann.size else np.array([], dtype=float)

    if vrb_to_mid_offsets.size:
        avg_offset = float(np.median(vrb_to_mid_offsets))
    elif half_inter_vrb.size:
        avg_offset = float(np.median(half_inter_vrb))
    elif half_inter_ann.size:
        avg_offset = float(np.median(half_inter_ann))
    else:
        avg_offset = 0.05 * (np.nanmax(x_plot) - np.nanmin(x_plot))

    # ---- Per-annotation intervals and last-crossings (unified arrays) ----
    last_cross_times = np.full_like(ann_times_all, np.nan, dtype=float)
    sigma_used_arr   = np.full_like(ann_times_all, np.nan, dtype=float)
    varthr_used_arr  = np.full_like(ann_times_all, np.nan, dtype=float)

    for k, t_ann in enumerate(ann_times_all):
        label = labels_all[k]
        if label == "end":
            t0, t1 = t_ann , t_ann + avg_offset
        else:
            # need next midpoint
            if k >= len(ann_times_all) - 1:
                t0 = t1 = np.nan  # no interval
            else:
                t0 = t_ann
                t1 = ann_midpoints[k]
        t_cross, sigma_used, thr_used = _adaptive_last_cross(
            t0, t1, bin_centers, variances,
            between_std=between_std if between_std is not None else np.nan,
            sigma0=sigma_multiplier, step=sigma_step, sigma_floor=min_sigma
        )
        last_cross_times[k] = t_cross
        sigma_used_arr[k]   = sigma_used
        varthr_used_arr[k]  = thr_used

    # black dots = midpoint between annotation and its own last-crossing
    black_dot_times = np.where(np.isfinite(last_cross_times),
                               0.5 * (ann_times_all + last_cross_times),
                               np.nan)

    
    # ===================== OUTPUT DATAFRAME =====================
    # Include absolute times in the DF for convenience
    if relative_time:
        abs_ann_times = ann_times_all + start
        abs_last_cross = np.where(np.isfinite(last_cross_times), last_cross_times + start, np.nan)
        abs_black_dots = np.where(np.isfinite(black_dot_times), black_dot_times + start, np.nan)
    else:
        abs_ann_times = ann_times_all.copy()
        abs_last_cross = last_cross_times.copy()
        abs_black_dots = black_dot_times.copy()

    df = pd.DataFrame({
        "BoutIndex": bout_index,
        "BoutStart": start,
        "BoutEnd": end,
        "AnnotationType": labels_all,         # 'start' | 'vrb' | 'end'
        "AnnotationTime": abs_ann_times,      # same length vector
        "LastCrossTime": abs_last_cross,      # same length vector
        "BlackDotTime": abs_black_dots,       # same length vector
        "SigmaUsed": sigma_used_arr,
        "VarThresholdUsed": varthr_used_arr,
    })

    return df
import numpy as np
import pandas as pd

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
    windows = _extract_bout_windows(
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



def plot_bouts(
    abf,
    annotations: pd.DataFrame,
    *,
    channel: int = 1,
    sweep: int = 0,
    seconds_col: str = "Seconds",
    tags_col: str = "Tags",
    start_kw: str = "bout start",
    end_kw: str = "bout end",
    pad_s: float = 0.010,
    ylim = (-0.2, 0.2),
    ncols: int = 2,
    max_bouts: int | None = None,
    draw_exact_tag_lines: bool = True,
    title_prefix: str = "Bout"
):
    """
    Plot the ABF sweep zoomed into each bout window (±pad_s), one panel per bout.

    Parameters
    ----------
    abf : pyABF.ABF
    annotations : pd.DataFrame (filtered to this ABF)
    channel, sweep : which trace to plot
    seconds_col, tags_col, start_kw, end_kw : annotation parsing controls
    pad_s : float  padding (seconds) before start and after end
    ylim : tuple   y-limits for all panels
    ncols : int    subplot columns
    max_bouts : int | None  limit number of bouts plotted (for long recordings)
    draw_exact_tag_lines : bool  draw dotted lines at raw tag times too
    title_prefix : str  panel title prefix
    """
    # Prepare ABF data
    abf.setSweep(sweepNumber=sweep, channel=channel)
    x, y = abf.sweepX, abf.sweepY

    # Compute windows
    windows = _extract_bout_windows(
        annotations,
        seconds_col=seconds_col,
        tags_col=tags_col,
        start_kw=start_kw,
        end_kw=end_kw,
        pad_s=pad_s,
    )
    if not windows:
        print("No bout windows found.")
        return []

    if max_bouts is not None:
        windows = windows[:max_bouts]

    n = len(windows)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 3.2*nrows), squeeze=False)
    axes = axes.ravel()

    for i, (a, b) in enumerate(windows):
        ax = axes[i]
        ax.plot(x, y, linewidth=0.6, label=f"Ch {channel+1}")
        ax.axvspan(a, b, alpha=0.15)
        # dashed at padded edges
        ax.axvline(a, linestyle="--", linewidth=0.8, alpha=0.9)
        ax.axvline(b, linestyle="--", linewidth=0.8, alpha=0.9)

        if draw_exact_tag_lines and len(annotations) > 0:
            for _, row in annotations.iterrows():
                t = float(row[seconds_col])
                if a <= t <= b:
                    ax.axvline(t, linestyle=":", linewidth=0.8, alpha=0.7)

        ax.set_xlim(a, b)
        ax.set_ylim(*ylim)
        ax.set_title(f"{title_prefix} {i+1}  [{a:.3f}s–{b:.3f}s]")
        ax.set_ylabel(abf.sweepLabelY)
        ax.set_xlabel(abf.sweepLabelX)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()

    return windows