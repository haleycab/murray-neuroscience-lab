# parent_folder_path = "/Users/haleyoro/Desktop/" # work on library computer
parent_folder_path = "/Users/Haley/Desktop/" # work on local computer

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
import os
from waveforms_helpers import *
import pickle


def split_cell_attached(sheets: dict[str, pd.DataFrame]):
    """
    Splits dictionary of DataFrames into:
      - matches: sheets with at least one 'Cell-attached' row
      - non_matches: sheets with no 'Cell-attached' rows
    """
    matches = {}
    non_matches = []

    for name, df in sheets.items():
        if "Type" not in df.columns:
            non_matches.append(name)
            continue

        mask = df["Type"] == "Cell-attached"

        if mask.any():
            matches[name] = df.loc[mask].copy()
        else:
            non_matches.append(name)

    return matches, non_matches

def split_cell_attached_spiking(sheets: dict[str, pd.DataFrame]):
    """
    Splits dictionary of DataFrames into:
      - matches: sheets with at least one 'Cell-attached' row
      - non_matches: sheets with no 'Cell-attached' rows
    """
    matches = {}
    non_matches = []

    for name, df in sheets.items():
        if "Type" not in df.columns:
            non_matches.append(name)
            continue

        mask = df["Type"] == "Cell-attached (spiking)"

        if mask.any():
            matches[name] = df.loc[mask].copy()
        else:
            non_matches.append(name)

    return matches, non_matches

import pandas as pd

def filter_cell_attached(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Return a new dictionary with only rows where Type is
    'Cell-attached' or 'Cell-attached (spiking)'.
    Sheets with no matches are excluded.
    """
    keep_types = {"Cell-attached", "Cell-attached (spiking)"}
    filtered = {}

    for name, df in sheets.items():
        if "Type" not in df.columns:
            continue
        mask = df["Type"].isin(keep_types)
        if mask.any():
            filtered[name] = df.loc[mask].copy()

    return filtered


def recalc_freq_cell_attached(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where Type == 'Cell-attached', calculate frequency as
    inverse of time difference between successive bursts (Seconds col).
    """
    df = df.copy()
    
    mask = df["Type"] == "Cell-attached"
    
    # Compute delta time only within Cell-attached rows
    cell_df = df.loc[mask]
    dt = cell_df["Seconds"].diff()  # time difference in seconds
    
    # Frequency = 1/dt (skip first row, will be NaN)
    df.loc[mask, "Freq"] = 1 / dt
    
    return df

def assign_spike_frequencies_one_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Like before, but additionally EXCLUDES spikes assigned to bursts tagged with 'bout start'.
    """
    out_rows = []

    for trace_name, g in df.groupby("Trace name", dropna=False):
        # Bursts (ventral root) and spikes
        bursts = g[g["Type"] == "Cell-attached"].copy().sort_values("Seconds")
        spikes = g[g["Type"] == "Cell-attached (spiking)"].copy().sort_values("Seconds")

        if len(bursts) < 2 or spikes.empty:
            continue

        t_burst = bursts["Seconds"].to_numpy()

        # Frequencies aligned to later burst
        freqs = np.empty(len(t_burst), dtype=float)
        freqs[:] = np.nan
        dt = np.diff(t_burst)
        with np.errstate(divide="ignore", invalid="ignore"):
            freqs[1:] = 1.0 / dt

        # Midpoints & bins
        midpoints = (t_burst[:-1] + t_burst[1:]) / 2.0
        bins = np.concatenate(([-np.inf], midpoints, [np.inf]))

        # Assign spikes to bins (maps to later burst i)
        s_times = spikes["Seconds"].to_numpy()
        bin_idx = np.digitize(s_times, bins, right=True)  # 1..K+1
        K = len(midpoints)

        # keep only spikes strictly between first/last midpoint
        valid_mask = (bin_idx >= 2) & (bin_idx <= K)
        if not valid_mask.any():
            continue

        s_times_valid = s_times[valid_mask]
        burst_indices = bin_idx[valid_mask]  # burst i (later burst), 2..K

        # ---- NEW: drop spikes whose associated later-burst is tagged 'bout start' ----
        if "Tags" in bursts.columns:
            burst_tags = bursts["Tags"].astype(str)
            is_bout_start = burst_tags.str.contains("bout start", case=False, na=False).to_numpy()
            # burst_indices are 1-based positions into 'freqs'/'bursts'; convert to 0-based
            keep_mask = ~is_bout_start[burst_indices - 1]
        else:
            keep_mask = np.ones_like(burst_indices, dtype=bool)

        if not keep_mask.any():
            continue

        s_times_valid = s_times_valid[keep_mask]
        burst_indices = burst_indices[keep_mask]

        # Map to frequencies
        assigned_freqs = freqs[burst_indices - 1]

        out_rows.append(pd.DataFrame({
            "Trace name": trace_name,
            "trace_time": s_times_valid,
            "spike_frequency": assigned_freqs,
            "burst_index": burst_indices
        }))

    if out_rows:
        return pd.concat(out_rows, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Trace name", "trace_time", "spike_frequency", "burst_index"])

def assign_spike_frequencies_for_dict(sheets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Apply the assignment to each sheet (DataFrame) in a dict.
    Returns a dict: sheet_name -> tidy result DataFrame
    """
    results = {}
    for name, df in sheets.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        results[name] = assign_spike_frequencies_one_df(df)
    return results




def combine_spike_dict(spike_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine dictionary of spike DataFrames into one big DataFrame.
    Adds a 'cell' column with the original dict key (sheet/df name).
    """
    frames = []
    for cell_name, df in spike_dict.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["cell"] = cell_name
        frames.append(tmp)
    if frames:
        return pd.concat(frames, ignore_index=True)
    else:
        return pd.DataFrame(columns=["Trace name", "trace_time", "spike_frequency", "burst_index", "cell"])
    

def _burst_frequencies(burst_times: np.ndarray) -> np.ndarray:
    """Frequency per burst (Hz), aligned to the later burst; first is NaN."""
    freqs = np.full(len(burst_times), np.nan, dtype=float)
    if len(burst_times) >= 2:
        dt = np.diff(burst_times)
        with np.errstate(divide="ignore", invalid="ignore"):
            freqs[1:] = 1.0 / dt
    return freqs

def _midpoints(burst_times: np.ndarray) -> np.ndarray:
    """Midpoints between successive bursts."""
    if len(burst_times) < 2:
        return np.array([], dtype=float)
    return (burst_times[:-1] + burst_times[1:]) / 2.0

def _assign_spikes_to_bursts(spike_times: np.ndarray, burst_times: np.ndarray, midpoints: np.ndarray):
    """
    Bin spikes by consecutive midpoints.
    Returns (kept_spike_times, burst_indices_for_spikes, assigned_freqs).
    Excludes spikes before first midpoint or after last midpoint.
    """
    if len(burst_times) < 2 or spike_times.size == 0:
        return np.array([]), np.array([], dtype=int), np.array([])

    bins = np.concatenate(([-np.inf], midpoints, [np.inf]))
    K = len(midpoints)
    bin_idx = np.digitize(spike_times, bins, right=True)  # 1..K+1
    keep = (bin_idx >= 2) & (bin_idx <= K)                # strictly inside midpoints
    if not keep.any():
        return np.array([]), np.array([], dtype=int), np.array([])

    kept_times = spike_times[keep]
    burst_idx = bin_idx[keep] - 1                         # 1..K-1 => bursts 1..K-1
    freqs = _burst_frequencies(burst_times)
    assigned = freqs[burst_idx]
    return kept_times, burst_idx, assigned

def _first5_window(burst_times: np.ndarray):
    """Return (x_start, x_end) spanning bursts #1..#5 (or last available)."""
    if burst_times.size == 0:
        return None, None
    x_start = burst_times[0]
    idx_end = min(4, len(burst_times)-1)  # 0-based; burst #5 is index 4
    x_end = burst_times[idx_end]
    return x_start, x_end

def plot_sheet_first5_bursts(sheet_df: pd.DataFrame, title_prefix: str | None = None):
    """
    Make a 1x2 plot (up to 2 traces) for a single sheet DataFrame.
    Each panel: spikes vs assigned ventral-root burst frequency,
    zoomed to the first 5 bursts of that trace.
    Uses constrained_layout and shared y-axis across panels.
    
    Bursts = orange vertical lines
    Midpoints = dashed gray lines with labels
    """
    traces = sheet_df["Trace name"].dropna().unique().tolist()[:2]
    if not traces:
        print("No Trace name found in sheet.")
        return

    n = len(traces)

    # First pass: compute per-trace data and collect y-values for global limits
    per_trace, y_values = [], []
    for tr in traces:
        g = sheet_df[sheet_df["Trace name"] == tr].copy()
        bursts = g[g["Type"] == "Cell-attached"].sort_values("Seconds")
        spikes = g[g["Type"] == "Cell-attached (spiking)"].sort_values("Seconds")

        if len(bursts) < 2:
            per_trace.append({"trace": tr, "valid": False})
            continue

        t_burst = bursts["Seconds"].to_numpy()
        mids = _midpoints(t_burst)
        x_start, x_end = _first5_window(t_burst)
        if x_start is None or x_end is None:
            per_trace.append({"trace": tr, "valid": False})
            continue

        s_times = spikes["Seconds"].to_numpy()
        s_times_valid, burst_idx, assigned_freqs = _assign_spikes_to_bursts(s_times, t_burst, mids)

        in_window = (s_times_valid >= x_start) & (s_times_valid <= x_end)
        s_plot = s_times_valid[in_window]
        f_plot = assigned_freqs[in_window]

        freqs = _burst_frequencies(t_burst)

        y_vals = []
        if f_plot.size:
            y_vals.append(f_plot[np.isfinite(f_plot)])
        mask_burst_in_window = (t_burst >= x_start) & (t_burst <= x_end)
        y_vals.append(freqs[mask_burst_in_window & np.isfinite(freqs)])
        if y_vals:
            y_values.append(np.concatenate(y_vals))

        per_trace.append({
            "trace": tr,
            "valid": True,
            "t_burst": t_burst,
            "mids": mids,
            "x_start": x_start,
            "x_end": x_end,
            "s_plot": s_plot,
            "f_plot": f_plot,
            "freqs": freqs
        })

    # Global y-limits
    if y_values:
        all_y = np.concatenate([y for y in y_values if y.size])
        if all_y.size:
            ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
            pad = 0.05 * (ymax - ymin if ymax > ymin else (ymax if ymax > 0 else 1.0))
            global_ylim = (max(0.0, ymin - pad), ymax + pad)
        else:
            global_ylim = None
    else:
        global_ylim = None

    # Figure
    fig, axes = plt.subplots(
        1, n, figsize=(8*n, 5),
        squeeze=False, sharey=True, constrained_layout=True
    )
    axes = axes.ravel()

    for i, data in enumerate(per_trace):
        ax = axes[i]
        tr = data["trace"]
        if not data["valid"]:
            ax.set_title(f"{tr} — need ≥2 bursts")
            ax.axis("off")
            continue

        t_burst = data["t_burst"]
        mids = data["mids"]
        x_start, x_end = data["x_start"], data["x_end"]
        s_plot, f_plot = data["s_plot"], data["f_plot"]
        freqs = data["freqs"]

        # Spikes
        if s_plot.size > 0:
            ax.scatter(s_plot, f_plot, label="Spikes (assigned freq)")

        # Bursts as vertical ORANGE lines
        for tb in t_burst:
            if x_start <= tb <= x_end:
                ax.axvline(tb, color="orange", linestyle="-", alpha=0.8, linewidth=1.5, label="Bursts" if tb == t_burst[0] else "")

        # Midpoints as dashed gray lines + labels
        ylim = ax.get_ylim()
        for k, m in enumerate(mids, start=1):
            if x_start <= m <= x_end:
                ax.axvline(m, linestyle="--", alpha=0.6, color="gray", linewidth=1)
                ax.text(m, ylim[1], f"m{k}", ha="center", va="bottom", fontsize=8, color="gray", rotation=90)

        # Annotate burst frequencies
        for j in range(1, len(t_burst)):  # skip first (NaN)
            if np.isfinite(freqs[j]) and (x_start <= t_burst[j] <= x_end):
                ax.text(t_burst[j], freqs[j], f"{freqs[j]:.2f} Hz",
                        ha="center", va="bottom", fontsize=8)

        label = f"{title_prefix} — {tr}" if title_prefix else tr
        ax.set_title(label, fontsize=10)
        ax.set_xlim(x_start, x_end)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Assigned burst frequency (Hz)")
        ax.legend(loc="best", fontsize=8)

        if global_ylim is not None:
            ax.set_ylim(*global_ylim)

    plt.show()


def clean_spikes_dataframe(df: pd.DataFrame, min_freq: float = 15) -> pd.DataFrame:
    """
    Clean the all_spikes DataFrame by dropping rows where spike_frequency is below min_freq.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a 'spike_frequency' column.
    min_freq : float, default=15
        Minimum frequency threshold; rows with spike_frequency < min_freq are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with low-frequency spikes removed.
    """
    if "spike_frequency" not in df.columns:
        raise ValueError("Input DataFrame must have a 'spike_frequency' column")

    cleaned = df.dropna(subset=["spike_frequency"]).copy()
    cleaned = cleaned[cleaned["spike_frequency"] >= min_freq]
    return cleaned.reset_index(drop=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- helpers --------------------

def _validate_df(df: pd.DataFrame):
    required = {"cell", "spike_frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

def order_cells(df: pd.DataFrame, by: str = "mean") -> list[str]:
    """
    Return cells ordered by ascending mean or median spike frequency.
    by : 'mean' or 'median'
    """
    _validate_df(df)
    if by not in {"mean", "median"}:
        raise ValueError("by must be 'mean' or 'median'")
    agg = (df.dropna(subset=["spike_frequency"])
             .groupby("cell")["spike_frequency"]
             .agg(mean="mean", median="median"))
    return agg[by].sort_values(kind="mergesort").index.tolist()

def plot_spike_dashes(df: pd.DataFrame, cell_order: list[str] | None = None, order_by: str = "mean",
                      base_lw: float = 1.5, lw_scale: float = 0.8, dash_halfwidth: float = 0.35,
                      show_means: bool = True, show_medians: bool = True,
                      spike_color: str = "tab:blue", mean_color: str = "tab:orange", median_color: str = "tab:green",
                      marker_size: float = 40.0, ax: plt.Axes | None = None):
    """
    Dash for each spike frequency per cell.
    - Spikes: blue horizontal dashes (thicker = more duplicates)
    - No SD bars.
    - Mean (orange) and Median (green) markers plotted above dashes (via zorder).
    """
    _validate_df(df)
    d = df.dropna(subset=["spike_frequency"]).copy()
    if cell_order is None:
        cell_order = order_cells(d, by=order_by)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(8, len(cell_order) * 0.7), 5), constrained_layout=True)

    # map cells to x positions
    xmap = {cell: i for i, cell in enumerate(cell_order, start=1)}
    means, medians, xs = [], [], []

    # Draw dashes (zorder=1 so markers can sit on top)
    for cell in cell_order:
        sub = d[d["cell"] == cell]
        if sub.empty:
            continue
        counts = sub["spike_frequency"].round(10).value_counts()
        x = xmap[cell]
        for freq, cnt in counts.items():
            lw = base_lw + lw_scale * (np.sqrt(cnt) - 1.0)
            ax.hlines(y=freq, xmin=x - dash_halfwidth, xmax=x + dash_halfwidth,
                      linewidth=lw, color=spike_color, zorder=1)

        means.append(sub["spike_frequency"].mean())
        medians.append(sub["spike_frequency"].median())
        xs.append(x)

    # Overlay stats (higher zorder so they’re on top of the dashes)
    if show_means:
        ax.scatter(xs, means, s=marker_size, color=mean_color, label="Mean",   zorder=4)
    if show_medians:
        ax.scatter(xs, medians, s=marker_size, color=median_color, label="Median", zorder=5)

    ax.set_title(f"Dash view (order by {order_by})")
    ax.set_ylabel("Spike frequency (Hz)")
    ax.set_xlabel("Cell")
    ax.set_xticks(list(xmap.values()))
    ax.set_xticklabels(cell_order, rotation=45, ha="right")
    if show_means or show_medians:
        ax.legend()

    if created_fig is not None:
        plt.show()

# -------------------- plot 2: violins with median/mean dots --------------------
def plot_violins_with_stats(df: pd.DataFrame, cell_order: list[str] | None = None, order_by: str = "mean",
                            show_means: bool = True, show_medians: bool = True,
                            mean_color: str = "tab:orange", median_color: str = "tab:green",
                            marker_size: float = 40.0, ax: plt.Axes | None = None):
    _validate_df(df)
    d = df.dropna(subset=["spike_frequency"]).copy()
    if cell_order is None:
        cell_order = order_cells(d, by=order_by)

    data = [d.loc[d["cell"] == c, "spike_frequency"].to_numpy() for c in cell_order]

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(8, len(cell_order) * 0.7), 5), constrained_layout=True)

    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.6)

    means = [np.nanmean(arr) if len(arr) else np.nan for arr in data]
    medians = [np.nanmedian(arr) if len(arr) else np.nan for arr in data]
    xs = np.arange(1, len(cell_order) + 1)

    if show_means:
        ax.scatter(xs, means, s=marker_size, color=mean_color, label="Mean")
    if show_medians:
        ax.scatter(xs, medians, s=marker_size, color=median_color, label="Median")

    ax.set_title(f"Violin + mean/median (order by {order_by})")
    ax.set_ylabel("Spike frequency (Hz)")
    ax.set_xlabel("Cell")
    ax.set_xticks(xs)
    ax.set_xticklabels(cell_order, rotation=45, ha="right")
    if show_means or show_medians:
        ax.legend()

    if created_fig is not None:
        plt.show()


# -------------------- plot 3: summary dots only (mean & median) --------------------
def plot_summary_dots(df: pd.DataFrame, cell_order: list[str] | None = None, order_by: str = "mean",
                      mean_color: str = "tab:orange", median_color: str = "tab:green",
                      marker_size: float = 50.0, ax: plt.Axes | None = None):
    _validate_df(df)
    d = df.dropna(subset=["spike_frequency"]).copy()
    if cell_order is None:
        cell_order = order_cells(d, by=order_by)

    agg = (d.groupby("cell")["spike_frequency"]
             .agg(mean="mean", median="median")
             .reindex(cell_order))

    xs = np.arange(1, len(cell_order) + 1)
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(8, len(cell_order) * 0.7), 5), constrained_layout=True)

    ax.scatter(xs, agg["mean"].to_numpy(), s=marker_size, color=mean_color, label="Mean")
    ax.scatter(xs, agg["median"].to_numpy(), s=marker_size, color=median_color, label="Median")

    ax.set_title(f"Summary dots (order by {order_by})")
    ax.set_ylabel("Spike frequency (Hz)")
    ax.set_xlabel("Cell")
    ax.set_xticks(xs)
    ax.set_xticklabels(cell_order, rotation=45, ha="right")
    ax.legend()

    if created_fig is not None:
        plt.show()


# -------------------- boxplot + jitter (matplotlib-only) --------------------
def plot_boxplot_spike_freq(df: pd.DataFrame, order: list[str] | None = None, order_by: str = "mean",
                            ax: plt.Axes | None = None):
    if order is None:
        order = order_cells(df, by=order_by)

    data = [df.loc[df["cell"] == c, "spike_frequency"].dropna().values for c in order]
    positions = np.arange(1, len(order) + 1)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.6), 5), constrained_layout=True)

    ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
               showfliers=False, boxprops=dict(facecolor="lightgray"))

    for i, y in enumerate(data, start=1):
        if len(y):
            x_jitter = np.random.normal(loc=i, scale=0.06, size=len(y))
            ax.scatter(x_jitter, y, color="tab:blue", alpha=0.6, s=12)

    ax.set_title("Boxplot + jitter")
    ax.set_ylabel("Spike frequency (Hz)")
    ax.set_xlabel("Cell")
    ax.set_xticks(positions)
    ax.set_xticklabels(order, rotation=45, ha="right")

    if created_fig is not None:
        plt.tight_layout()
        plt.show()


# -------------------- strip (swarm-like) --------------------
def plot_strip_spike_freq(df: pd.DataFrame, order: list[str] | None = None, order_by: str = "mean",
                          ax: plt.Axes | None = None):
    if order is None:
        order = order_cells(df, by=order_by)

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.6), 5), constrained_layout=True)

    for i, cell in enumerate(order, start=1):
        y = df.loc[df["cell"] == cell, "spike_frequency"].dropna().values
        if len(y):
            x_jitter = np.random.normal(loc=i, scale=0.10, size=len(y))
            ax.scatter(x_jitter, y, color="tab:blue", alpha=0.7, s=20)

    ax.set_title("Strip (swarm-like)")
    ax.set_ylabel("Spike frequency (Hz)")
    ax.set_xlabel("Cell")
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=45, ha="right")

    if created_fig is not None:
        plt.tight_layout()
        plt.show()
def plot_all_views_grid(all_spikes_df: pd.DataFrame, order_by: str = "mean", sharey: bool = True):
    """
    Render all five views in a 1x5 vertical grid by calling the existing
    plotting functions on provided axes.
    """
    _validate_df(all_spikes_df)
    if order_by not in {"mean", "median"}:
        raise ValueError("order_by must be 'mean' or 'median'")

    ordered_cells = order_cells(all_spikes_df, by=order_by)

    fig, axes = plt.subplots(5, 1, figsize=(15, 25), constrained_layout=True, sharey=sharey)
    ax_dash, ax_box, ax_strip, ax_violin, ax_summary = axes

    # Call your existing functions on the axes
    plot_spike_dashes(all_spikes_df, cell_order=ordered_cells, order_by=order_by, ax=ax_dash)
    plot_boxplot_spike_freq(all_spikes_df, order=ordered_cells, order_by=order_by, ax=ax_box)
    plot_strip_spike_freq(all_spikes_df, order=ordered_cells, order_by=order_by, ax=ax_strip)
    plot_violins_with_stats(all_spikes_df, cell_order=ordered_cells, order_by=order_by, ax=ax_violin)
    plot_summary_dots(all_spikes_df, cell_order=ordered_cells, order_by=order_by, ax=ax_summary)

    plt.show()
# Optional cleaning step

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_all_views_grid2(all_spikes_df: pd.DataFrame, order_by: str = "mean"):
    """
    Show 5 visualizations in a 1x5 vertical grid:
      [1] Dash view (blue dashes, Mean±SD + Median)
      [2] Boxplot + jitter
      [3] Strip (swarm-like)
      [4] Violin + mean/median
      [5] Summary dots (mean & median)
    Cells ordered by ascending mean/median (order_by).
    """
    # --------- validate & order ----------
    if {"cell", "spike_frequency"} - set(all_spikes_df.columns):
        raise ValueError("DataFrame must have 'cell' and 'spike_frequency' columns")
    if order_by not in {"mean", "median"}:
        raise ValueError("order_by must be 'mean' or 'median'")

    d = all_spikes_df.dropna(subset=["spike_frequency"]).copy()
    agg = (d.groupby("cell")["spike_frequency"]
             .agg(mean="mean", median="median"))
    ordered_cells = agg[order_by].sort_values(kind="mergesort").index.tolist()

    # Precompute per-cell arrays
    data = [d.loc[d["cell"] == c, "spike_frequency"].to_numpy() for c in ordered_cells]
    positions = np.arange(1, len(ordered_cells) + 1)

    # --------- layout: 5 rows, 1 column ----------
    fig, axes = plt.subplots(5, 1, figsize=(15, 25), constrained_layout=True)
    ax_dash, ax_box, ax_strip, ax_violin, ax_summary = axes.ravel()

    # --------- [1] Dash view ----------
    means, medians, stds = [], [], []
    for x, arr in zip(positions, data):
        vals, counts = np.unique(np.round(arr, 10), return_counts=True)
        for v, cnt in zip(vals, counts):
            lw = 1.5 + 0.8 * (np.sqrt(cnt) - 1.0)
            ax_dash.hlines(y=v, xmin=x - 0.35, xmax=x + 0.35,
                           linewidth=lw, color="tab:blue")
        means.append(np.nanmean(arr) if arr.size else np.nan)
        medians.append(np.nanmedian(arr) if arr.size else np.nan)
        stds.append(np.nanstd(arr, ddof=1) if arr.size > 1 else np.nan)

    ax_dash.errorbar(positions, means, yerr=stds, fmt='o', color="tab:orange",
                     markersize=4, capsize=4, label="Mean ± SD")
    ax_dash.scatter(positions, medians, s=40, color="tab:green", label="Median")
    ax_dash.set_title(f"Dash view (order by {order_by})")
    ax_dash.set_ylabel("Spike frequency (Hz)")
    ax_dash.set_xticks(positions)
    ax_dash.set_xticklabels(ordered_cells, rotation=45, ha="right")
    ax_dash.legend()

    # --------- [2] Boxplot + jitter ----------
    bp = ax_box.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, boxprops=dict(facecolor="lightgray"))
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x_jitter = np.random.normal(loc=i, scale=0.06, size=len(arr))
            ax_box.scatter(x_jitter, arr, color="tab:blue", alpha=0.6, s=12)
    ax_box.set_title("Boxplot + jitter")
    ax_box.set_ylabel("Spike frequency (Hz)")
    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(ordered_cells, rotation=45, ha="right")

    # --------- [3] Strip (swarm-like) ----------
    for i, arr in enumerate(data, start=1):
        if arr.size:
            x_jitter = np.random.normal(loc=i, scale=0.10, size=len(arr))
            ax_strip.scatter(x_jitter, arr, color="tab:blue", alpha=0.8, s=20)
    ax_strip.set_title("Strip (swarm-like)")
    ax_strip.set_ylabel("Spike frequency (Hz)")
    ax_strip.set_xticks(positions)
    ax_strip.set_xticklabels(ordered_cells, rotation=45, ha="right")

    # --------- [4] Violin + mean/median ----------
    if len(data) > 0:
        parts = ax_violin.violinplot(data, positions=positions, showmeans=False,
                                     showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.6)
    ax_violin.scatter(positions, means, s=40, color="tab:orange", label="Mean")
    ax_violin.scatter(positions, medians, s=40, color="tab:green", label="Median")
    ax_violin.set_title("Violin + mean/median")
    ax_violin.set_ylabel("Spike frequency (Hz)")
    ax_violin.set_xticks(positions)
    ax_violin.set_xticklabels(ordered_cells, rotation=45, ha="right")
    ax_violin.legend()

    # --------- [5] Summary dots ----------
    ax_summary.scatter(positions, agg.loc[ordered_cells, "mean"].to_numpy(),
                       s=50, color="tab:orange", label="Mean")
    ax_summary.scatter(positions, agg.loc[ordered_cells, "median"].to_numpy(),
                       s=50, color="tab:green", label="Median")
    ax_summary.set_title("Summary dots")
    ax_summary.set_ylabel("Spike frequency (Hz)")
    ax_summary.set_xticks(positions)
    ax_summary.set_xticklabels(ordered_cells, rotation=45, ha="right")
    ax_summary.legend()

    plt.show()

def plot_all_views(all_spikes_df: pd.DataFrame, order_by: str = "mean"):
    """
    Orders cells by ascending mean/median frequency, then renders ALL five views:
      1) Dash view (blue dashes, Mean±SD + Median markers)
      2) Boxplot + jittered points (pure matplotlib)
      3) Strip (swarm-like) plot (pure matplotlib)
      4) Violin + mean/median markers
      5) Summary dots (mean & median)
    """
    _validate_df(all_spikes_df)

    if order_by not in {"mean", "median"}:
        raise ValueError("order_by must be 'mean' or 'median'")

    # Compute one consistent cell order
    ordered_cells = order_cells(all_spikes_df, by=order_by)

    # 1) Dash view  (expects: df, cell_order)
    plot_spike_dashes(all_spikes_df, cell_order=ordered_cells)

    # 2) Boxplot + jitter (expects: df, order=...)
    plot_boxplot_spike_freq(all_spikes_df, order=ordered_cells)

    # 3) Strip (swarm-like) (expects: df, order=...)
    plot_strip_spike_freq(all_spikes_df, order=ordered_cells)

    # 4) Violin + stats (most versions accept cell_order; omit order_by to avoid mismatch)
    plot_violins_with_stats(all_spikes_df, cell_order=ordered_cells)

    # 5) Summary dots (most versions accept cell_order)
    plot_summary_dots(all_spikes_df, cell_order=ordered_cells)

    # Keyword

import pandas as pd
import numpy as np

def summarize_spike_stats(
    all_spikes: pd.DataFrame,
    cell_col: str = "cell",
    freq_col: str = "spike_frequency",
    dropna: bool = True,
    round_decimals: int | None = None,
) -> pd.DataFrame:
    """
    Summarize spike frequencies per cell.

    Parameters
    ----------
    all_spikes : DataFrame with at least [cell_col, freq_col]
    cell_col   : column name for cell IDs (default 'cell')
    freq_col   : column name for spike frequency (Hz) (default 'spike_frequency')
    dropna     : if True, drop rows with NaN freq before aggregating
    round_decimals : optional int to round numeric outputs

    Returns
    -------
    DataFrame with columns:
      ['cell', 'mean_spike_frequency', 'median_spike_frequency',
       'min_spike_frequency', 'max_spike_frequency', 'total_spikes']
    """
    required = {cell_col, freq_col}
    missing = required - set(all_spikes.columns)
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")

    df = all_spikes[[cell_col, freq_col]].copy()
    # ensure numeric freq; coerce bad strings to NaN
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")
    if dropna:
        df = df.dropna(subset=[freq_col])

    out = (
        df.groupby(cell_col)[freq_col]
          .agg(
              mean_spike_frequency="mean",
              median_spike_frequency="median",
              min_spike_frequency="min",
              max_spike_frequency="max",
              total_spikes="count",   # counts non-NaN
          )
          .reset_index()
          .rename(columns={cell_col: "cell"})
    )

    if round_decimals is not None:
        num_cols = ["mean_spike_frequency", "median_spike_frequency",
                    "min_spike_frequency", "max_spike_frequency"]
        out[num_cols] = out[num_cols].round(round_decimals)

    return out
