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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def histogram_cell_medians(
    df: pd.DataFrame,
    cell_col: str = "cell",
    freq_col: str = "spike_frequency",
    bins: int | str = "auto",     # or an int
    use_fd_rule: bool = False,    # Freedman–Diaconis bin rule
    color: str = "tab:blue",
    edgecolor: str = "black",
    alpha: float = 0.85,
    show_mean: bool = True,
    show_median: bool = True,
    annotate: bool = True,
    ax: plt.Axes | None = None,
):
    """
    Plot a histogram of per-cell median spike frequencies.

    Returns
    -------
    medians : pd.Series (index=cell, values=median frequencies)
    """
    if cell_col not in df.columns or freq_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{cell_col}' and '{freq_col}'")

    d = df.copy()
    d[freq_col] = pd.to_numeric(d[freq_col], errors="coerce")
    d = d.dropna(subset=[freq_col, cell_col])

    # per-cell medians
    medians = d.groupby(cell_col)[freq_col].median().dropna()
    n = medians.size
    if n == 0:
        raise ValueError("No valid medians to plot.")

    # Freedman–Diaconis optional bin selection
    if use_fd_rule and n >= 2:
        q75, q25 = np.percentile(medians, [75, 25])
        iqr = max(q75 - q25, 1e-9)
        bin_width = 2 * iqr * n ** (-1/3)
        data_range = medians.max() - medians.min()
        bins = max(1, int(np.ceil(data_range / bin_width))) if bin_width > 0 else "auto"

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    # histogram
    counts, edges, _ = ax.hist(medians.values, bins=bins, color=color,
                               edgecolor=edgecolor, alpha=alpha)

    # mean/median lines
    if show_mean:
        m = medians.mean()
        ax.axvline(m, linestyle="--", linewidth=1.5, label=f"Mean = {m:.2f} Hz")
    if show_median:
        md = medians.median()
        ax.axvline(md, linestyle="-", linewidth=1.5, label=f"Median = {md:.2f} Hz")

    if show_mean or show_median:
        ax.legend(fontsize=9)

    ax.set_title(f"Histogram of cell median spike frequency (n={n} cells)")
    ax.set_xlabel("Cell median spike frequency (Hz)")
    ax.set_ylabel("Count of cells")
    ax.grid(axis="y", alpha=0.2)

    if created_fig is not None:
        plt.show()

    return medians
import numpy as np
import pandas as pd

def _ls_fit_metrics(x, y):
    """Return slope, intercept, R2, RMSE, AIC, BIC for y ~ a + b*x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]; y = y[good]
    n = len(y)
    if n < 3:
        return dict(n=n, slope=np.nan, intercept=np.nan, R2=np.nan, RMSE=np.nan, AIC=np.nan, BIC=np.nan)
    X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta
    yhat = X @ beta
    resid = y - yhat
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean())**2))
    R2 = 1.0 - sse/sst if sst > 0 else np.nan
    rmse = np.sqrt(sse / n)
    k = 2  # intercept + slope
    AIC = n * np.log(sse / n) + 2 * k
    BIC = n * np.log(sse / n) + k * np.log(n)
    return dict(n=n, slope=slope, intercept=intercept, R2=R2, RMSE=rmse, AIC=AIC, BIC=BIC)

def compare_x_transforms(per_cell: pd.DataFrame,
                         rin_col="rin",
                         y_col="median_spike_frequency"):
    """
    per_cell needs columns: rin (MΩ, >0), median_spike_frequency.
    Returns a table of metrics for x = Rin, log10(Rin), ln(Rin), sqrt(Rin).
    """
    df = per_cell.copy()
    df[rin_col] = pd.to_numeric(df[rin_col], errors="coerce")
    df = df[(df[rin_col] > 0) & df[y_col].notna()].copy()

    tests = {
        "raw Rin": df[rin_col].values,
        "log10(Rin)": np.log10(df[rin_col].values),
        "ln(Rin)": np.log(df[rin_col].values),
        "sqrt(Rin)": np.sqrt(df[rin_col].values),
    }
    rows = []
    for name, x in tests.items():
        m = _ls_fit_metrics(x, df[y_col].values)
        m["transform"] = name
        rows.append(m)
    out = (pd.DataFrame(rows)
             .loc[:, ["transform", "n", "R2", "RMSE", "AIC", "BIC", "slope", "intercept"]]
             .sort_values(["AIC", "BIC"]))
    return out
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else np.nan

def scatter_rin_vs_median(
    all_spikes_enriched: pd.DataFrame,
    cell_col: str = "cell",
    freq_col: str = "spike_frequency",
    rin_col: str = "Input Resistance",
    motoneuron_col: str = "Motoneuron",
    celltype_col: str = "Cell Type",
    color_by: str | None = "Motoneuron",   # "Motoneuron", "Cell Type", or None
    fit_line: bool = False,                 # fit y = a + b*Rin across ALL cells
    annotate_fit: bool = True,              # print a,b,R^2 on the plot
    marker_size: float = 40.0,
    alpha: float = 0.9,
    ax: plt.Axes | None = None,
):
    """
    Scatter: Input Resistance (raw MΩ) vs median spike frequency (one dot per cell).
    Optionally color by a category and draw a global linear fit.

    Returns
    -------
    per_cell : DataFrame with ['cell','median_spike_frequency','rin', ...]
    fit_info : dict or None  (keys: 'slope','intercept','r2','n') if fit_line=True and n>=2
    """
    # ---- validate & coerce ----
    need_cols = {cell_col, freq_col, rin_col}
    missing = need_cols - set(all_spikes_enriched.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = all_spikes_enriched.copy()
    d[freq_col] = pd.to_numeric(d[freq_col], errors="coerce")
    d[rin_col]  = pd.to_numeric(d[rin_col],  errors="coerce")

    # ---- per-cell summary ----
    agg_dict = {
        "median_spike_frequency": (freq_col, "median"),
        "rin": (rin_col, "median"),  # Rin expected constant per cell; median is robust
    }
    if motoneuron_col in d.columns:
        agg_dict[motoneuron_col] = (motoneuron_col, _mode_or_nan)
    if celltype_col in d.columns:
        agg_dict[celltype_col] = (celltype_col, _mode_or_nan)

    per_cell = (
        d.groupby(cell_col).agg(**agg_dict)
         .reset_index()
         .rename(columns={cell_col: "cell"})
    )

    # Drop invalid rows
    per_cell = per_cell.dropna(subset=["rin", "median_spike_frequency"]).copy()

    # ---- plotting ----
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    # Color by category via multiple scatter calls
    label_title = None
    if color_by == "Motoneuron" and motoneuron_col in per_cell.columns:
        label_title = "Motoneuron"
        cats = pd.Categorical(per_cell[motoneuron_col]).categories.tolist()
        for cat in cats:
            sub = per_cell[per_cell[motoneuron_col] == cat]
            ax.scatter(sub["rin"], sub["median_spike_frequency"],
                       s=marker_size, alpha=alpha, label=str(cat))
    elif color_by == "Cell Type" and celltype_col in per_cell.columns:
        label_title = "Cell Type"
        cats = pd.Categorical(per_cell[celltype_col]).categories.tolist()
        for cat in cats:
            sub = per_cell[per_cell[celltype_col] == cat]
            ax.scatter(sub["rin"], sub["median_spike_frequency"],
                       s=marker_size, alpha=alpha, label=str(cat))
    else:
        # no color coding
        ax.scatter(per_cell["rin"], per_cell["median_spike_frequency"],
                   s=marker_size, alpha=alpha)

    # ---- optional global linear fit (raw Rin) ----
    fit_info = None
    if fit_line and len(per_cell) >= 2:
        x = per_cell["rin"].to_numpy()
        y = per_cell["median_spike_frequency"].to_numpy()
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() >= 2:
            xg, yg = x[good], y[good]
            slope, intercept = np.polyfit(xg, yg, deg=1)
            r = np.corrcoef(xg, yg)[0, 1]
            r2 = float(r*r) if np.isfinite(r) else np.nan
            xfit = np.linspace(xg.min(), xg.max(), 200)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, label=f"Fit (R²={r2:.2f})")
            if annotate_fit:
                txt = f"y = {intercept:.2f} + {slope:.4f}·Rin\nR² = {r2:.2f} (n={good.sum()})"
                ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                        ha="left", va="top", fontsize=9)
            fit_info = {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(good.sum())}

    # labels, legend, grid
    ax.set_xlabel("Input Resistance (MΩ)")
    ax.set_ylabel("Median spike frequency (Hz)")
    title_suffix = f"colored by {label_title}" if label_title else "uncolored"
    ax.set_title(f"Median Frequency vs Input Resistance — {title_suffix}")
    if label_title:
        ax.legend(fontsize=9, title=label_title)
    else:
        # show fit in legend if uncolored
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    if created_fig is not None:
        plt.show()

    return per_cell, fit_info

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else np.nan

def plot_spike_dashes_color_by_meta(
    df: pd.DataFrame,
    sort_by=("Input Resistance", "Motoneuron","Cell Type"),  # default order (if not using order_by_freq)
    color_by="Motoneuron",                                    # categorical for colors
    group_order: list[str] | None = None,                     # explicit order for Motoneuron
    celltype_order: list[str] | None = None,                  # explicit order for Cell Type
    rin_ascending: bool = True,                               # Rin sort direction
    show_group_bands: bool = True,                            # shade blocks by primary sort key
    band_alpha: float = 0.08,
    base_lw: float = 1.5, lw_scale: float = 0.8, dash_halfwidth: float = 0.35,
    show_means: bool = True, show_medians: bool = True,
    marker_size: float = 36.0,
    ax: plt.Axes | None = None,
    # If color_by is numeric, auto-bin it so we still get a categorical legend
    bin_numeric_color: bool = True,
    n_color_bins: int = 4,
    bin_labels: list[str] | None = None,
    # Styling for stats markers
    mean_color: str = "brown",
    median_color: str = "black",
    median_hollow: bool = True,
    # NEW: frequency-based ordering
    order_by_freq: str | None = None,     # None, "mean", or "median"
    freq_ascending: bool = True,          # direction when ordering by mean/median
):
    """
    Dash plot of spike frequencies per cell, colored by a categorical metadata field.

    NEW:
      - order_by_freq: set to "mean" or "median" to order cells by that frequency summary.
        When provided, cells are sorted by (mean/median) first, then by `sort_by` as tiebreakers.
      - freq_ascending: True for low→high; False for high→low.

    Means are plotted as filled brown circles; medians as hollow black squares (by default).
    """
    # ------------ validate & prep ------------
    need = {"cell", "spike_frequency"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    d = df.copy()
    d["spike_frequency"] = pd.to_numeric(d["spike_frequency"], errors="coerce")
    d = d.dropna(subset=["spike_frequency"])

    sort_by = list(sort_by)

    # Only require columns that are actually used
    for col in ([color_by] if color_by else []) + sort_by:
        if col not in d.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    if "Input Resistance" in d.columns:
        d["Input Resistance"] = pd.to_numeric(d["Input Resistance"], errors="coerce")

    # -------- per-cell metadata table --------
    agg_map = {}
    if "Motoneuron" in d.columns:
        agg_map["Motoneuron"] = ("Motoneuron", _mode_or_nan)
    if "Cell Type" in d.columns:
        agg_map["Cell Type"] = ("Cell Type", _mode_or_nan)
    if "Input Resistance" in d.columns:
        agg_map["Input Resistance"] = ("Input Resistance", "median")

    # include color_by if not already included
    if color_by and color_by not in agg_map and color_by in d.columns:
        if pd.api.types.is_numeric_dtype(d[color_by]):
            agg_map[color_by] = (color_by, "median")
        else:
            agg_map[color_by] = (color_by, _mode_or_nan)

    # Frequency summaries per cell for optional ordering
    freq_agg = d.groupby("cell")["spike_frequency"].agg(freq_mean="mean", freq_median="median")

    meta = d.groupby("cell").agg(**agg_map).join(freq_agg, how="left")

    # Optional explicit category orders
    if "Motoneuron" in meta.columns and group_order is not None:
        meta["Motoneuron"] = pd.Categorical(meta["Motoneuron"], categories=group_order, ordered=True)
    if "Cell Type" in meta.columns and celltype_order is not None:
        meta["Cell Type"] = pd.Categorical(meta["Cell Type"], categories=celltype_order, ordered=True)

    # Build sort keys
    if order_by_freq is not None:
        key = order_by_freq.strip().lower()
        if key not in {"mean", "median"}:
            raise ValueError("order_by_freq must be one of: None, 'mean', 'median'")
        freq_col = "freq_mean" if key == "mean" else "freq_median"
        sort_cols = [freq_col] + sort_by
        asc = [freq_ascending] + [(rin_ascending if c == "Input Resistance" else True) for c in sort_by]
    else:
        sort_cols = sort_by
        asc = [(rin_ascending if c == "Input Resistance" else True) for c in sort_by]

    meta_sorted = meta.sort_values(by=sort_cols, ascending=asc, kind="mergesort")
    cell_order = meta_sorted.index.tolist()

    # ------------ color categories ------------
    color_series = meta_sorted[color_by] if color_by else pd.Series(index=meta_sorted.index, dtype="object")
    if color_by and pd.api.types.is_numeric_dtype(color_series) and bin_numeric_color:
        labels = bin_labels if (bin_labels and len(bin_labels) == n_color_bins) else [f"Q{i+1}" for i in range(n_color_bins)]
        color_series = pd.qcut(color_series, q=n_color_bins, duplicates="drop", labels=labels)
    cats = pd.Categorical(color_series).categories.tolist() if color_by else []

    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}

    # ------------ plotting ------------
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(10, len(cell_order) * 0.5), 5), constrained_layout=True)

    xmap = {cell: i for i, cell in enumerate(cell_order, start=1)}
    mean_pts, median_pts = [], []

    for cell in cell_order:
        sub = d[d["cell"] == cell]
        if sub.empty:
            continue
        cat_val = color_series.loc[cell] if color_by else None
        col = color_map.get(cat_val, "gray") if color_by else "gray"

        counts = sub["spike_frequency"].round(10).value_counts()
        x = xmap[cell]
        for freq, cnt in counts.items():
            lw = base_lw + lw_scale * (np.sqrt(cnt) - 1.0)
            ax.hlines(y=freq, xmin=x - dash_halfwidth, xmax=x + dash_halfwidth,
                      linewidth=lw, color=col, zorder=1)

        if show_means:
            mean_pts.append((x, sub["spike_frequency"].mean()))
        if show_medians:
            median_pts.append((x, sub["spike_frequency"].median()))

    # --- stats markers ---
    if show_means and mean_pts:
        xs, ys = zip(*mean_pts)
        ax.scatter(xs, ys, s=marker_size, c=mean_color, marker="o",
                   edgecolors=mean_color, linewidths=1.0, label="Mean", zorder=6)

    if show_medians and median_pts:
        xs, ys = zip(*median_pts)
        if median_hollow:
            ax.scatter(xs, ys, s=marker_size, facecolors="none", edgecolors=median_color,
                       marker="s", linewidths=1.25, label="Median", zorder=7)
        else:
            ax.scatter(xs, ys, s=marker_size, c=median_color, marker="s",
                       edgecolors=median_color, linewidths=1.0, label="Median", zorder=7)

    # Optional bands by the primary sort key (only if categorical)
    if show_group_bands and sort_by:
        primary = sort_by[0]
        if primary in meta_sorted.columns and not pd.api.types.is_numeric_dtype(meta_sorted[primary]):
            vals = meta_sorted[primary].tolist()
            start = 0
            for i in range(1, len(vals) + 1):
                if i == len(vals) or vals[i] != vals[i-1]:
                    if (start % 2) == 0:
                        left = xmap[cell_order[start]] - 0.5
                        right = xmap[cell_order[i-1]] + 0.5
                        ax.axvspan(left, right, color="k", alpha=band_alpha, zorder=0)
                        midx = 0.5 * (left + right)
                        ax.text(midx, ax.get_ylim()[1], str(vals[start]),
                                ha="center", va="bottom", fontsize=9)
                    start = i

    # Legend: color categories + stats markers
    color_handles = [Line2D([0], [0], color=color_map[c], lw=4, label=str(c)) for c in cats]
    stats_handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor=mean_color, markeredgecolor=mean_color,
               markersize=7, label='Mean'),
        Line2D([0], [0], marker='s', linestyle='None',
               markerfacecolor='none' if median_hollow else median_color,
               markeredgecolor=median_color, markersize=7, label='Median'),
    ]
    ax.legend(handles=stats_handles + color_handles, title=(color_by or ""), loc="best", fontsize=9)

    # Clean axes
    order_desc = "" if order_by_freq is None else f" (ordered by {order_by_freq}{' ↑' if freq_ascending else ' ↓'})"
    ax.set_title(f"Spike frequencies per cell — colored by {color_by}{order_desc}")
    ax.set_ylabel("Spike frequency (Hz)")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.2)

    if created_fig is not None:
        plt.show()

import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.empty else np.nan

def order_cells(
    df: pd.DataFrame,
    by: str = "mean",                      # 'mean', 'median', or 'meta'
    group_col: str = "Motoneuron",
    celltype_col: str = "Cell Type",
    rin_col: str = "Input Resistance",
    group_order: list[str] | None = None,  # optional explicit order for groups
    celltype_order: list[str] | None = None,  # optional explicit order for cell types
    rin_ascending: bool = True
) -> list[str]:
    """
    Return cells ordered by:
      - 'mean' or 'median' spike frequency (old behavior), OR
      - 'meta': (group -> cell type -> input resistance) hierarchical sort.

    For 'meta', the per-cell metadata are:
      group      = mode of group_col within the cell
      cell type  = mode of celltype_col within the cell
      Rin        = numeric (median of rin_col) within the cell
    """
    required = {"cell", "spike_frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    d = df.copy()
    d["spike_frequency"] = pd.to_numeric(d["spike_frequency"], errors="coerce")

    if by in {"mean", "median"}:
        agg = (d.dropna(subset=["spike_frequency"])
                 .groupby("cell")["spike_frequency"]
                 .agg(mean="mean", median="median"))
        return agg[by].sort_values(kind="mergesort").index.tolist()

    if by != "meta":
        raise ValueError("by must be 'mean', 'median', or 'meta'")

    # --- metadata-driven order ---
    # build per-cell metadata table
    if group_col not in d.columns or celltype_col not in d.columns or rin_col not in d.columns:
        missing = [c for c in [group_col, celltype_col, rin_col] if c not in d.columns]
        raise ValueError(f"'meta' ordering requires columns: {missing}")

    # compute mode for categorical fields; median for Rin (ensure numeric)
    d[rin_col] = pd.to_numeric(d[rin_col], errors="coerce")

    meta = (d.groupby("cell")
              .agg(
                  **{
                      group_col: (group_col, _mode_or_nan),
                      celltype_col: (celltype_col, _mode_or_nan),
                      rin_col: (rin_col, "median"),
                  }
              ))
    meta.columns = [group_col, celltype_col, rin_col]  # flatten names if needed

    # apply optional explicit category orders
    if group_order is not None:
        meta[group_col] = pd.Categorical(meta[group_col], categories=group_order, ordered=True)
    if celltype_order is not None:
        meta[celltype_col] = pd.Categorical(meta[celltype_col], categories=celltype_order, ordered=True)

    # sort: group → cell type → Rin → (stable) cell name for tie-break
    meta_sorted = meta.sort_values(
        by=[group_col, celltype_col, rin_col, meta.index.name],
        ascending=[True, True, rin_ascending, True],
        kind="mergesort"
    )

    return meta_sorted.index.tolist()


def plot_spike_dashes(
    df: pd.DataFrame,
    cell_order: list[str] | None = None,
    order_by: str = "mean",                # now accepts 'meta'
    group_col: str = "group",
    celltype_col: str = "Cell Type",
    rin_col: str = "Input Resistance",
    group_order: list[str] | None = None,
    celltype_order: list[str] | None = None,
    rin_ascending: bool = True,
    base_lw: float = 1.5, lw_scale: float = 0.8, dash_halfwidth: float = 0.35,
    show_means: bool = True, show_medians: bool = True,
    spike_color: str = "tab:blue", mean_color: str = "tab:orange", median_color: str = "tab:green",
    marker_size: float = 40.0, ax: plt.Axes | None = None
):
    """
    Dash for each spike frequency per cell (same plot as before).
    If `cell_order` is None, cells are ordered by `order_by`:
      - 'mean' / 'median' (old behavior) or 'meta' (group → type → Rin).
    """
    # basic validation
    if {"cell", "spike_frequency"} - set(df.columns):
        raise ValueError("DataFrame must have 'cell' and 'spike_frequency' columns")
    d = df.dropna(subset=["spike_frequency"]).copy()

    if cell_order is None:
        cell_order = order_cells(
            d, by=order_by,
            group_col=group_col, celltype_col=celltype_col, rin_col=rin_col,
            group_order=group_order, celltype_order=celltype_order,
            rin_ascending=rin_ascending
        )

    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=(max(8, len(cell_order) * 0.7), 5), constrained_layout=True)

    # map cells to x positions
    xmap = {cell: i for i, cell in enumerate(cell_order, start=1)}
    means, medians, xs = [], [], []

    # Draw dashes
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

    # Overlay stats
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

# 1) Per-cell spike summary (stats only)
def summarize_spike_stats(
    all_spikes: pd.DataFrame,
    cell_col: str = "cell",
    freq_col: str = "spike_frequency",
    round_decimals: int | None = None,
) -> pd.DataFrame:
    required = {cell_col, freq_col}
    missing = required - set(all_spikes.columns)
    if missing:
        raise ValueError(f"all_spikes missing columns: {missing}")

    df = all_spikes[[cell_col, freq_col]].copy()
    df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")
    df = df.dropna(subset=[freq_col])

    stats = (
        df.groupby(cell_col)[freq_col]
          .agg(mean="mean", median="median", min="min", max="max", total_spikes="count")
          .reset_index()
          .rename(columns={cell_col: "cell"})
    )
    if round_decimals is not None:
        stats[["mean","median","min","max"]] = stats[["mean","median","min","max"]].round(round_decimals)
    return stats


# 2) Merge stats with your metadata table
def build_cell_summary_with_meta(
    all_spikes: pd.DataFrame,
    cell_meta: pd.DataFrame,
    spikes_cell_col: str = "cell",      # col in all_spikes that names the cell/sheet
    meta_key_col: str = "Cell",        # key in cell_meta that matches spikes_cell_col
    meta_cols: tuple = ("Motoneuron", "Cell Type", "Input Resistance"),
    how: str = "inner",                 # 'inner' keeps only cells found in BOTH
    round_decimals: int | None = 2,
    return_diagnostics: bool = True,
):
    # compute stats from spikes
    stats = summarize_spike_stats(all_spikes.rename(columns={spikes_cell_col: "cell"}),
                                  cell_col="cell", freq_col="spike_frequency",
                                  round_decimals=round_decimals)

    # clean meta (unique key; rename to 'cell')
    meta_clean = (
        cell_meta.dropna(subset=[meta_key_col])
                 .drop_duplicates(subset=[meta_key_col], keep="first")
                 .rename(columns={meta_key_col: "cell"})[["cell", *meta_cols]]
                 .copy()
    )
    # ensure numeric Rin
    if "Input Resistance" in meta_clean.columns:
        meta_clean["Input Resistance"] = pd.to_numeric(meta_clean["Input Resistance"], errors="coerce")

    out = stats.merge(meta_clean, on="cell", how=how)

    # reorder columns nicely
    col_order = ["cell", *meta_cols, "mean", "median", "min", "max", "total_spikes"]
    out = out[[c for c in col_order if c in out.columns]]

    if return_diagnostics:
        missing_meta = sorted(set(stats["cell"]) - set(meta_clean["cell"]))
        missing_spikes = sorted(set(meta_clean["cell"]) - set(stats["cell"]))
        return out, {"cells_missing_meta": missing_meta, "cells_missing_spikes": missing_spikes}
    return out


# 3) (Optional) Attach meta to every spike row (useful for plotting/order_by='meta')
def enrich_all_spikes_with_meta(
    all_spikes: pd.DataFrame,
    cell_meta: pd.DataFrame,
    spikes_cell_col: str = "cell",
    meta_key_col: str = "Cell",
    meta_cols: tuple = ("Motoneuron", "Cell Type", "Input Resistance"),
) -> pd.DataFrame:
    meta_small = (
        cell_meta.dropna(subset=[meta_key_col])
                 .drop_duplicates(subset=[meta_key_col], keep="first")
                 .rename(columns={meta_key_col: spikes_cell_col})[[spikes_cell_col, *meta_cols]]
    )
    out = all_spikes.merge(meta_small, on=spikes_cell_col, how="left")
    # coerce Rin to numeric if present
    if "Input Resistance" in out.columns:
        out["Input Resistance"] = pd.to_numeric(out["Input Resistance"], errors="coerce")
    return out

import pandas as pd
import numpy as np

def find_low_frequency_spikes(
    df: pd.DataFrame,
    freq_col: str = "spike_frequency",
    threshold: float = 5.0,
    inclusive: bool = False,
    dropna: bool = True,
    return_index: bool = False,
) -> pd.DataFrame | pd.Index:
    """
    Return rows (or their index) where spike frequency is below a threshold.

    Parameters
    ----------
    df : DataFrame with column `freq_col`
    freq_col : name of frequency column (default 'spike_frequency')
    threshold : cutoff in Hz (default 5.0)
    inclusive : if True, use <= threshold; else use < threshold
    dropna : if True, exclude NaN frequencies
    return_index : if True, return df.index[mask] instead of the filtered DataFrame

    Returns
    -------
    DataFrame (filtered) or Index (row indices), depending on `return_index`.
    """
    if freq_col not in df.columns:
        raise ValueError(f"Column '{freq_col}' not found in DataFrame")

    freq = pd.to_numeric(df[freq_col], errors="coerce")
    if dropna:
        valid = freq.notna()
    else:
        valid = pd.Series(True, index=df.index)

    if inclusive:
        mask = valid & (freq <= threshold)
    else:
        mask = valid & (freq < threshold)

    return df.index[mask] if return_index else df.loc[mask].copy()

def find_sheets_with_all_na_tags(
    cell_attached_dict: dict[str, pd.DataFrame],
    tags_col: str = "Tags",
    include_missing_col: bool = False,  # if True, sheets without Tags column count as "all NA"
    include_empty_sheet: bool = False,  # if True, empty sheets count as "all NA"
) -> list[str]:
    """
    Return sheet names where the Tags column is entirely NA-like.
    NA-like includes: actual NaN/pd.NA/None and strings: "<NA>", "NA", "NaN", "None", "".
    """
    na_like_strings = {"<na>", "na", "nan", "none", ""}

    def _all_na_like(s: pd.Series) -> bool:
        if s.empty:
            return include_empty_sheet
        # Normalize to actual NaN for NA-like strings
        s2 = s.copy()
        s2 = s2.map(lambda x: (
            np.nan if isinstance(x, str) and x.strip().lower() in na_like_strings
            else x
        ))
        # Now check nulls
        return s2.isna().all()

    out = []
    for name, df in (cell_attached_dict or {}).items():
        if not isinstance(df, pd.DataFrame):
            continue
        if tags_col not in df.columns:
            if include_missing_col:
                out.append(name)
            continue
        if _all_na_like(df[tags_col]):
            out.append(name)

    return sorted(out)