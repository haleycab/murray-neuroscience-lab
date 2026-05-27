"""
Helper functions extracted from `finalwaveform_processor copy.ipynb`.
Function bodies are preserved from notebook definitions.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
import os
from waveforms_helpers2 import *
import pickle
from center_vrb import *
from vrb_plotbouts import *
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr
from collections import Counter
from collections import defaultdict
import matplotlib.cm as cm
import math


def make_sheets_dict(sheet_names,parent_folder_path):
    sheets = {}

    for sheet in sheet_names:
        file_path = parent_folder_path+"murray-neuroscience-lab/New processed excels/"+sheet+".csv"
        df = pd.read_csv(file_path)
        df[["Trace name","Tags","Type"]] = df[["Trace name","Tags","Type"]].astype("string")
        types = cell_types_df[cell_types_df["Cell"]==sheet]
        df.loc[:,"Currents Channel"] = types.iloc[0]["currents"]
        df.loc[:,"VRB Channel"] = types.iloc[0]["vrb"]
        df.loc[:,"Median Spiking"] = types.iloc[0]["median"]
        df.loc[:,"Mean Spiking"] = types.iloc[0]["mean"]
        sheets[sheet] = df

    return sheets

def add_abfs(sheets,abfs_names,parent_folder_path_ABFS):
    for sheet in sheets.keys():
        traces = []
        df = sheets[sheet]
        traces = df["Trace name"].unique()  
        
        abfs = {}
        for trace in abfs_names:
            if trace in traces:  # only proceed if trace is in the sheet traces
                file_path2 = os.path.join(parent_folder_path_ABFS, trace+'.abf')
                if os.path.isfile(file_path2):
                    abf = pyabf.ABF(file_path2)
                    abfs[trace] = abf
                else:
                    print(f"Warning: File not found {file_path2}")

        sheets[sheet] = {
            "annotations": df,
            "abfs": abfs
        }

    return sheets

def calculate_midpoints_and_frequencies(all_annotations_in):  
    all_annotations = all_annotations_in.copy().reset_index(drop=True)
    for i in range(len(all_annotations)):
        # print(i)
            # if it is the last annotation
        if i+1 == len(all_annotations):
            # calculate interval from previous midpoint to current black dot time
            half_interval = all_annotations.loc[i,"BlackDotTime"] - all_annotations.loc[i-1,"Midpoint"]
            midpoint = all_annotations.loc[i,"BlackDotTime"] + half_interval
            all_annotations.loc[i,"Midpoint"] = midpoint
            # all_annotations.loc[i,"VRB duration"]= 
            print(half_interval)
            print("last midpoint",midpoint)
            print("last blackdot",all_annotations.loc[i,"BlackDotTime"])
            print("prev mid",all_annotations.loc[i-1,"Midpoint"])
        else:
            interval = all_annotations.loc[i+1,"BlackDotTime"] - all_annotations.loc[i,"BlackDotTime"] 
            all_annotations.loc[i+1,"Freq_Blackdots"] = 1/interval
            midpoint = all_annotations.loc[i,"BlackDotTime"] + interval/2
            all_annotations.loc[i,"Midpoint"] = midpoint
        
        if i == 0:
            all_annotations.loc[i,"Freq_Midpoints"] = np.nan
            all_annotations.loc[i,"Freq_Blackdots"] = np.nan

        else:
            interval_midpoints = all_annotations.loc[i,"Midpoint"] - all_annotations.loc[i-1,"Midpoint"] 
            all_annotations.loc[i,"Freq_Midpoints"] = 1/interval_midpoints
    return all_annotations

def calculate_midpoints_and_frequencies_avg(all_annotations_in):  
    all_annotations = all_annotations_in.copy().reset_index(drop=True)
    for i in range(len(all_annotations)):
        # --- choose center time ---
        is_start = str(all_annotations.loc[i, "AnnotationType"]).lower() == "start"
        prev_is_start = (i > 0) and (str(all_annotations.loc[i-1, "AnnotationType"]).lower() == "start")

        if is_start or prev_is_start:
            center_i = all_annotations.loc[i, "BlackDotTime"]
        else:
            # average of BlackDotTime and FirstLastMidpointTime
            center_i = np.nanmean([
                all_annotations.loc[i, "BlackDotTime"],
                all_annotations.loc[i, "FirstLastMidpointTime"]
            ])

        # --- midpoint and frequency logic (same as your original) ---
        if i+1 == len(all_annotations):
            half_interval = center_i - all_annotations.loc[i-1,"Midpoint"]
            midpoint = center_i + half_interval
            all_annotations.loc[i,"Midpoint"] = midpoint
            print(half_interval)
            print("last midpoint",midpoint)
            print("last center",center_i)
            print("prev mid",all_annotations.loc[i-1,"Midpoint"])
        else:
            # choose next center using same logic
            next_is_start = str(all_annotations.loc[i+1, "AnnotationType"]).lower() == "start"
            if is_start or next_is_start:
                center_ip1 = all_annotations.loc[i+1, "BlackDotTime"]
            else:
                center_ip1 = np.nanmean([
                    all_annotations.loc[i+1, "BlackDotTime"],
                    all_annotations.loc[i+1, "FirstLastMidpointTime"]
                ])
            interval = center_ip1 - center_i
            all_annotations.loc[i+1,"Freq_Blackdots"] = 1/interval
            midpoint = center_i + interval/2
            all_annotations.loc[i,"Midpoint"] = midpoint
        
        if i == 0:
            all_annotations.loc[i,"Freq_Midpoints"] = np.nan
            all_annotations.loc[i,"Freq_Blackdots"] = np.nan
        else:
            interval_midpoints = all_annotations.loc[i,"Midpoint"] - all_annotations.loc[i-1,"Midpoint"] 
            all_annotations.loc[i,"Freq_Midpoints"] = 1/interval_midpoints

    return all_annotations

def process_and_plot_vrb_durations(df, freq_col="Freq_Midpoints"):
    """
    Processes a dataframe to compute VRB durations and plot spread by frequency bins.

    Steps:
      1. Calls calculate_midpoints_and_frequencies(df)
      2. Computes VRB Duration = LastCrossTime - AnnotationTime
      3. Applies get_freq_bin() to create 'Freq Bin Midpoints'
      4. Plots:
         - Boxplot of VRB Duration by Frequency Bin (with counts)
         - Violin plot of VRB Duration by Frequency Bin (with counts)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns for frequency midpoints, LastCrossTime, and AnnotationTime.
    freq_col : str
        Column name in df that stores the frequency midpoints.
    """

    # --- Step 1: Compute midpoints/frequencies (user-defined helper)
    # df = calculate_midpoints_and_frequencies(df)

    # --- Step 2: Compute VRB duration
    # df["VRB Duration"] = df["LastCrossTime"] - df["AnnotationTime"]

    # --- Step 3: Bin frequencies
    df["Freq Bin Midpoints"] = df[freq_col].apply(get_freq_bin)

    # --- Step 4: Set plotting order
    freq_order = ["15–25", "25–35", "35–45", "45+"]

    # --- Step 5: Plot 1 — Boxplot with counts ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="Freq Bin Midpoints", y="VRB Duration",
                order=freq_order, palette="Blues", fliersize=0)

    counts = df["Freq Bin Midpoints"].value_counts()
    for i, freq_bin in enumerate(freq_order):
        n = counts.get(freq_bin, 0)
        plt.text(i, df["VRB Duration"].max() * 1.02, f"n = {n}",
                 ha="center", va="bottom", fontsize=10, color="gray")

    plt.title("Spread of VRB Duration by Frequency Bin (Boxplot)", fontsize=14)
    plt.xlabel("Frequency Bin")
    plt.ylabel("VRB Duration (s)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Step 6: Plot 2 — Violin plot with counts ---
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x="Freq Bin Midpoints", y="VRB Duration",
                   order=freq_order, inner="box", palette="coolwarm")

    for i, freq_bin in enumerate(freq_order):
        n = counts.get(freq_bin, 0)
        plt.text(i, df["VRB Duration"].max() * 1.02, f"n = {n}",
                 ha="center", va="bottom", fontsize=10, color="gray")

    plt.title("Distribution of VRB Duration by Frequency Bin (Violin Plot)", fontsize=14)
    plt.xlabel("Frequency Bin")
    plt.ylabel("VRB Duration (s)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    return df

def compare_frequency_measures(
    df: pd.DataFrame,
    freq_cols: list[str],
    *,
    freq_min: float = 15,
    freq_max: float = 55,
    point_size: float = 18,
    alpha: float = 0.6,
    title: str = "Pairwise comparison of frequency measures"
) -> pd.DataFrame:
    """
    Plot three pairwise scatter plots comparing the given frequency columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    freq_cols : list[str]
        Exactly 3 column names of frequency measures to compare.
    freq_min, freq_max : float
        Keep only rows where BOTH measures in a pair are within [freq_min, freq_max].
    point_size : float
        Scatter point size.
    alpha : float
        Scatter point transparency.
    title : str
        Figure title.

    Returns
    -------
    stats_df : pd.DataFrame
        Summary stats per pair: Pearson r, p-value, MAE (Hz), bias (Y - X), and n.
    """
    assert len(freq_cols) == 3, "Please provide exactly 3 frequency column names."

    pairs = list(combinations(range(3), 2))  # (0,1), (0,2), (1,2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    rows = []
    for ax, (i, j) in zip(axes, pairs):
        xcol, ycol = freq_cols[i], freq_cols[j]

        # Clean + bound filter for this pair
        sub = df[[xcol, ycol]].dropna()
        sub = sub[(sub[xcol] >= freq_min) & (sub[xcol] <= freq_max) &
                  (sub[ycol] >= freq_min) & (sub[ycol] <= freq_max)]

        if sub.empty:
            ax.set_title(f"{xcol} vs {ycol}\n(no data in {freq_min}–{freq_max} Hz)")
            ax.set_xlabel(xcol); ax.set_ylabel(ycol)
            continue

        x = sub[xcol].to_numpy()
        y = sub[ycol].to_numpy()
        n = len(sub)

        # Stats
        r, p = pearsonr(x, y)
        mae = float(np.mean(np.abs(x - y)))
        bias = float(np.mean(y - x))  # positive = y tends higher than x

        # Plot scatter
        ax.scatter(x, y, s=point_size, alpha=alpha)

        # Identity line y = x
        vmin = float(min(x.min(), y.min()))
        vmax = float(max(x.max(), y.max()))
        pad = 0.05 * max(vmax - vmin, 1.0)
        lo, hi = vmin - pad, vmax + pad
        ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, linestyle="--", label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel(f"{xcol} (Hz)")
        ax.set_ylabel(f"{ycol} (Hz)")
        ax.set_title(f"{xcol} vs {ycol}\nr = {r:.2f}, MAE = {mae:.2f} Hz, n = {n}")
        ax.legend(loc="lower right")

        rows.append({
            "x": xcol, "y": ycol,
            "r": r, "p": p, "MAE_Hz": mae, "bias_(y_minus_x)_Hz": bias, "n": n
        })

    fig.suptitle(f"{title}\n(filtered {freq_min} ≤ freq ≤ {freq_max} Hz)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    return pd.DataFrame(rows, columns=["x","y","r","p","MAE_Hz","bias_(y_minus_x)_Hz","n"])

def compare_vrb_duration_vs_frequency(
    df,
    freq_cols,
    duration_col="VRB Duration",
    title="VRB Duration vs. Frequency Comparison",
    freq_min=15,
    freq_max=55
):
    """
    Compare how VRB Duration relates to different frequency estimates (15–55 Hz only).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing duration and frequency columns.
    freq_cols : list of str
        List of 3 column names representing different frequency calculations.
    duration_col : str
        Column name for VRB duration.
    title : str
        Title for the plots.
    freq_min, freq_max : float
        Frequency range to include.
    """

    assert len(freq_cols) == 3, "Please provide exactly 3 frequency column names."

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    corrs = {}

    for i, freq_col in enumerate(freq_cols):
        # Drop NaNs and filter by frequency range
        sub = df[[freq_col, duration_col]].dropna()
        sub = sub[(sub[freq_col] >= freq_min) & (sub[freq_col] <= freq_max)]

        if sub.empty:
            print(f"⚠️ Skipped {freq_col}: no values within {freq_min}–{freq_max} Hz.")
            continue

        # Calculate correlation
        r, p = pearsonr(sub[freq_col], sub[duration_col])
        corrs[freq_col] = (r, p)

        # Scatter + regression line
        sns.regplot(
            data=sub, x=freq_col, y=duration_col,
            ax=axes[i],
            scatter_kws={"alpha": 0.6},
            line_kws={"color": "red"}
        )

        axes[i].set_title(f"{freq_col}\nPearson r = {r:.2f}, p = {p:.3e}")
        axes[i].set_xlabel("Frequency (Hz)")
        if i == 0:
            axes[i].set_ylabel("VRB Duration (s)")
        else:
            axes[i].set_ylabel("")

    fig.suptitle(f"{title}\n(filtered {freq_min} ≤ freq ≤ {freq_max} Hz)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return pd.DataFrame.from_dict(
        {col: {"r": r, "p": p} for col, (r, p) in corrs.items()},
        orient="index"
    )

def compare_frequency_measures_with_bins(
    df: pd.DataFrame,
    freq_cols: list[str],
    *,
    freq_min: float = 15,
    freq_max: float = 55,
    freq_bins: list[float] = [15, 25, 35, 45, 55],
    point_size: float = 10,   # smaller points
    alpha: float = 0.6,
    title: str = "Pairwise Comparison of Frequency Measures with Bin Gridlines"
) -> pd.DataFrame:
    """
    Plot three pairwise scatter plots comparing the given frequency columns,
    with gridlines marking frequency bins and smaller scatter dots.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    freq_cols : list[str]
        Exactly 3 column names of frequency measures to compare.
    freq_min, freq_max : float
        Keep only rows where both measures are within [freq_min, freq_max].
    freq_bins : list[float]
        Bin edges for gridlines on both axes.
    point_size : float
        Scatter point size.
    alpha : float
        Scatter transparency.
    title : str
        Figure title.

    Returns
    -------
    stats_df : pd.DataFrame
        Summary stats per pair (Pearson r, p-value, MAE, bias, n).
    """
    assert len(freq_cols) == 3, "Please provide exactly 3 frequency column names."

    pairs = list(combinations(range(3), 2))  # (0,1), (0,2), (1,2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    rows = []
    for ax, (i, j) in zip(axes, pairs):
        xcol, ycol = freq_cols[i], freq_cols[j]

        # Filter and clean data
        sub = df[[xcol, ycol]].dropna()
        sub = sub[(sub[xcol] >= freq_min) & (sub[xcol] <= freq_max) &
                  (sub[ycol] >= freq_min) & (sub[ycol] <= freq_max)]

        if sub.empty:
            ax.set_title(f"{xcol} vs {ycol}\n(no data in {freq_min}–{freq_max} Hz)")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            continue

        x = sub[xcol].to_numpy()
        y = sub[ycol].to_numpy()
        n = len(sub)

        # Compute stats
        r, p = pearsonr(x, y)
        mae = float(np.mean(np.abs(x - y)))
        bias = float(np.mean(y - x))

        # Scatter plot
        ax.scatter(x, y, s=point_size, alpha=alpha, color="tab:blue")

        # Axes setup
        ax.set_xlim(freq_min, freq_max)
        ax.set_ylim(freq_min, freq_max)
        ax.set_aspect('equal', adjustable='box')

        # Gridlines at frequency bin boundaries
        for b in freq_bins:
            ax.axhline(y=b, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.axvline(x=b, color="gray", lw=0.8, ls="--", alpha=0.5)

        ax.set_xticks(freq_bins)
        ax.set_yticks(freq_bins)

        ax.set_xlabel(f"{xcol} (Hz)")
        ax.set_ylabel(f"{ycol} (Hz)")
        ax.set_title(f"{xcol} vs {ycol}\nr = {r:.2f}, MAE = {mae:.2f} Hz, n = {n}")

        rows.append({
            "x": xcol, "y": ycol,
            "r": r, "p": p, "MAE_Hz": mae, "bias_(y_minus_x)_Hz": bias, "n": n
        })

    fig.suptitle(f"{title}\n(filtered {freq_min} ≤ freq ≤ {freq_max} Hz)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    return pd.DataFrame(rows, columns=["x","y","r","p","MAE_Hz","bias_(y_minus_x)_Hz","n"])

def get_freq_bin(freq):
    if 15 <= freq < 25:
        return "15–25"
    elif 25 <= freq < 35:
        return "25–35"
    elif 35 <= freq < 45:
        return "35–45"
    elif freq >= 45:
        return "45+"
    else:
        return None

def new_make_waveforms(abf, df):
    '''
    Function that takes an abf file and a df of the annotations
    Returns a dictionary with waveforms labeled by their frequency, cell type, signal type, and rin
    '''

    # extract channel data
    # if len(df["Currents Channel"].unique()) != 1:
    #     print("Warning: Multiple Currents Channels found:", all_annotations["Currents Channel"].unique())
    # if len(df["VRB Channel"].unique()) != 1:
    #     print("Warning: Multiple Currents Channels found:", all_annotations["VRB Channel"].unique())
    # vrb_ch = df.loc[0,"VRB Channel"]
    currents_ch = df.loc[0,"Currents Channel"]
    print("currents channel",currents_ch)
    full_time, full_current = concat_one_channel(abf,ch = currents_ch)
    abf_df = pd.DataFrame({"Time": full_time, "Current": full_current})


    waveforms = {}
    for i in range(len(df) - 1):
        t_0 = df.iloc[i]["Midpoint"]
        t_f = df.iloc[i + 1]["Midpoint"]

        abf_waveform = abf_df[(abf_df["Time"] >= t_0) & (abf_df["Time"] <= t_f)].copy()
        # previous error
        if abf_waveform.empty:
            raise ValueError(
                f"abf_waveform is empty for index {i}, t_0={t_0}, t_f={t_f}, Trace name={df.iloc[i]['Trace name']}"
            )

        # Add phase (-0.5 to 0.5 across the segment)
        abf_waveform["Phase"] = ((abf_waveform["Time"] - t_0) / (t_f - t_0))-0.5

        # Normalize Current
        y_max = abf_waveform["Current"].max()
        y_min = abf_waveform["Current"].min()
        abf_waveform["Normalized Current"] = (abf_waveform["Current"] - y_min) / (y_max - y_min)

        # Dict keys
        freq = 1 / (t_f - t_0)
        signal_type = df.iloc[i]["Type"]
        median = df.iloc[i]["Median Spiking"]
        mean = df.iloc[i]["Mean Spiking"]
        cell_type = df.iloc[i]["Cell Type"]
        rin = df.iloc[i]["Input Resistance"]
        group = df.iloc[i]["Group"]
        cell = df.iloc[i]["Cell"]
        key = (freq, signal_type, median, mean, cell_type, rin, group, cell)

        waveforms[key] = abf_waveform

    return waveforms

def get_freq_bin(freq):
    if 15 <= freq < 25:
        return "15–25"
    elif 25 <= freq < 35:
        return "25–35"
    elif 35 <= freq < 45:
        return "35–45"
    elif freq >= 45:
        return "45+"
    else:
        return None

def get_speed_bin(value, cutoff=35):
    return "fast" if value > cutoff else "slow"

def normalize_signal_type(s: str) -> str:
    s_low = s.strip().lower()
    if s_low.startswith("excitatory"):
        return "Excitatory"
    if s_low.startswith("inhibitory"):
        return "Inhibitory"
    if "cell-attached" in s_low:
        return None
    return s  

def bin_wave_100(onewave):
    """
    Bin 'Phase' into 100 equal-width bins between -0.5 and 0.5,
    compute the mean of 'Current' and 'Normalized Current' per bin,
    and return a DataFrame with bin intervals, averages, and numeric bin centers.
    """
    # 100 bins over [-0.5, 0.5]
    bins = np.linspace(-0.5, 0.5, 101, endpoint=True)

    # Assign each Phase value to a bin
    onewave = onewave.copy()
    onewave['Phase Bin'] = pd.cut(onewave['Phase'], bins=bins, include_lowest=True)

    # Compute mean Current and Normalized Current for each bin
    binned_avg = (
        onewave.groupby('Phase Bin', observed=True)[['Current', 'Normalized Current']]
        .mean()
        .reset_index()
    )

    # Add numeric bin centers
    binned_avg['Phase'] = binned_avg['Phase Bin'].apply(lambda x: x.mid)

    return binned_avg

def average_waveforms_for_key_median(counter_key):
    """
    Takes a counter key like ('25–35', 'Excitatory', 'fast')
    Returns a DataFrame of the averaged waveform over 'Phase Bin' for that key.
    """
    # unpack new 3-part key
    freq_bin, norm_signal_type, fast_slow = counter_key

    dfs = []  # collect matching waveform DataFrames

    # binned_waveforms keys assumed to be (freq, signal_type, cell_type, fast_slow)
    for key in binned_waveforms.keys():
        freq, raw_signal_type, median, mean = key

        # normalize / filter signal type
        norm_st = normalize_signal_type(raw_signal_type)
        if norm_st is None:         # skip cell-attached
            continue

        # match against the new grouped key
        if get_freq_bin(freq) == freq_bin and norm_st == norm_signal_type and get_speed_bin(median, cutoff=35) == fast_slow:
            dfs.append(binned_waveforms[key].copy())

    if not dfs:
        print(f"No matching waveforms found for {counter_key}")
        return None

    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby('Phase', observed=False)
    averaged = grouped.mean(numeric_only=True).reset_index()
    sem = grouped.sem(numeric_only=True).reset_index()
    return averaged, sem

def average_waveforms_for_key_median(counter_key):
    """
    counter_key: ('25–35', 'Excitatory', 'fast')
    returns: (averaged_df, sem_df) or (None, None) if no matches
    """
    freq_bin, norm_signal_type, fast_slow = counter_key

    dfs = []
    for key, df in binned_waveforms.items():
        freq, raw_signal_type, median_spiking, mean_spiking = key

        norm_st = normalize_signal_type(raw_signal_type)
        if norm_st is None:
            continue

        # match bins and speed derived from the key's numeric median
        if get_freq_bin(freq) == freq_bin and norm_st == norm_signal_type and get_speed_bin(median_spiking, cutoff=35) == fast_slow:
            # REQUIRE: df has numeric 'Phase' (centers). If not, compute it once when you build df.
            if "Phase" not in df.columns and "Phase Bin" in df.columns:
                df = df.copy()
                df["Phase"] = df["Phase Bin"].apply(lambda x: x.mid if pd.notna(x) else np.nan)
            dfs.append(df.copy())

    if not dfs:
        return None, None

    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby('Phase', observed=False)
    averaged = grouped.mean(numeric_only=True).reset_index()
    sem = grouped.sem(numeric_only=True).reset_index()
    return averaged, sem

def average_waveforms_for_key_invert(counter_key):
    """
    Like average_waveforms_for_key, but optionally inverts excitatory signals.
    
    Parameters
    ----------
    counter_key : tuple
        (freq_bin, signal_type, fast_slow)
    invert_excitatory : bool
        If True, flip the sign of excitatory signals.
    value_col : str
        Column to invert (e.g., 'Normalized Current' or 'Current').
    
    Returns
    -------
    DataFrame or None
        Averaged waveform with optional inversion.
    """
    invert_excitatory=True 
    value_col="Normalized Current"
    averaged, sem = average_waveforms_for_key_median(counter_key)
    if averaged is None:
        return None
    
    freq_bin, signal_type, fast_slow = counter_key
    if invert_excitatory and signal_type == "Excitatory":
        if value_col in averaged.columns:
            averaged[value_col] = 1-averaged[value_col]
    return averaged,sem

def plot_e_i_pseudo(
    average_func,
    freq_bins=("15–25", "25–35", "35–45", "45+"),
    speed_categories=("fast", "slow"),
    title="Averaged E/I & Pseudo (rectified E−I) by Swimming Frequency",
    show=True
):
    """
    Plot Excitatory, Inhibitory, and Pseudo (rectified E−I) currents using
    pre-averaged data returned by `average_func(counter_key)`.

    Colors:
      - Excitatory: tab:blue
      - Inhibitory: tab:orange
      - Pseudo: tab:gray
    """
    import matplotlib.pyplot as plt
    import numpy as np

    signal_type_colors = {
        "Excitatory": "tab:blue",
        "Inhibitory": "tab:orange",
        "Pseudo": "tab:gray"
    }

    fig, axes = plt.subplots(len(speed_categories), len(freq_bins),
                             figsize=(20, 12), sharex=True, sharey=True)

    used_labels = set()

    for i, speed in enumerate(speed_categories):
        for j, freq_bin in enumerate(freq_bins):
            ax = axes[i, j] if len(speed_categories) > 1 else axes[j]

            # --- get averaged E and I for this (speed, freq_bin) ---
            key_E = (freq_bin, "Excitatory", speed)
            key_I = (freq_bin, "Inhibitory", speed)

            avg_E, sem_E = average_func(key_E)
            avg_I, sem_I = average_func(key_I)

            if avg_E is None or avg_I is None:
                if i == 0:
                    ax.set_title(f"{freq_bin} Hz")
                if j == 0:
                    ax.set_ylabel(f"{speed}\nNormalized Current")
                if i == len(speed_categories) - 1:
                    ax.set_xlabel("Phase")
                continue

            # Align phase axes if needed
            phase_E = avg_E["Phase"].to_numpy()
            E = avg_E["Normalized Current"].to_numpy()
            E_sem = sem_E["Normalized Current"].to_numpy()

            phase_I_raw = avg_I["Phase"].to_numpy()
            I_raw = avg_I["Normalized Current"].to_numpy()
            I_sem_raw = sem_I["Normalized Current"].to_numpy()

            if (len(phase_E) != len(phase_I_raw)) or (not np.allclose(phase_E, phase_I_raw)):
                I = np.interp(phase_E, phase_I_raw, I_raw)
                I_sem = np.interp(phase_E, phase_I_raw, I_sem_raw)
                phase = phase_E
            else:
                phase = phase_I_raw
                I = I_raw
                I_sem = I_sem_raw

            # --- plot Excitatory and Inhibitory currents ---
            ax.plot(phase, E, label="Excitatory" if "Excitatory" not in used_labels else None,
                    color=signal_type_colors["Excitatory"], ls=':')
            ax.fill_between(phase, E - E_sem, E + E_sem,
                            color=signal_type_colors["Excitatory"], alpha=0.3)

            ax.plot(phase, I, label="Inhibitory" if "Inhibitory" not in used_labels else None,
                    color=signal_type_colors["Inhibitory"], ls=':')
            ax.fill_between(phase, I - I_sem, I + I_sem,
                            color=signal_type_colors["Inhibitory"], alpha=0.3)
            used_labels.update(["Excitatory", "Inhibitory"])

            # --- compute and plot Pseudo = rectified (E - I) ---
            pseudo = np.maximum(E - I, 0.0)
            pseudo_sem = np.sqrt(E_sem**2 + I_sem**2)  # conservative propagation
            pseudo_upper = np.maximum(pseudo + pseudo_sem, 0.0)
            pseudo_lower = np.maximum(pseudo - pseudo_sem, 0.0)

            ax.plot(phase, pseudo, label="Pseudo (rectified E−I)" if "Pseudo" not in used_labels else None,
                    color=signal_type_colors["Pseudo"], lw=2)
            ax.fill_between(phase, pseudo_lower, pseudo_upper,
                            color=signal_type_colors["Pseudo"], alpha=0.2)
            used_labels.add("Pseudo")

            # --- subplot labels ---
            if i == 0:
                ax.set_title(f"{freq_bin} Hz")
            if j == 0:
                ax.set_ylabel(f"{speed}\nNormalized Current")
            if i == len(speed_categories) - 1:
                ax.set_xlabel("Phase")
            ax.set_xlim(-0.5, 0.5)

    # Legend
    handles, labels = [], []
    for key in ["Excitatory", "Inhibitory", "Pseudo"]:
        handles.append(plt.Line2D([0], [0], color=signal_type_colors[key], lw=2))
        labels.append(key)
    fig.legend(handles, labels, loc='upper right')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    if show:
        plt.show()

def plot_e_i_pseudo(
    average_func,
    key_counter=None,
    freq_bins=("15–25", "25–35", "35–45", "45+"),
    speed_categories=("fast", "slow"),
    title="Averaged E/I & Pseudo (rectified E−I) by Swimming Frequency",
    show=True
):
    """
    Plot Excitatory, Inhibitory, and Pseudo (rectified E−I) currents using
    pre-averaged data returned by `average_func(counter_key)`.

    Colors:
      - Excitatory: tab:blue
      - Inhibitory: tab:orange
      - Pseudo: tab:gray

    Parameters
    ----------
    average_func : callable
        Should accept a key tuple (freq_bin, signal_type, speed) and return:
        (averaged_df, sem_df) with columns ['Phase', 'Normalized Current'].

    key_counter : dict, optional
        Dictionary or defaultdict mapping (freq_bin, signal_type, speed) → count.
        If provided, each subplot will display 'n = count' in the corner.

    freq_bins : iterable of str
        Frequency-bin labels to iterate over (columns).

    speed_categories : iterable of str
        Swimming speeds to iterate over (rows).

    title : str
        Figure title.

    show : bool
        If True, calls plt.show() at the end.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    signal_type_colors = {
        "Excitatory": "tab:blue",
        "Inhibitory": "tab:orange",
        "Pseudo": "tab:gray"
    }

    fig, axes = plt.subplots(len(speed_categories), len(freq_bins),
                             figsize=(20, 12), sharex=True, sharey=True)

    used_labels = set()

    for i, speed in enumerate(speed_categories):
        for j, freq_bin in enumerate(freq_bins):
            ax = axes[i, j] if len(speed_categories) > 1 else axes[j]

            # --- get averaged E and I for this (speed, freq_bin) ---
            key_E = (freq_bin, "Excitatory", speed)
            key_I = (freq_bin, "Inhibitory", speed)

            avg_E, sem_E = average_func(key_E)
            avg_I, sem_I = average_func(key_I)

            if avg_E is None or avg_I is None:
                if i == 0:
                    ax.set_title(f"{freq_bin} Hz")
                if j == 0:
                    ax.set_ylabel(f"{speed}\nNormalized Current")
                if i == len(speed_categories) - 1:
                    ax.set_xlabel("Phase")
                continue

            # Align phase axes if needed
            phase_E = avg_E["Phase"].to_numpy()
            E = avg_E["Normalized Current"].to_numpy()
            E_sem = sem_E["Normalized Current"].to_numpy()

            phase_I_raw = avg_I["Phase"].to_numpy()
            I_raw = avg_I["Normalized Current"].to_numpy()
            I_sem_raw = sem_I["Normalized Current"].to_numpy()

            if (len(phase_E) != len(phase_I_raw)) or (not np.allclose(phase_E, phase_I_raw)):
                I = np.interp(phase_E, phase_I_raw, I_raw)
                I_sem = np.interp(phase_E, phase_I_raw, I_sem_raw)
                phase = phase_E
            else:
                phase = phase_I_raw
                I = I_raw
                I_sem = I_sem_raw

            # --- plot Excitatory and Inhibitory currents ---
            ax.plot(phase, E, label="Excitatory" if "Excitatory" not in used_labels else None,
                    color=signal_type_colors["Excitatory"], ls=':')
            ax.fill_between(phase, E - E_sem, E + E_sem,
                            color=signal_type_colors["Excitatory"], alpha=0.3)

            ax.plot(phase, I, label="Inhibitory" if "Inhibitory" not in used_labels else None,
                    color=signal_type_colors["Inhibitory"], ls=':')
            ax.fill_between(phase, I - I_sem, I + I_sem,
                            color=signal_type_colors["Inhibitory"], alpha=0.3)
            used_labels.update(["Excitatory", "Inhibitory"])

            # --- compute and plot Pseudo = rectified (E - I) ---
            pseudo = np.maximum(E - I, 0.0)
            pseudo_sem = np.sqrt(E_sem**2 + I_sem**2)
            pseudo_upper = np.maximum(pseudo + pseudo_sem, 0.0)
            pseudo_lower = np.maximum(pseudo - pseudo_sem, 0.0)

            ax.plot(phase, pseudo, label="Pseudo (rectified E−I)" if "Pseudo" not in used_labels else None,
                    color=signal_type_colors["Pseudo"], lw=2)
            ax.fill_between(phase, pseudo_lower, pseudo_upper,
                            color=signal_type_colors["Pseudo"], alpha=0.2)
            used_labels.add("Pseudo")

            # --- subplot labels ---
            if i == 0:
                ax.set_title(f"{freq_bin} Hz")
            if j == 0:
                ax.set_ylabel(f"{speed}\nNormalized Current")
            if i == len(speed_categories) - 1:
                ax.set_xlabel("Phase")
            ax.set_xlim(-0.5, 0.5)

            # --- add 'n = count' annotation if available ---
            if key_counter is not None:
                n_E = key_counter.get(key_E, 0)
                n_I = key_counter.get(key_I, 0)
                n_total = n_E + n_I
                ax.text(0.85, 0.95, f"n = {n_E}", fontsize=20, color='blue',
                        ha='right', va='top', transform=ax.transAxes)
                ax.text(0.35, 0.95, f"n = {n_I}", fontsize=20, color='orange',
                        ha='right', va='top', transform=ax.transAxes)

    # Legend
    handles, labels = [], []
    for key in ["Excitatory", "Inhibitory", "Pseudo"]:
        handles.append(plt.Line2D([0], [0], color=signal_type_colors[key], lw=2))
        labels.append(key)
    fig.legend(handles, labels, loc='upper right')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    if show:
        plt.show()

def plot_e_i_pseudo(
    average_func,
    freq_bins=("15–25", "25–35", "35–45", "45+"),
    speed_categories=("fast", "slow"),
    title="Averaged E/I & Pseudo (rectified E−I) by Swimming Frequency",
    show=True
):
    """
    Plot Excitatory, Inhibitory, and Pseudo (rectified E−I) currents using
    pre-averaged data returned by `average_func(counter_key)`.

    Parameters
    ----------
    average_func : callable
        Should accept a key tuple (freq_bin, signal_type, speed) and return:
        (averaged_df, sem_df) where each has columns ['Phase', 'Normalized Current'].
        Example: average_waveforms_for_key_invert

    freq_bins : iterable of str
        Frequency-bin labels to iterate over (columns).

    speed_categories : iterable of str
        Swimming speeds to iterate over (rows).

    title : str
        Figure title.

    show : bool
        If True, calls plt.show() at the end.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # Colors: reuse your mapping for E/I and add a third for Pseudo
    signal_types = ["Inhibitory", "Excitatory"]
    cmap = cm.get_cmap('tab10', len(signal_types) + 1)
    signal_type_colors = {stype: cmap(i) for i, stype in enumerate(sorted(signal_types))}
    pseudo_color = cmap(len(signal_types))  # third distinct color for pseudo

    fig, axes = plt.subplots(len(speed_categories), len(freq_bins),
                             figsize=(20, 12), sharex=True, sharey=True)

    used_labels = set()

    for i, speed in enumerate(speed_categories):
        for j, freq_bin in enumerate(freq_bins):
            ax = axes[i, j] if len(speed_categories) > 1 else axes[j]

            # --- get averaged E and I for this (speed, freq_bin) ---
            key_E = (freq_bin, "Excitatory", speed)
            key_I = (freq_bin, "Inhibitory", speed)

            avg_E, sem_E = average_func(key_E)
            avg_I, sem_I = average_func(key_I)

            # skip if missing
            if avg_E is None or avg_I is None:
                if i == 0:
                    ax.set_title(f"{freq_bin} Hz")
                if j == 0:
                    ax.set_ylabel(f"{speed}\nNormalized Current")
                if i == len(speed_categories) - 1:
                    ax.set_xlabel("Phase")
                continue

            # Ensure aligned Phase grids (interpolate I onto E if needed)
            phase_E = avg_E["Phase"].to_numpy()
            E = avg_E["Normalized Current"].to_numpy()
            E_sem = sem_E["Normalized Current"].to_numpy()

            phase_I_raw = avg_I["Phase"].to_numpy()
            I_raw = avg_I["Normalized Current"].to_numpy()
            I_sem_raw = sem_I["Normalized Current"].to_numpy()

            if (len(phase_E) != len(phase_I_raw)) or (not np.allclose(phase_E, phase_I_raw)):
                # Interpolate I and its SEM onto E's phase grid
                I = np.interp(phase_E, phase_I_raw, I_raw)
                I_sem = np.interp(phase_E, phase_I_raw, I_sem_raw)
                phase = phase_E
            else:
                phase = phase_I_raw
                I = I_raw
                I_sem = I_sem_raw

            # --- plot E and I with SEM ---
            for stype, y, y_sem in [
                ("Excitatory", E, E_sem),
                ("Inhibitory", I, I_sem)
            ]:
                label = stype if stype not in used_labels else None
                ax.plot(phase, y, label=label, color=signal_type_colors[stype])
                ax.fill_between(phase, y - y_sem, y + y_sem,
                                color=signal_type_colors[stype], alpha=0.3)
                used_labels.add(stype)

            # --- compute and plot Pseudo = rectified (E - I) ---
            # mean:
            pseudo = np.maximum(E - I, 0.0)

            # conservative envelope for pseudo SEM:
            # Use upper = max((E+E_sem)-(I-I_sem), 0), lower = max((E-E_sem)-(I+I_sem), 0)
            pseudo_upper = np.maximum((E + E_sem) - (I - I_sem), 0.0)
            pseudo_lower = np.maximum((E - E_sem) - (I + I_sem), 0.0)

            # Plot pseudo
            label = "Pseudo (rectified E−I)" if "Pseudo (rectified E−I)" not in used_labels else None
            ax.plot(phase, pseudo, label=label, color=pseudo_color)
            ax.fill_between(phase, pseudo_lower, pseudo_upper, color=pseudo_color, alpha=0.2)
            used_labels.add("Pseudo (rectified E−I)")

            # Cosmetics
            if i == 0:
                ax.set_title(f"{freq_bin} Hz")
            if j == 0:
                ax.set_ylabel(f"{speed}\nNormalized Current")
            if i == len(speed_categories) - 1:
                ax.set_xlabel("Phase")

            ax.set_xlim(-0.5, 0.5)

    # Legend (one per figure, top-right)
    handles, labels = [], []
    for stype in ["Excitatory", "Inhibitory"]:
        handles.append(plt.Line2D([0], [0], color=signal_type_colors[stype]))
        labels.append(stype)
    handles.append(plt.Line2D([0], [0], color=pseudo_color))
    labels.append("Pseudo (rectified E−I)")

    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    if show:
        plt.show()

def average_waveforms_for_key_invert(counter_key):
    """
    Like average_waveforms_for_key, but optionally inverts excitatory signals.
    
    Parameters
    ----------
    counter_key : tuple
        (freq_bin, signal_type, fast_slow)
    invert_excitatory : bool
        If True, flip the sign of excitatory signals.
    value_col : str
        Column to invert (e.g., 'Normalized Current' or 'Current').
    
    Returns
    -------
    DataFrame or None
        Averaged waveform with optional inversion.
    """
    invert_excitatory=True 
    value_col="Normalized Current"
    averaged = average_waveforms_for_key_median(counter_key)
    if averaged is None:
        return None
    
    freq_bin, signal_type, fast_slow = counter_key
    if invert_excitatory and signal_type == "Excitatory":
        if value_col in averaged.columns:
            averaged[value_col] = 1-averaged[value_col]
    return averaged

def plot_overlay_by_bin_speed(
    key_counts,
    value_col="Normalized Current",   # or "Current"
    invert_excitatory=False,          # flip Excitatory sign if True
    ncols=3,
    linewidth=1.8,
    sharey=True,
    figsize_per_row=3.4,
    title_fontsize=10,
    suptitle=None
):
    """
    Create one subplot per (freq_bin, speed) and overlay Excitatory vs Inhibitory in different colors.

    Requirements:
      - average_waveforms_for_key(counter_key) is defined.
      - If invert_excitatory=True, average_waveforms_for_key_invert(counter_key) is used for Excitatory.
      - key_counts is a Counter keyed by (freq_bin, signal_type, speed) -> count.
    """
    # Sort order for panels
    freq_order = ["<15", "15–25", "25–35", "35–45", "45+"]
    speed_order = ["slow", "fast"]

    # What groups (freq_bin, speed) exist? (require at least one of E/I present)
    groups = set()
    for (fb, st, sp) in key_counts.keys():
        if st in ("Excitatory", "Inhibitory"):
            groups.add((fb, sp))

    if not groups:
        print("No Excitatory/Inhibitory keys found to plot.")
        return

    def sort_group(g):
        fb, sp = g
        return (
            freq_order.index(fb) if fb in freq_order else len(freq_order),
            speed_order.index(sp) if sp in speed_order else len(speed_order),
        )

    groups = sorted(groups, key=sort_group)

    # Helper to fetch averaged df (optionally inverting excitatory)
    def _avg_df(counter_key):
        fb, st, sp = counter_key
        if invert_excitatory and st == "Excitatory":
            return average_waveforms_for_key_invert(counter_key, invert_excitatory=True, value_col=value_col)
        return average_waveforms_for_key_median(counter_key)

    # Preload data & compute global ylim if sharey
    per_group = []
    ymins, ymaxs = [], []
    for fb, sp in groups:
        key_e = (fb, "Excitatory", sp)
        key_i = (fb, "Inhibitory", sp)

        df_e = _avg_df(key_e) if key_e in key_counts else None
        df_i = _avg_df(key_i) if key_i in key_counts else None

        per_group.append(((fb, sp), df_e, df_i))

        for df in (df_e, df_i):
            if df is not None and value_col in df:
                vals = df[value_col].to_numpy()
                if np.isfinite(vals).any():
                    ymins.append(np.nanmin(vals))
                    ymaxs.append(np.nanmax(vals))

    global_ylim = None
    if sharey and ymins and ymaxs:
        pad = 0.05 * (np.nanmax(ymaxs) - np.nanmin(ymins) + 1e-12)
        global_ylim = (np.nanmin(ymins) - pad, np.nanmax(ymaxs) + pad)

    # Layout
    n = len(per_group)
    nrows = math.ceil(n / ncols)
    figsize = (ncols * figsize_per_row, nrows * figsize_per_row)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    # Plot each panel
    for ax, ((fb, sp), df_e, df_i) in zip(axes, per_group):
        # Counts for legend labels
        n_e = key_counts.get((fb, "Excitatory", sp), 0)
        n_i = key_counts.get((fb, "Inhibitory", sp), 0)

        plotted_any = False

        if df_e is not None and not df_e.empty and "Phase" in df_e and value_col in df_e:
            ax.plot(df_e["Phase"], df_e[value_col], linewidth=linewidth, label=f"Excitatory (n={n_e})")
            plotted_any = True

        if df_i is not None and not df_i.empty and "Phase" in df_i and value_col in df_i:
            ax.plot(df_i["Phase"], df_i[value_col], linewidth=linewidth, label=f"Inhibitory (n={n_i})")
            plotted_any = True

        if global_ylim is not None:
            ax.set_ylim(global_ylim)

        ax.axhline(0, lw=0.8, alpha=0.6)
        ax.set_title(f"{fb} | {sp}", fontsize=title_fontsize)
        ax.set_xlabel("Phase")
        ax.set_ylabel(value_col)
        if plotted_any:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()

    # Hide any extra axes
    for j in range(len(per_group), len(axes)):
        axes[j].set_axis_off()

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_fontsize + 2, y=0.98)

    fig.tight_layout()
    plt.show()
