"""
Utility functions and shared configuration for neuroscience data analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
import os

# Default parent folder path (can be overridden)
DEFAULT_PARENT_PATH = "/Users/Haley/Desktop/"

def get_std_range(array, stdnum=8):
    """
    Returns the lower and upper bounds of a region centered at the mean
    that contains data within stdnum standard deviations.
    
    Parameters:
    -----------
    array : array-like
        Data array
    stdnum : int
        Number of standard deviations (default: 8)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    mean = np.mean(array)
    std = np.std(array)
    lower = mean - stdnum * std
    upper = mean + stdnum * std
    return lower, upper


def load_cell_metadata(parent_folder_path=DEFAULT_PARENT_PATH):
    """
    Load cell type metadata and summary statistics.
    
    Returns:
    --------
    cell_types_df : DataFrame
        Cell metadata with types, resistance, and spiking stats
    sheet_names : array
        Array of cell/sheet names
    """
    cell_types_df = pd.read_csv(
        os.path.join(parent_folder_path, 
                     "murray-neuroscience-lab/data/annotations/summary_spikes2.csv")
    )
    cell_types_df.reset_index(drop=True, inplace=True)
    sheet_names = cell_types_df["Cell"].unique()
    
    return cell_types_df, sheet_names


def load_trace_names(parent_folder_path=DEFAULT_PARENT_PATH):
    """
    Load list of all ABF trace names.
    
    Returns:
    --------
    trace_names : array
        Array of trace names
    """
    trace_df = pd.read_csv(
        os.path.join(parent_folder_path,
                     "murray-neuroscience-lab/data/annotations/all_trace_names.csv"),
        header=None
    )
    return trace_df.iloc[:, 0].to_numpy()


def concat_one_channel(abf, ch=0):
    """
    Concatenate all sweeps from one ABF channel into continuous time series.
    
    Parameters:
    -----------
    abf : pyabf.ABF
        ABF file object
    ch : int
        Channel number (default: 0)
    
    Returns:
    --------
    full_time : array
        Concatenated time array (seconds)
    full_current : array
        Concatenated current/voltage array
    """
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
        tx = abf.sweepX.copy()
        y = abf.sweepY.copy().astype(np.float32, copy=False)
        full_time_chunks.append(tx + offsets_sec[s])
        full_current_chunks.append(y)

    full_time = np.concatenate(full_time_chunks)
    full_current = np.concatenate(full_current_chunks)
    
    return full_time, full_current


def calculate_midpoints_and_frequencies(all_annotations):
    """
    Calculate midpoints between annotations and frequencies based on black dot timing.
    
    Parameters:
    -----------
    all_annotations : DataFrame
        Annotations with BlackDotTime column
    
    Returns:
    --------
    all_annotations : DataFrame
        Modified DataFrame with Midpoint and Freq_Blackdots columns
    """
    all_annotations = all_annotations.copy()
    all_annotations["Midpoint"] = np.nan
    all_annotations["Freq_Blackdots"] = np.nan
    
    for i in range(len(all_annotations)):
        # If it is the last annotation
        if i + 1 == len(all_annotations):
            if i > 0:
                half_interval = (all_annotations.loc[i, "BlackDotTime"] - 
                               all_annotations.loc[i-1, "Midpoint"])
                midpoint = all_annotations.loc[i, "BlackDotTime"] + half_interval
                all_annotations.loc[i, "Midpoint"] = midpoint
        else:
            interval = (all_annotations.loc[i+1, "BlackDotTime"] - 
                       all_annotations.loc[i, "BlackDotTime"])
            all_annotations.loc[i+1, "Freq_Blackdots"] = 1 / interval
            midpoint = all_annotations.loc[i, "BlackDotTime"] + interval / 2
            all_annotations.loc[i, "Midpoint"] = midpoint
    
    return all_annotations
