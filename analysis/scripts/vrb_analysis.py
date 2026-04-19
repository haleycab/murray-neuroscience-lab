"""
Ventral root burst (VRB) analysis and bout detection.

This module handles:
- Detecting VRB bursts from electrophysiology data
- Identifying bout start/end times
- Calculating burst frequencies and timing
- Plotting motor neuron activity and VRB traces
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
try:
    from .utils import get_std_range
except ImportError:
    from utils import get_std_range


def extract_bout_windows(annotations, seconds_col="Seconds", tags_col="Tags",
                         start_kw="bout start", end_kw="bout end", pad_s=0.010):
    """
    Extract bout time windows from annotations.
    
    Parameters:
    -----------
    annotations : DataFrame
        Annotations with timing and tags
    seconds_col : str
        Column name for timing in seconds
    tags_col : str
        Column name for tags/labels
    start_kw : str
        Keyword for bout start
    end_kw : str
        Keyword for bout end
    pad_s : float
        Padding in seconds (default: 0.010 = 10ms)
    
    Returns:
    --------
    windows : list of tuples
        List of (start_time, end_time) in seconds
    """
    df = annotations.copy()

    def has_keyword(x, kw):
        return isinstance(x, str) and kw.lower() in x.lower()

    start_kw = start_kw.lower()
    end_kw = end_kw.lower()

    df["_is_start"] = df[tags_col].apply(lambda x: has_keyword(x, start_kw))
    df["_is_end"] = df[tags_col].apply(lambda x: has_keyword(x, end_kw))
    df = df.sort_values(seconds_col, kind="mergesort")

    starts = df.loc[df["_is_start"], seconds_col].astype(float).to_list()
    ends = df.loc[df["_is_end"], seconds_col].astype(float).to_list()

    windows = []
    si, ei = 0, 0
    
    while si < len(starts) and ei < len(ends):
        s = starts[si]
        while ei < len(ends) and ends[ei] < s:
            ei += 1
        if ei >= len(ends):
            break
        e = ends[ei]
        windows.append((max(0.0, s - pad_s), e + pad_s))
        si += 1

    return windows


def plot_motorneuron_activity(abf, channel=0, xlim=(0, 1), std_range=8):
    """
    Plot motor neuron activity from ABF file.
    
    Parameters:
    -----------
    abf : pyabf.ABF
        ABF file object
    channel : int
        Channel number for motor neuron (default: 0)
    xlim : tuple
        X-axis limits in seconds
    std_range : int
        Number of standard deviations for y-axis range
    """
    fig = plt.figure(figsize=(15, 4))
    
    abf.setSweep(sweepNumber=0, channel=channel)
    plt.plot(abf.sweepX, abf.sweepY, label=f"Channel {channel+1}", 
             linewidth=0.3, color='maroon')
    
    y_min, y_max = get_std_range(abf.sweepY, stdnum=std_range)

    plt.title("Motor Neuron Activity")
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.axis([xlim[0], xlim[1], y_min, y_max])
    plt.legend()
    plt.tight_layout()
    
    return fig


def plot_ventralroot_bursts(abf, channel=1, xlim=(0, 1), std_range=8):
    """
    Plot ventral root burst activity from ABF file.
    
    Parameters:
    -----------
    abf : pyabf.ABF
        ABF file object
    channel : int
        Channel number for VRB (default: 1)
    xlim : tuple
        X-axis limits in seconds
    std_range : int
        Number of standard deviations for y-axis range
    """
    fig = plt.figure(figsize=(15, 4))

    abf.setSweep(sweepNumber=0, channel=channel)
    plt.plot(abf.sweepX, abf.sweepY, label=f"Channel {channel+1}", 
             linewidth=0.3, color='green')
    
    y_min, y_max = get_std_range(abf.sweepY, stdnum=std_range)

    plt.title("Ventral Root Bursts")
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.axis([xlim[0], xlim[1], y_min, y_max])
    plt.legend()
    plt.tight_layout()
    
    return fig


def plot_both_channels(abf, annotations=None, figsize=(15, 8), 
                       xlim=None, std_range=8):
    """
    Plot both motor neuron and VRB channels together.
    
    Parameters:
    -----------
    abf : pyabf.ABF
        ABF file object
    annotations : DataFrame, optional
        Annotations to overlay (bout markers, etc.)
    figsize : tuple
        Figure size
    xlim : tuple, optional
        X-axis limits in seconds
    std_range : int
        Number of standard deviations for y-axis range
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Motor neuron (channel 0)
    abf.setSweep(sweepNumber=0, channel=0)
    ax1.plot(abf.sweepX, abf.sweepY, linewidth=0.3, color='maroon')
    y_min, y_max = get_std_range(abf.sweepY, stdnum=std_range)
    ax1.set_ylabel(abf.sweepLabelY)
    ax1.set_ylim(y_min, y_max)
    ax1.set_title("Motor Neuron Activity (Channel 1)")
    ax1.legend()
    
    # VRB (channel 1)
    abf.setSweep(sweepNumber=0, channel=1)
    ax2.plot(abf.sweepX, abf.sweepY, linewidth=0.3, color='green')
    y_min, y_max = get_std_range(abf.sweepY, stdnum=std_range)
    ax2.set_ylabel(abf.sweepLabelY)
    ax2.set_xlabel(abf.sweepLabelX)
    ax2.set_ylim(y_min, y_max)
    ax2.set_title("Ventral Root Bursts (Channel 2)")
    ax2.legend()
    
    if xlim:
        ax2.set_xlim(xlim)
    
    # Add bout markers if annotations provided
    if annotations is not None:
        bout_windows = extract_bout_windows(annotations)
        for start, end in bout_windows:
            ax1.axvspan(start, end, alpha=0.2, color='yellow')
            ax2.axvspan(start, end, alpha=0.2, color='yellow')
    
    plt.tight_layout()
    return fig


def calculate_vrb_frequencies(annotations, method="blackdot"):
    """
    Calculate VRB frequencies from annotations.
    
    Parameters:
    -----------
    annotations : DataFrame
        Annotations with timing information
    method : str
        Method for frequency calculation: 'blackdot', 'midpoint', or 'interval'
    
    Returns:
    --------
    frequencies : array
        Array of frequencies in Hz
    """
    df = annotations.copy()
    
    if method == "blackdot" and "BlackDotTime" in df.columns:
        times = df["BlackDotTime"].dropna().values
    elif method == "midpoint" and "FirstLastMidpointTime" in df.columns:
        times = df["FirstLastMidpointTime"].dropna().values
    elif "Seconds" in df.columns:
        times = df["Seconds"].dropna().values
    else:
        return np.array([])
    
    # Calculate intervals and convert to frequency
    intervals = np.diff(times)
    frequencies = 1.0 / intervals[intervals > 0]
    
    return frequencies
