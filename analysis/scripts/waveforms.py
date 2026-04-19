"""
Waveform processing and analysis functions.

This module handles:
- Loading annotation sheets with cell metadata
- Associating ABF files with annotations
- Extracting waveform segments
- Normalizing and binning waveforms
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
import os
try:
    from .utils import DEFAULT_PARENT_PATH, concat_one_channel
except ImportError:
    from utils import DEFAULT_PARENT_PATH, concat_one_channel


def make_sheets_dict(sheet_names, parent_folder_path=DEFAULT_PARENT_PATH, 
                     cell_types_df=None):
    """
    Load annotation sheets and associate with cell metadata.
    
    Parameters:
    -----------
    sheet_names : array-like
        List of cell/sheet names
    parent_folder_path : str
        Path to project folder
    cell_types_df : DataFrame, optional
        Cell metadata (if None, will load from file)
    
    Returns:
    --------
    sheets : dict
        Dictionary mapping sheet names to DataFrames with annotations
    """
    if cell_types_df is None:
        cell_types_df = pd.read_csv(
            os.path.join(parent_folder_path,
                        "murray-neuroscience-lab/data/annotations/summary_spikes2.csv")
        )
    
    sheets = {}
    
    for sheet in sheet_names:
        file_path = os.path.join(parent_folder_path,
                                "murray-neuroscience-lab/New processed excels",
                                f"{sheet}.csv")
        
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        df[["Trace name", "Tags", "Type"]] = df[["Trace name", "Tags", "Type"]].astype("string")
        
        # Add cell metadata
        types = cell_types_df[cell_types_df["Cell"] == sheet]
        if not types.empty:
            df.loc[:, "Currents Channel"] = types.iloc[0].get("currents", 0)
            df.loc[:, "VRB Channel"] = types.iloc[0].get("vrb", 1)
            df.loc[:, "Median Spiking"] = types.iloc[0].get("median", np.nan)
            df.loc[:, "Mean Spiking"] = types.iloc[0].get("mean", np.nan)
        
        sheets[sheet] = df
    
    return sheets


def add_abfs(sheets, abfs_names, parent_folder_path_ABFS):
    """
    Associate ABF files with annotation sheets.
    
    Parameters:
    -----------
    sheets : dict
        Dictionary of annotation DataFrames
    abfs_names : array-like
        List of ABF trace names
    parent_folder_path_ABFS : str
        Path to folder containing ABF files
    
    Returns:
    --------
    sheets : dict
        Modified dictionary with structure:
        {sheet_name: {"annotations": df, "abfs": {trace_name: abf_object}}}
    """
    for sheet in sheets.keys():
        df = sheets[sheet]
        traces = df["Trace name"].unique()
        
        abfs = {}
        for trace in abfs_names:
            if trace in traces:
                file_path = os.path.join(parent_folder_path_ABFS, f"{trace}.abf")
                if os.path.isfile(file_path):
                    try:
                        abf = pyabf.ABF(file_path)
                        abfs[trace] = abf
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"Warning: File not found {file_path}")
        
        sheets[sheet] = {
            "annotations": df,
            "abfs": abfs
        }
    
    return sheets


def make_waveforms(abf, df, remove_outliers=False, iqr_multiplier=1.5):
    """
    Extract waveforms from ABF file based on annotation timing.
    
    Parameters:
    -----------
    abf : pyabf.ABF
        ABF file object
    df : DataFrame
        Annotations with timing information
    remove_outliers : bool
        Whether to remove outliers using IQR method
    iqr_multiplier : float
        Multiplier for IQR outlier detection
    
    Returns:
    --------
    waveforms : dict
        Dictionary with keys (freq, signal_type, median, mean) and values
        as DataFrames with columns: Time, Current, Phase, Normalized Current
    """
    # Use optimized concat function to get full trace once
    full_time, full_current = concat_one_channel(abf, ch=0)

    # Ensure Seconds column exists
    if "Seconds" not in df.columns:
        df["Seconds"] = pd.to_numeric(df["On time"], errors="coerce") * 0.001

    waveforms = {}
    
    for i in range(len(df) - 1):
        t_0 = df.iloc[i]["Seconds"]
        t_f = df.iloc[i + 1]["Seconds"]

        # Use numpy indexing for speed - much faster than DataFrame filtering
        mask = (full_time >= t_0) & (full_time <= t_f)
        time_segment = full_time[mask]
        current_segment = full_current[mask]
        
        if len(time_segment) == 0:
            continue
        
        # Create DataFrame only for the extracted segment
        abf_waveform = pd.DataFrame({
            'Time': time_segment,
            'Current': current_segment
        })
        
        if abf_waveform.empty:
            print(f"Warning: Empty waveform at index {i}, t_0={t_0}, t_f={t_f}")
            continue
        
        # Optional: Remove outliers (IQR method)
        if remove_outliers:
            Q1 = abf_waveform["Current"].quantile(0.25)
            Q3 = abf_waveform["Current"].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            abf_waveform = abf_waveform[
                (abf_waveform["Current"] >= lower) & 
                (abf_waveform["Current"] <= upper)
            ]
            
            if abf_waveform.empty:
                continue

        # Add phase (0 to 1 across the segment)
        abf_waveform["Phase"] = (abf_waveform["Time"] - t_0) / (t_f - t_0)

        # Normalize Current
        y_max = abf_waveform["Current"].max()
        y_min = abf_waveform["Current"].min()
        if y_max != y_min:
            abf_waveform["Normalized Current"] = (
                (abf_waveform["Current"] - y_min) / (y_max - y_min)
            )
        else:
            abf_waveform["Normalized Current"] = 0.5

        # Create dictionary key
        freq = 1 / (t_f - t_0) if t_f != t_0 else 0
        signal_type = df.iloc[i].get("Type", "Unknown")
        median = df.iloc[i].get("Median Spiking", np.nan)
        mean = df.iloc[i].get("Mean Spiking", np.nan)
        key = (freq, signal_type, median, mean)

        waveforms[key] = abf_waveform

    return waveforms


def sheets_to_waveforms(sheets):
    """
    Extract waveforms from all sheets.
    
    Parameters:
    -----------
    sheets : dict
        Dictionary with structure from add_abfs()
    
    Returns:
    --------
    all_waveforms : dict
        Nested dictionary {sheet_name: {trace_name: waveforms_dict}}
    """
    all_waveforms = {}
    total_sheets = len(sheets)
    
    for idx, (sheet_name, sheet_data) in enumerate(sheets.items(), 1):
        print(f"Processing {idx}/{total_sheets}: {sheet_name}", end="")
        
        if isinstance(sheet_data, dict):
            annotations = sheet_data["annotations"]
            abfs = sheet_data["abfs"]
        else:
            print(" - skipped (unexpected structure)")
            continue
        
        sheet_waveforms = {}
        
        for trace_name, abf in abfs.items():
            # Filter annotations for this trace
            trace_annotations = annotations[
                annotations["Trace name"] == trace_name
            ]
            
            if not trace_annotations.empty:
                try:
                    waveforms = make_waveforms(abf, trace_annotations)
                    sheet_waveforms[trace_name] = waveforms
                except Exception as e:
                    print(f"\n  Error processing {trace_name}: {e}")
        
        all_waveforms[sheet_name] = sheet_waveforms
        print(f" - {len(sheet_waveforms)} traces processed")
    
    return all_waveforms


def bin_wave(onewave, num_bins=100):
    """
    Bin a waveform into equal phase bins and average within bins.
    
    Parameters:
    -----------
    onewave : DataFrame
        Waveform with Phase and Normalized Current columns
    num_bins : int
        Number of bins (default: 100)
    
    Returns:
    --------
    binned_df : DataFrame
        Binned waveform with columns: Phase, Normalized Current
    """
    bins = np.linspace(0, 1, num_bins + 1)
    onewave['Bin'] = pd.cut(onewave['Phase'], bins=bins, labels=False, include_lowest=True)
    
    binned = onewave.groupby('Bin').agg({
        'Phase': 'mean',
        'Normalized Current': 'mean'
    }).reset_index(drop=True)
    
    return binned
