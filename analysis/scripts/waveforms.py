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


LEGACY_SPRING_DIR = os.path.join(
    DEFAULT_PARENT_PATH,
    "murray-neuroscience-lab",
    "Cleaned, updated files from Spring",
)


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


def make_waveforms(abf, df):
    """
    Function that takes an ABF file and a DataFrame of annotations.
    Returns a dictionary with waveforms labeled by
    (frequency, signal_type, median_spiking, mean_spiking).
    
    Parameters:
    -----------
    abf : pyabf.ABF
        ABF file object
    df : DataFrame
        Annotations with timing information
    The implementation below is intentionally aligned with the most recent
    legacy notebook function `new_make_waveforms`.
    """
    currents_ch = df.loc[0, "Currents Channel"] if "Currents Channel" in df.columns else 0
    full_time, full_current = concat_one_channel(abf, ch=currents_ch)
    abf_df = pd.DataFrame({"Time": full_time, "Current": full_current})

    if "Midpoint" not in df.columns and "Seconds" not in df.columns:
        df["Seconds"] = pd.to_numeric(df["On time"], errors="coerce") * 0.001

    waveforms = {}

    for i in range(len(df) - 1):
        if "Midpoint" in df.columns:
            t_0 = df.iloc[i]["Midpoint"]
            t_f = df.iloc[i + 1]["Midpoint"]
            phase_shift = -0.5
        else:
            t_0 = df.iloc[i]["Seconds"]
            t_f = df.iloc[i + 1]["Seconds"]
            phase_shift = 0.0

        abf_waveform = abf_df[(abf_df["Time"] >= t_0) & (abf_df["Time"] <= t_f)].copy()

        if abf_waveform.empty:
            raise ValueError(
                f"abf_waveform is empty for index {i}, t_0={t_0}, t_f={t_f}, Trace name={df.iloc[i].get('Trace name', 'Unknown')}"
            )

        # Add phase (legacy midpoint behavior: -0.5 to 0.5)
        abf_waveform["Phase"] = ((abf_waveform["Time"] - t_0) / (t_f - t_0)) + phase_shift

        # Normalize Current
        y_max = abf_waveform["Current"].max()
        y_min = abf_waveform["Current"].min()
        abf_waveform["Normalized Current"] = (abf_waveform["Current"] - y_min) / (y_max - y_min)

        # Create dictionary key
        freq = 1 / (t_f - t_0)
        signal_type = df.iloc[i].get("Type", "Unknown")
        median = df.iloc[i].get("Median Spiking", np.nan)
        mean = df.iloc[i].get("Mean Spiking", np.nan)
        key = (freq, signal_type, median, mean)

        waveforms[key] = abf_waveform

    return waveforms


def get_abf_and_annotations(abf_name, sheets_abfs):
    """
    Given an abf_name, return the ABF object and its corresponding
    annotations DataFrame subset.
    """
    for _, content in sheets_abfs.items():
        if abf_name in content["abfs"]:
            abf_obj = content["abfs"][abf_name]
            annotations_df = content["annotations"]
            annotations_subset = annotations_df[annotations_df["Trace name"] == abf_name]
            return abf_obj, annotations_subset
    return None, None


def calculate_midpoints_and_frequencies_avg(all_annotations_in):
    """
    Legacy midpoint logic used by the newest notebooks that call
    `new_make_waveforms`.
    """
    all_annotations = all_annotations_in.copy().reset_index(drop=True)
    all_annotations["Midpoint"] = np.nan
    all_annotations["Freq_Blackdots"] = np.nan
    all_annotations["Freq_Midpoints"] = np.nan

    for i in range(len(all_annotations)):
        is_start = str(all_annotations.loc[i, "AnnotationType"]).lower() == "start"
        prev_is_start = (i > 0) and (
            str(all_annotations.loc[i - 1, "AnnotationType"]).lower() == "start"
        )

        if is_start or prev_is_start:
            center_i = all_annotations.loc[i, "BlackDotTime"]
        else:
            center_i = np.nanmean([
                all_annotations.loc[i, "BlackDotTime"],
                all_annotations.loc[i, "FirstLastMidpointTime"],
            ])

        if i + 1 == len(all_annotations):
            if i > 0:
                half_interval = center_i - all_annotations.loc[i - 1, "Midpoint"]
                midpoint = center_i + half_interval
                all_annotations.loc[i, "Midpoint"] = midpoint
        else:
            next_is_start = str(all_annotations.loc[i + 1, "AnnotationType"]).lower() == "start"
            if is_start or next_is_start:
                center_ip1 = all_annotations.loc[i + 1, "BlackDotTime"]
            else:
                center_ip1 = np.nanmean([
                    all_annotations.loc[i + 1, "BlackDotTime"],
                    all_annotations.loc[i + 1, "FirstLastMidpointTime"],
                ])

            interval = center_ip1 - center_i
            all_annotations.loc[i + 1, "Freq_Blackdots"] = 1 / interval
            midpoint = center_i + interval / 2
            all_annotations.loc[i, "Midpoint"] = midpoint

        if i == 0:
            all_annotations.loc[i, "Freq_Midpoints"] = np.nan
            all_annotations.loc[i, "Freq_Blackdots"] = np.nan
        else:
            interval_midpoints = (
                all_annotations.loc[i, "Midpoint"] - all_annotations.loc[i - 1, "Midpoint"]
            )
            all_annotations.loc[i, "Freq_Midpoints"] = 1 / interval_midpoints

    return all_annotations


def load_legacy_merged_annotations(parent_folder_path=DEFAULT_PARENT_PATH):
    """
    Load variance-based annotations calculated using variance binning strategy.
    
    This function loads all_vrb_annotationsnoNaNs.csv which contains BlackDotTime
    values calculated from the variance binning detection method. No merging with
    Chebyshev annotations is performed - only the variance-based annotations are used.
    """
    spring_dir = os.path.join(
        parent_folder_path,
        "murray-neuroscience-lab",
        "Cleaned, updated files from Spring",
    )
    variance_path = os.path.join(spring_dir, "all_vrb_annotationsnoNaNs.csv")
    
    # Load only the variance-based annotations
    merged = pd.read_csv(variance_path)
    
    return merged


def sheets_to_waveforms(sheets, merged_annotations=None, use_legacy_new_make_waveforms=True,
                        parent_folder_path=DEFAULT_PARENT_PATH):
    """
    Extract waveforms from all sheets.
    
    Parameters:
    -----------
    sheets : dict
        Dictionary with structure from add_abfs()
    
    Returns:
    --------
    all_waveforms : dict
        If `use_legacy_new_make_waveforms=True`, returns the same flat dictionary
        structure produced by the newest legacy notebooks that use
        `new_make_waveforms`. Otherwise returns the original nested refactor
        structure.
    """
    if use_legacy_new_make_waveforms:
        if merged_annotations is None:
            merged_annotations = load_legacy_merged_annotations(
                parent_folder_path=parent_folder_path
            )

        all_waveforms = {}
        abf_names = merged_annotations["Trace name"].unique()

        for abf_name in abf_names:
            abf, _ = get_abf_and_annotations(abf_name, sheets)
            if abf is None:
                continue

            annotations = merged_annotations[merged_annotations["Trace name"] == abf_name]
            annotations = calculate_midpoints_and_frequencies_avg(annotations)
            waveforms = make_waveforms(abf, annotations)

            for key, value in waveforms.items():
                all_waveforms[key] = value

        return all_waveforms

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
    Bin waveform phase and average Current and Normalized Current in each bin.
    This mirrors legacy `bin_wave_100` behavior from the most recent notebooks.
    
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
    bins = np.linspace(-0.5, 0.5, num_bins + 1, endpoint=True)
    onewave = onewave.copy()
    onewave['Phase Bin'] = pd.cut(onewave['Phase'], bins=bins, include_lowest=True)

    binned_avg = (
        onewave.groupby('Phase Bin', observed=True)[['Current', 'Normalized Current']]
        .mean()
        .reset_index()
    )

    binned_avg['Phase'] = binned_avg['Phase Bin'].apply(lambda x: x.mid)
    return binned_avg
