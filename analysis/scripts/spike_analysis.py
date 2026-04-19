"""
Cell-attached spike analysis functions.

This module handles:
- Splitting cell-attached recordings from other types
- Identifying spiking vs non-spiking cells
- Calculating spike frequency statistics
"""

import pandas as pd
import numpy as np


def split_cell_attached(sheets):
    """
    Split sheets into those with cell-attached recordings and those without.
    
    Parameters:
    -----------
    sheets : dict
        Dictionary of sheet_name: DataFrame
    
    Returns:
    --------
    matches : dict
        Sheets with at least one 'Cell-attached' row
    non_matches : list
        Sheet names with no 'Cell-attached' rows
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


def split_cell_attached_spiking(sheets):
    """
    Split sheets into those with cell-attached spiking recordings and those without.
    
    Parameters:
    -----------
    sheets : dict
        Dictionary of sheet_name: DataFrame
    
    Returns:
    --------
    matches : dict
        Sheets with at least one 'Cell-attached (spiking)' row
    non_matches : list
        Sheet names with no 'Cell-attached (spiking)' rows
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


def calculate_spike_statistics(sheets, cell_types_df=None):
    """
    Calculate summary spike statistics for each cell.
    
    Parameters:
    -----------
    sheets : dict
        Dictionary of annotation DataFrames
    cell_types_df : DataFrame, optional
        Cell metadata
    
    Returns:
    --------
    summary_df : DataFrame
        Summary with columns: Cell, Motoneuron, Cell Type, Input Resistance,
        mean, median, min, max, total_spikes
    """
    summary_list = []
    
    for cell_name, cell_data in sheets.items():
        if isinstance(cell_data, dict):
            df = cell_data["annotations"]
        else:
            df = cell_data
        
        if "Median Spiking" in df.columns and "Mean Spiking" in df.columns:
            median_spike = df["Median Spiking"].iloc[0] if len(df) > 0 else np.nan
            mean_spike = df["Mean Spiking"].iloc[0] if len(df) > 0 else np.nan
        else:
            # Calculate from frequency data if available
            if "Freq" in df.columns:
                freqs = pd.to_numeric(df["Freq"], errors='coerce').dropna()
                median_spike = freqs.median() if len(freqs) > 0 else np.nan
                mean_spike = freqs.mean() if len(freqs) > 0 else np.nan
            else:
                median_spike = np.nan
                mean_spike = np.nan
        
        # Get min/max/count
        if "Freq" in df.columns:
            freqs = pd.to_numeric(df["Freq"], errors='coerce').dropna()
            min_spike = freqs.min() if len(freqs) > 0 else np.nan
            max_spike = freqs.max() if len(freqs) > 0 else np.nan
            total_spikes = len(freqs)
        else:
            min_spike = np.nan
            max_spike = np.nan
            total_spikes = 0
        
        # Get cell metadata if available
        motoneuron = ""
        cell_type = ""
        input_resistance = np.nan
        
        if cell_types_df is not None:
            cell_info = cell_types_df[cell_types_df["Cell"] == cell_name]
            if not cell_info.empty:
                motoneuron = cell_info.iloc[0].get("Motoneuron", "")
                cell_type = cell_info.iloc[0].get("Cell Type", "")
                input_resistance = cell_info.iloc[0].get("Input Resistance", np.nan)
        
        summary_list.append({
            "Cell": cell_name,
            "Motoneuron": motoneuron,
            "Cell Type": cell_type,
            "Input Resistance": input_resistance,
            "mean": mean_spike,
            "median": median_spike,
            "min": min_spike,
            "max": max_spike,
            "total_spikes": total_spikes
        })
    
    summary_df = pd.DataFrame(summary_list)
    return summary_df
