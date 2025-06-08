import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict
from scipy.stats import ttest_ind
from scipy.stats import f as fdistribution

def bin_rin(rin_value):
    if rin_value < 200:
        return "<200"
    elif rin_value < 400:
        return "200-400"
    else:
        return "400+"
def get_freq_bin(freq):
    if 20 <= freq < 30:
        return "20-30"
    elif 30 <= freq < 40:
        return "30-40"
def group_EI(signal_type):
    if signal_type in ["Inhibitory", "Inhibitory (Rs compensation)"]:
        return "Inhibitory"
   
    elif signal_type in ["Excitatory", "Excitatory (Rs compensation)"]:
        return "Excitatory"
    return signal_type

def group_iSMN(cell_type):
    cell_type = cell_type.strip()
    if cell_type in ["iSMN (dorsal) muscle", "iSMN"]:
        return "iSMN"
    if cell_type == "vSMN":
        return "vSMN"
    return cell_type

def grouprins(cell):
    cell_types_df = pd.read_csv("/Users/Haley/Desktop/murray-neuroscience-lab/Excel processor/List of cells.csv")
    cell_types_df = cell_types_df.dropna(how='all')
    cell_types_df = cell_types_df.iloc[:,:4]
    cell_types_df.reset_index(drop=True,inplace=True)
    types = cell_types_df[cell_types_df["Cell"]==cell]

    return types.iloc[0][3]

def extract_excitatory_features(binned_waveforms):
    """
    Extract excitatory peak gain, peak timing, and peak amplitude for each cell.
    
    Parameters:
        binned_waveforms: dict
            Keys: (freq, signal_type, cell_type, cell_id, Rin)
            Values: DataFrames with waveform data
    
    Returns:
        DataFrame with columns: 
        ['Cell', 'Cell Type', 'Rin', 'Freq', 'Freq Bin', 'Peak Gain', 'Peak Phase', 'Peak Amplitude']
    """
    records = []
    # Assuming phase goes from 0.01 to 1 by 0.02 increments (50 points)
    phase = np.arange(0.01, 1.00001, 0.02)

    for (freq, signal_type, cell_type, cell_id, rin), df in binned_waveforms.items():
        # Check data shape matches expected waveform shape (50 time points x 4 columns)
        if df.shape != (50, 4):
            continue
        #  Excitatory signals only
        if group_EI(signal_type) != 'Excitatory':
            continue

        norm_currents = df['Normalized Current'].values
        peak_idx = np.argmax(norm_currents)
        peak_phase = phase[peak_idx]
        peak_amplitude = norm_currents[peak_idx]  
        slope, intercept = np.polyfit(phase, norm_currents, 1)  # linear fit

        freq_bin = get_freq_bin(freq)

        records.append({
            'Cell': cell_id,
            'Cell Type': group_iSMN(cell_type),
            'Rin': rin,
            'Freq': freq,
            'Freq Bin': freq_bin,
            'Peak Gain': slope,
            'Peak Phase': peak_phase,
            'Peak Amplitude': peak_amplitude
        })

    return pd.DataFrame(records)

def aggregate_features(features_df):
    features_df['Rin group'] = features_df['Cell'].apply(grouprins)

    # Group by Cell Type, Rin Bin, Frequency Bin
    grouped = features_df.groupby(['Cell Type','Freq Bin','Rin group'])
    
    summary = grouped.agg({
        'Peak Gain': ['mean', 'std', 'count'],
        'Peak Phase': ['mean', 'std'],
        'Peak Amplitude': ['mean', 'std']
    })
    
    # Flatten multi-index columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    return summary

def compute_ttests(features_df, feature='Peak Gain', group_col='Cell Type', compare_col='Rin Bin', freq_col='Freq Bin'):
    """
    Compute t-tests for feature values across cell types or Rin bins within each frequency bin.
    
    Parameters:
        features_df: pd.DataFrame
            DataFrame containing columns for the feature, group_col, compare_col, and freq_col.
        feature: str
            The feature column on which to perform t-tests.
        group_col: str
            Column name to group by first (e.g. 'Cell Type' or 'Rin Bin').
        compare_col: str
            Column name for categories to compare via t-tests within each group (e.g. 'Rin Bin' or 'Cell Type').
        freq_col: str
            Column name for swim frequency bins.
    
    Returns:
        pd.DataFrame with columns:
            [freq_col, group_col, compare_col_1, compare_col_2, t_stat, p_value]
    """
    results = []

    freq_bins = features_df[freq_col].unique()
    groups = features_df[group_col].unique()
    
    for freq in freq_bins:
        freq_df = features_df[features_df[freq_col] == freq]
        
        for group in groups:
            subset = freq_df[freq_df[group_col] == group]
            compare_categories = subset[compare_col].unique()
            
            # Compare all pairs within compare_col categories
            for i, cat1 in enumerate(compare_categories):
                for cat2 in compare_categories[i+1:]:
                    data1 = subset[subset[compare_col] == cat1][feature].dropna()
                    data2 = subset[subset[compare_col] == cat2][feature].dropna()
                    
                    if len(data1) > 1 and len(data2) > 1:
                        t_stat, p_value = ttest_ind(data1, data2, equal_var=False)
                        results.append({
                            freq_col: freq,
                            group_col: group,
                            f"{compare_col}_1": cat1,
                            f"{compare_col}_2": cat2,
                            't_stat': t_stat,
                            'p_value': p_value
                        })

    return pd.DataFrame(results)
 
def hotellings_t2_test(X, Y):
    """
    Perform Hotelling's T-squared test for two multivariate samples.
    
    Parameters:
        X: np.ndarray of shape (n1, p)
        Y: np.ndarray of shape (n2, p)
    
    Returns:
        T2_stat: float
        F_stat: float
        p_value: float
    """
    n1, p = X.shape
    n2, _ = Y.shape

    mean_diff = np.mean(X, axis=0) - np.mean(Y, axis=0)
    S1 = np.cov(X, rowvar=False)
    S2 = np.cov(Y, rowvar=False)
    Sp = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
    Sp_inv = np.linalg.pinv(Sp)  

    T2 = (n1 * n2) / (n1 + n2) * mean_diff.T @ Sp_inv @ mean_diff
    F_stat = (n1 + n2 - p - 1) * T2 / ((n1 + n2 - 2) * p)
    df1 = p
    df2 = n1 + n2 - p - 1
    p_value = 1 - fdistribution.cdf(F_stat, df1, df2)

    return T2, F_stat, p_value

def compute_hotellings_tests(features_df, feature_cols, group_col='Cell Type', compare_col='Rin Bin', freq_col='Freq Bin'):
    """
    Perform Hotelling's T² tests for multiple features across categories within each group.
    
    Parameters:
        features_df: DataFrame containing the features and grouping columns.
        feature_cols: list of feature names to include in the test.
    
    Returns:
        DataFrame with Hotelling's T² test results across groupings.
    """
    results = []
    freq_bins = features_df[freq_col].dropna().unique()
    groups = features_df[group_col].dropna().unique()

    for freq in freq_bins:
        freq_df = features_df[features_df[freq_col] == freq]

        for group in groups:
            subset = freq_df[freq_df[group_col] == group]
            compare_categories = subset[compare_col].dropna().unique()

            for i, cat1 in enumerate(compare_categories):
                for cat2 in compare_categories[i+1:]:
                    data1 = subset[subset[compare_col] == cat1][feature_cols].dropna()
                    data2 = subset[subset[compare_col] == cat2][feature_cols].dropna()

                    if len(data1) > len(feature_cols) and len(data2) > len(feature_cols):
                        try:
                            T2, F_stat, p_val = hotellings_t2_test(data1.values, data2.values)
                            results.append({
                                freq_col: freq,
                                group_col: group,
                                f"{compare_col}_1": cat1,
                                f"{compare_col}_2": cat2,
                                'T2_stat': T2,
                                'F_stat': F_stat,
                                'p_value': p_val
                            })
                        except Exception as e:
                            print(f"Failed comparison {cat1} vs {cat2} at {freq}, {group}: {e}")
    return pd.DataFrame(results)
def compute_hotellings_tests_grouped(features_df, feature_cols, compare_col='Rin Bin', freq_col='Freq Bin'):
    """
    Perform Hotelling's T² tests for multiple features across categories within each group.
    
    Parameters:
        features_df: DataFrame containing the features and grouping columns.
        feature_cols: list of feature names to include in the test.
    
    Returns:
        DataFrame with Hotelling's T² test results across groupings.
    """
    results = []
    freq_bins = features_df[freq_col].dropna().unique()

    for freq in freq_bins:
        freq_df = features_df[features_df[freq_col] == freq]
        compare_categories = freq_df[compare_col].dropna().unique()

        for i, cat1 in enumerate(compare_categories):
            for cat2 in compare_categories[i+1:]:
                data1 = freq_df[freq_df[compare_col] == cat1][feature_cols].dropna()
                data2 = freq_df[freq_df[compare_col] == cat2][feature_cols].dropna()

                if len(data1) > len(feature_cols) and len(data2) > len(feature_cols):
                    try:
                        T2, F_stat, p_val = hotellings_t2_test(data1.values, data2.values)
                        results.append({
                            freq_col: freq,
                            f"{compare_col}_1": cat1,
                            f"{compare_col}_2": cat2,
                            'T2_stat': T2,
                            'F_stat': F_stat,
                            'p_value': p_val
                        })
                    except Exception as e:
                            print(f"Failed comparison {cat1} vs {cat2} at {freq},: {e}")
    return pd.DataFrame(results)
