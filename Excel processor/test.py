import pandas as pd
import numpy as np

def load_trace_data_general(filepath):
    # Read CSV, skip initial empty lines and infer header
    df_raw = pd.read_csv(filepath, header=None)

    # Try to find the row containing "Trace name" to locate the header
    header_row = df_raw.apply(lambda row: row.astype(str).str.contains("Trace name", na=False)).any(axis=1).idxmax()

    # Re-read the file using that header row as column names
    df = pd.read_csv(filepath, header=header_row)
    
    # Keep only columns relevant for waveform data (contains 'Trace name', 'ID', 'On time', 'Freq', 'Tags')
    possible_groups = []
    i = 0
    while i < len(df.columns):
        subset = df.columns[i:i+5]
        if set(['Trace name', 'ID', 'On time', 'Freq', 'Tags']).issubset(subset):
            possible_groups.append(subset)
            i += 5
        else:
            i += 1

    # Create combined dataframe with label (Inhibitory, Excitatory, Cell attached, etc.)
    all_data = []
    for group in possible_groups:
        trace_col = group[0]
        trace_name = df[trace_col].dropna().astype(str).iloc[0]
        
        if 'inhibitory' in trace_name.lower():
            label = 'Inhibitory'
        elif 'excitatory' in trace_name.lower():
            label = 'Excitatory'
        elif 'cell' in trace_name.lower():
            label = 'Cell attached'
        else:
            label = 'Unknown'

        sub_df = df[list(group)].dropna(how='all')
        sub_df.columns = ['Trace name', 'ID', 'On time', 'Freq', 'Tags']
        sub_df['Type'] = label
        all_data.append(sub_df)

    combined = pd.concat(all_data, ignore_index=True)
    return combined

# Dummy run (file path needs to be updated to actual file)
# combined_df = load_trace_data_general('/Users/Haley/Desktop/Neuroscience Lab/4dpf_VC_MN_Mac_2012_04_25_cell1.csv')
# combined_df.head()

df = pd.read_csv('/Users/haleyoro/Desktop/murray-neuroscience-lab/Excel processor/unprocessed_2012_04_25_cell3.csv',header = None)
print(df.head())
