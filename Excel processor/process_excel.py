'''
April 19

'''

# Imports
import pandas as pd
import numpy as np

file_path = "/Users/haleyoro/Desktop/murray-neuroscience-lab/4dpf_VC_MN_Mac.2012_04_25_cell3.csv"


def make_blocks(data):
    '''
    Function that takes raw df with "Trace name" as in position [0,0] and returns the blocks
    '''
    # Each block has 5 columns: Trace name, ID, On time, Freq, Tags
    num_cols = data.shape[1]
    num_rows = data.shape[0]    
    block_size = 6
    # Separate into blocks
    blocks = []
    for start in range(0, num_cols, block_size):  
        end = start + block_size
        if (end <= num_cols) & (data.iloc[0,start] == "Trace name"):
            block = data.iloc[:, start:end]
            # print(block)
            block.columns = ['Trace name', 'ID', 'On time', 'Freq', 'Tags','Type']
            if start == 0:
                block['Type']=['Inhibitory']*num_rows
            elif start == 6:
                block['Type']=['Excitatory']*num_rows
            elif start == 12:
                block['Type']=['Cell-attached']*num_rows
            blocks.append(block)
    # Combine all blocks into one DataFrame
    df_clean = pd.concat(blocks, ignore_index=True)

    # Drop rows that have NaN in Trace name or ID
    df_clean.dropna(subset=['Trace name', 'ID'], how='all', inplace=True)

    df_clean = df_clean.dropna(how='all')
    return df_clean


def prepare_df(df):
    '''
    Function to convet column data types to feed make_ waveworms function
    '''
    df['Trace name'] = df['Trace name'].astype('string')
    df["On time"] = pd.to_numeric(df["On time"], errors="coerce")
    df["Freq"] = pd.to_numeric(df["Freq"], errors="coerce")
    df["Seconds"] = df['On time']*0.001
    return df


def trace_filter(df):
    pattern = r'^\d{4}_\d{2}_\d{2}_\d{4}$'
    index = df['Trace name'].str.match(pattern, na=False)
    df_filtered = df[index]
    return df_filtered

def create_df(file_path):
    '''
    Complete function that takes csv file path from excel spreadsheets and turns it into a readable dataframe that show event times

    '''

    df = pd.read_csv(file_path,header=None)

    # Read the raw CSV , headers are irregular
    data = df.iloc[1:,1:]

    df_clean = make_blocks(data)
    prep_df = prepare_df(df_clean)
    df_filtered = trace_filter(prep_df)
    # if df_filtered.iloc[0]['ID'] != 1:
    df_filtered = df_filtered.iloc[1:]
    df_filtered = df_filtered.dropna(how='all')

    return df_filtered

df_filtered = create_df(file_path)
print(df_filtered)

# df_filtered.to_csv('processed_data.csv', index=False)
