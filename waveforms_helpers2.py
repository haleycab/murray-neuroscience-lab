
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyabf
import os
# parent_folder_path = "/Users/haleyoro/Desktop/" # work on library computer
parent_folder_path = "/Users/Haley/Desktop/" # work on local computer

# Get sheets
cell_types_df = pd.read_csv(parent_folder_path+"murray-neuroscience-lab/Cleaned, updated files from Spring/summary_spikes2.csv")

cell_types_df.reset_index(drop=True,inplace=True)
sheets_types = cell_types_df["Cell"].unique().tolist()

sheet_names_df = pd.read_csv(parent_folder_path+'murray-neuroscience-lab/New processed excels/sheet_names.csv', header=None)
sheet_names = sheet_names_df.iloc[:,0].values

# First load annotated sheets into csv
# def make_sheets_dict(sheet_names,parent_folder_path):
#     sheets = {}

#     for sheet in sheet_names:
#         file_path = parent_folder_path+"murray-neuroscience-lab/New processed excels/"+sheet+".csv"
#         df = pd.read_csv(file_path)
#         df[["Trace name","Tags","Type"]] = df[["Trace name","Tags","Type"]].astype("string")
#         types = cell_types_df[cell_types_df["Cell"]==sheet]
#         df.loc[:,"Median Spiking"] = types.iloc[0]["median"]
#         df.loc[:,"Mean Spiking"] = types.iloc[0]["mean"]
#         sheets[sheet] = df

#     return sheets

# Second, add abfs to sheet

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
    


iqr_multiplier =  1.5

def make_waveforms(abf, df):
    '''
    Function that takes an abf file and a df of the annotations
    Returns a dictionary with waveforms labeled by their frequency, cell type, signal type, and rin
    '''

    # Build full trace across all sweeps
    full_time = []
    full_current = []
    for sweep in range(abf.sweepCount):
        abf.setSweep(sweepNumber=sweep, channel=0)
        sweep_time = abf.sweepX + sweep * abf.sweepLengthSec
        sweep_current = abf.sweepY
        full_time.append(sweep_time)
        full_current.append(sweep_current)

    # Flatten lists
    full_time = np.concatenate(full_time)
    full_current = np.concatenate(full_current)
    # Make dataframe 
    abf_df = pd.DataFrame({
        'Time': full_time,
        'Current': full_current
    })

    if "Seconds" not in df.columns:
        df["Seconds"] = pd.to_numeric(df["On time"], errors="coerce") * 0.001

    waveforms = {}
    for i in range(len(df) - 1):
        t_0 = df.iloc[i]["Seconds"]
        t_f = df.iloc[i + 1]["Seconds"]

        abf_waveform = abf_df[(abf_df["Time"] >= t_0) & (abf_df["Time"] <= t_f)].copy()
        # previous error
        if abf_waveform.empty:
            raise ValueError(
                f"abf_waveform is empty for index {i}, t_0={t_0}, t_f={t_f}, Trace name={df.iloc[i]['Trace name']}"
            )
            
        # Optional: Remove outliers (IQR method)
        # Q1 = abf_waveform["Current"].quantile(0.25)
        # Q3 = abf_waveform["Current"].quantile(0.75)
        # IQR = Q3 - Q1
        # lower = Q1 - iqr_multiplier * IQR
        # upper = Q3 + iqr_multiplier * IQR
        # abf_waveform = abf_waveform[(abf_waveform["Current"] >= lower) & (abf_waveform["Current"] <= upper)]

        # if abf_waveform.empty:
        #     continue  


        # Add phase (0 to 1 across the segment)
        abf_waveform["Phase"] = (abf_waveform["Time"] - t_0) / (t_f - t_0)

        # Normalize Current
        y_max = abf_waveform["Current"].max()
        y_min = abf_waveform["Current"].min()
        abf_waveform["Normalized Current"] = (abf_waveform["Current"] - y_min) / (y_max - y_min)

        # Dict keys
        freq = 1 / (t_f - t_0)
        signal_type = df.iloc[i]["Type"]
        median = df.iloc[i]["Median Spiking"]
        mean = df.iloc[i]["Mean Spiking"]
        key = (freq, signal_type, median, mean)

        waveforms[key] = abf_waveform

    return waveforms

def concat_one_channel(abf, ch=0):
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
        tx = abf.sweepX.copy()           # (n,) seconds, starts at 0
        y  = abf.sweepY.copy().astype(np.float32, copy=False)
        full_time_chunks.append(tx + offsets_sec[s])
        full_current_chunks.append(y)

    full_time    = np.concatenate(full_time_chunks)
    full_current = np.concatenate(full_current_chunks)
    return full_time, full_current

def calculate_midpoints_and_frequencies(all_annotations):  

    for i in range(len(all_annotations)):
        # print(i)
            # if it is the last annotation
        if i+1 == len(all_annotations):
            # calculate interval from previous midpoint to current black dot time
            half_interval = all_annotations.loc[i,"BlackDotTime"] - all_annotations.loc[i-1,"Midpoint"]
            midpoint = all_annotations.loc[i,"BlackDotTime"] + half_interval
            all_annotations.loc[i,"Midpoint"] = midpoint
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

def new_make_waveforms(abf, df):
    '''
    Function that takes an abf file and a df of the annotations
    Returns a dictionary with waveforms labeled by their frequency, cell type, signal type, and rin
    '''

    # extract channel data
    if len(df["Currents Channel"].unique()) != 1:
        print("Warning: Multiple Currents Channels found:", all_annotations["Currents Channel"].unique())
    # if len(df["VRB Channel"].unique()) != 1:
    #     print("Warning: Multiple Currents Channels found:", all_annotations["VRB Channel"].unique())
    # vrb_ch = df.loc[0,"VRB Channel"]
    currents_ch = df.loc[0,"Currents Channel"]

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
        key = (freq, signal_type, median, mean)

        waveforms[key] = abf_waveform

    return waveforms

def sheets_to_waveforms(sheets):
    all_waveforms = {}
    for sheet in sheets_types: # using sheet_types instead of sheet_keys() since not all sheets are labeled
        # print(sheet)
        traces_df_mkwf = sheets[sheet]["annotations"]
        # print(traces_df_mkwf)
        abfs_mkwv = sheets[sheet]["abfs"]
        # print(abfs_mkwv)
        traces_df_mkwf = traces_df_mkwf[traces_df_mkwf["Type"] != "Cell-attached (spiking)"]
        traces = traces_df_mkwf["Trace name"].unique().tolist()
        # print(traces)
        for trace in abfs_mkwv.keys():

            abf_mkwv = abfs_mkwv[trace]
            # print(abf_mkwv)
            df_mkwv = traces_df_mkwf[traces_df_mkwf["Trace name"]==trace]
            # print(df_mkwv)

            waveforms = make_waveforms(abf_mkwv,df_mkwv)

            for key, value in waveforms.items():
                # combined_key = f"{sheet}_{trace}_{key}"  
                all_waveforms[key] = value
        
    return all_waveforms

def bin_wave(onewave):

    bins = np.linspace(0, 1, 51, endpoint = True)

    onewave['Phase Bin'] = pd.cut(onewave['Phase'], bins=bins, include_lowest=True)

    binned_avg = onewave.groupby('Phase Bin',observed=True)[['Current','Normalized Current']].mean().reset_index()

    bin_centers = binned_avg['Phase Bin'].apply(lambda x: x.mid)
    binned_avg['Phase'] = bin_centers
    return binned_avg

def bin_wave_100(onewave):
    """
    Bins the 'Phase' column into 100 equal-width bins between 0 and 1,
    computes the mean of 'Current' and 'Normalized Current' within each bin,
    and returns a DataFrame with bin intervals, averages, and numeric bin centers.
    """
    # Create 100 bins between 0 and 1
    bins = np.linspace(0, 1, 101, endpoint=True)

    # Assign each Phase value to a bin
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


def recenter_phase(df, col="Phase"):
    """
    Take a DataFrame with a Phase column in [0, 1] and remap to [-0.5, 0.5].

    - Values < 0.5 stay the same.
    - Values >= 0.5 are shifted by -1.
    """
    df = df.copy()
    phases = df[col].to_numpy()
    phases = np.where(phases >= 0.5, phases - 1.0, phases)
    df[col] = phases
    return df
