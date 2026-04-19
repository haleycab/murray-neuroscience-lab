# Murray Neuroscience Lab - Analysis Pipeline

## Overview
This directory contains the core analysis pipeline for processing electrophysiology data from zebrafish motor neurons, including ventral root burst (VRB) detection, spike analysis, and waveform characterization.

## Directory Structure

```
analysis/
├── scripts/           # Core Python modules with reusable functions
│   ├── waveforms.py          # Waveform processing and analysis
│   ├── vrb_analysis.py       # Ventral root burst detection and bout analysis
│   ├── spike_analysis.py     # Cell-attached spike processing
│   └── utils.py              # Shared utility functions
├── notebooks/         # Jupyter notebooks for interactive analysis
│   └── main_pipeline.ipynb   # Main analysis workflow
└── README.md         # This file

data/
├── annotations/      # ABF file annotations and metadata
└── processed/        # Processed results and summaries
```

## Main Analysis Components

### 1. **VRB Analysis** (`vrb_analysis.py`)
- Detects ventral root bursts from electrophysiology recordings
- Identifies bout start/end times with configurable padding
- Calculates burst frequencies and timing
- Generates plots of motor neuron activity and burst timing

**Key Functions:**
- `_extract_bout_windows()` - Extracts bout timing from annotations
- `plot_motorneuron_activity()` - Visualizes channel 1 (motor neuron)
- `plot_ventralroot_bursts()` - Visualizes channel 2 (ventral root)

### 2. **Spike Analysis** (`spike_analysis.py`)
- Processes cell-attached recordings
- Filters for spiking vs non-spiking cells
- Calculates spike statistics (mean, median, min, max frequencies)
- Generates summary statistics by cell type

**Key Functions:**
- `split_cell_attached()` - Separates cell-attached recordings
- `split_cell_attached_spiking()` - Filters for spiking cells only

### 3. **Waveform Processing** (`waveforms.py`)
- Loads and processes ABF files
- Extracts waveforms from annotated time windows
- Bins and averages waveforms by experimental conditions
- Normalizes current traces

**Key Functions:**
- `make_sheets_dict()` - Loads annotation sheets with cell metadata
- `add_abfs()` - Associates ABF files with annotation sheets
- `make_waveforms()` - Extracts waveform segments
- `bin_wave()` - Bins waveforms for averaging

### 4. **Utilities** (`utils.py`)
- Shared configuration (file paths)
- Data loading helpers
- Standard plotting functions
- Statistical analysis helpers

## Data Files

### Annotations
- `abf_annotations_combined.csv` - Master annotation file with all VRB timing, frequencies, and tags
- `all_trace_names.csv` - List of all ABF trace names
- `summary_spikes2.csv` - Summary statistics for each cell (median/mean spiking rates, cell type, resistance)

### Processed Results
- `averaged_waveforms_by_freq_speed_signaltype_median.csv` - Waveforms averaged by experimental conditions
- `normalized_currents.csv` - Normalized current traces for comparison
- Various pickle files (`.pkl`) for intermediate data storage

## Workflow

### Basic Pipeline
1. **Load Data**: Read annotation CSVs and cell type metadata
2. **Associate ABF Files**: Link annotations with raw ABF electrophysiology files
3. **Extract Waveforms**: Pull waveform segments from bout windows
4. **Process & Analyze**: Calculate statistics, normalize, bin, and average
5. **Generate Outputs**: Create summary tables and visualizations

### Running the Analysis
```python
# See notebooks/main_pipeline.ipynb for complete workflow
import sys
sys.path.append('../scripts')
from waveforms import *
from vrb_analysis import *
from spike_analysis import *

# Set your local path
parent_folder_path = "/Users/Haley/Desktop/"

# Load cell metadata
cell_types_df = pd.read_csv("../data/annotations/summary_spikes2.csv")

# Create sheets dictionary
sheets = make_sheets_dict(sheet_names, parent_folder_path)

# Add ABF files
sheets = add_abfs(sheets, abfs_names, parent_folder_path_ABFS)

# Process and analyze...
```

## Key Experimental Variables

- **Cell Types**: Primary MiP, vSMN, CaP, etc.
- **Resistance Categories**: High Rin, Low Rin, Primary
- **Recording Types**: Cell-attached, Cell-attached (spiking), Excitatory, Inhibitory
- **Channels**: 
  - Channel 0: Motor neuron currents
  - Channel 1: Ventral root bursts (VRB)

## Dependencies

```python
matplotlib
pandas
numpy
pyabf
pickle
```

## Notes

- File paths are configured for local Mac environment (`/Users/Haley/Desktop/`)
- Original ABF files should be in `/Users/Haley/Desktop/ABF files annotated/`
- Standard padding for bout detection: 10ms (0.010s)
- Waveform extraction uses Chebyshev filtering and variance thresholding for burst detection

## Original Files
The consolidated scripts in this directory replace multiple iterative versions previously in "Cleaned, updated files from Spring":
- Multiple `vrb*.ipynb` notebooks (vrb1-7) → consolidated in `main_pipeline.ipynb`
- Multiple `run_all*.ipynb` notebooks (run_all1-5) → consolidated in `main_pipeline.ipynb`
- `waveforms_helpers.py` and `waveforms_helpers2.py` → consolidated in `waveforms.py`
- `cellattachedspikes.py` and `cellattachedspikes2.py` → consolidated in `spike_analysis.py`
