# Variance Binning Strategy for VRB Detection

## Overview

The `all_vrb_annotationsnoNaNs.csv` file was created using a **variance-based binning strategy** to automatically detect when voluntary rhythmic breathing (VRB) activity ends. This document describes the algorithm, implementation, and how the resulting annotations are used in the pipeline.

## Algorithm

### Core Concept

Instead of relying solely on manual annotation endpoints, the variance binning strategy analyzes the actual electrophysiological signal to detect where VRB activity terminates based on changes in signal variance.

### Detailed Steps

#### 1. Signal Extraction
- For each VRB bout marked by "bout start" and "bout end" manual annotations
- Extract the VRB channel data for the bout duration
- Compute between-bout variance (baseline noise level)

#### 2. Adaptive Binning
- Divide each bout into small time bins (default: 50 microsecond bins)
- Calculate signal variance in each bin
- This creates a time-series profile of activity levels throughout the bout

#### 3. Last-Crossing Detection (`adaptive_last_cross` function)
For each VRB annotation (start time):
- Scan from the annotation start time to the next midpoint
- Find the **LAST bin** where: `variance > (sigma × between_std)²`
- Start with `sigma = 1.0` (configurable)
- If no bins exceed threshold, decrease sigma iteratively by 0.1
- Fallback to minimum sigma if still no crossing found

Result: `LastCrossTime` = the time when VRB activity drops below the variance threshold

#### 4. Midpoint Calculation
```
BlackDotTime = (AnnotationTime + LastCrossTime) / 2
```
This midpoint is used as the center for phase interval extraction in waveform analysis.

## Key Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bin_width_s` | 0.00005 | Bin width in seconds (50 µs) |
| `sigma_multiplier` | 1.0 | Initial variance threshold multiplier |
| `sigma_step` | 0.1 | Step size when decreasing sigma |
| `min_sigma` | 0.0 | Minimum sigma to try |
| `pad_s` | 0.010 | Padding around bout boundaries (10 ms) |

## Output Columns

The resulting `all_vrb_annotationsnoNaNs.csv` contains:

| Column | Description |
|--------|-------------|
| `Trace name` | ABF file name |
| `ID` | Annotation unique identifier |
| `AnnotationTime` | Manual VRB start time (from human annotation) |
| `LastCrossTime` | Time when variance fell below threshold |
| `BlackDotTime` | (AnnotationTime + LastCrossTime) / 2 **← Used in pipeline** |
| `SigmaUsed` | Which sigma multiplier was needed to detect end |
| `VarThresholdUsed` | Actual variance threshold: (sigma × between_std)² |
| `AnnotationType` | "vrb" or "start" label |
| `Freq` | Stimulus frequency |
| `Type` | Annotation type (e.g., "VRB") |
| `Tags` | Manual tags/notes |
| Plus: `Currents Channel`, `VRB Channel`, `Mean Spiking`, `Median Spiking`, etc. |

## File Statistics

- **File**: `Cleaned, updated files from Spring/all_vrb_annotationsnoNaNs.csv`
- **Total rows**: 21,087 VRB events
- **Cells**: 31 recorded neurons across multiple dates
- **Created**: Iteratively through `vrb_clean*.ipynb` notebooks
- **Size**: ~4.2 MB

## Integration with Pipeline

### In the Refactored Pipeline

The `all_vrb_annotationsnoNaNs.csv` is loaded by `load_legacy_merged_annotations()`:

```python
def load_legacy_merged_annotations(parent_folder_path):
    """Load variance-based annotations (NO merging with Chebyshev)"""
    variance_path = os.path.join(
        parent_folder_path,
        "Cleaned, updated files from Spring",
        "all_vrb_annotationsnoNaNs.csv"
    )
    merged = pd.read_csv(variance_path)
    return merged
```

### Waveform Extraction

For each trace in the merged annotations:
1. Get the `BlackDotTime` value (variance-detected midpoint)
2. Use it to define phase intervals: `Phase ∈ [-0.5, 0.5]` centered at `BlackDotTime`
3. Extract waveform segment from the ABF file
4. Bin by phase into 100 equal bins

This symmetric phase binning ensures waveforms are centered on the detected VRB event, capturing the full rhythmic cycle.

## Why Variance-Based Detection?

### Advantages
1. **Objective**: Based on signal characteristics, not subjective human judgment
2. **Adaptive**: Automatically adjusts to different noise levels in different recordings
3. **Consistent**: Same algorithm applied uniformly across all 21,087 annotations
4. **Accurate**: Detects the actual end of activity, not arbitrary time points

### Comparison to Alternatives
- **Manual endpoints**: Subject to fatigue, inconsistency
- **Fixed time window**: Ignores actual signal dynamics
- **Simple threshold**: Doesn't account for between-bout variance

## Code Location

- **Algorithm implementation**: `Cleaned, updated files from Spring/vrb_plotbouts.py`
  - `adaptive_last_cross()` - Last-crossing detection
  - `compute_last_crossings()` - Applies to all annotations in a bout
  - `plot_bouts_like_lastcross2()` - Main analysis function

- **Pipeline integration**: `analysis/scripts/waveforms.py`
  - `load_legacy_merged_annotations()` - Loads the CSV
  - `calculate_midpoints_and_frequencies_avg()` - Processes for waveform extraction
  - `make_waveforms()` - Uses BlackDotTime to define phase intervals

## Reproducibility

To regenerate `all_vrb_annotationsnoNaNs.csv`:
1. Run `vrb_clean.ipynb` (loads ABF files and manual annotations)
2. For each cell, run `plot_bouts_like_lastcross2()` with variance parameters
3. Merge results with manual annotations
4. Drop rows with NaN in AnnotationTime or LastCrossTime
5. Save to CSV

However, the current file is pre-computed and available at:
```
Cleaned, updated files from Spring/all_vrb_annotationsnoNaNs.csv
```

No regeneration is necessary for pipeline execution.

---

**Last Updated**: April 19, 2026  
**Related Files**: 
- `analysis/scripts/waveforms.py` 
- `analysis/notebooks/main_pipeline.ipynb`
- `Cleaned, updated files from Spring/vrb_plotbouts.py`
