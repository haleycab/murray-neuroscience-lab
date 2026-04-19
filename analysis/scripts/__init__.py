"""
Murray Neuroscience Lab Analysis Pipeline

Core modules for processing electrophysiology data from zebrafish motor neurons.
"""

from .utils import (
    get_std_range,
    load_cell_metadata,
    load_trace_names,
    concat_one_channel,
    calculate_midpoints_and_frequencies,
    DEFAULT_PARENT_PATH
)

from .waveforms import (
    make_sheets_dict,
    add_abfs,
    make_waveforms,
    sheets_to_waveforms,
    bin_wave
)

from .spike_analysis import (
    split_cell_attached,
    split_cell_attached_spiking,
    calculate_spike_statistics
)

from .vrb_analysis import (
    extract_bout_windows,
    plot_motorneuron_activity,
    plot_ventralroot_bursts,
    plot_both_channels,
    calculate_vrb_frequencies
)

__all__ = [
    # Utils
    'get_std_range',
    'load_cell_metadata',
    'load_trace_names',
    'concat_one_channel',
    'calculate_midpoints_and_frequencies',
    'DEFAULT_PARENT_PATH',
    
    # Waveforms
    'make_sheets_dict',
    'add_abfs',
    'make_waveforms',
    'sheets_to_waveforms',
    'bin_wave',
    
    # Spike analysis
    'split_cell_attached',
    'split_cell_attached_spiking',
    'calculate_spike_statistics',
    
    # VRB analysis
    'extract_bout_windows',
    'plot_motorneuron_activity',
    'plot_ventralroot_bursts',
    'plot_both_channels',
    'calculate_vrb_frequencies',
]
