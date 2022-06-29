"""Submodule for the package"""

from .ecg_plotting import ecg_plot,ecg_plot_bpm
from .ppg_plotting import ppg_plot_quality, ppg_plot_beats, ppg_plot_bpm
from .signal_plots import * # all functions in this .py file

__all__ = [
    "ecg_plot",
    "ecg_plot_bpm",
    "ppg_plot_quality",
    "ppg_plot_beats",
    "ppg_plot_bpm",
    "plot_signal",
    "plot_filtered",
    "plot_segments",
    "plot_epoch",
    "plot_epoch_hr",
    "fft_graph_compare",
    "fft_graph",
    "plot_crosscorr",
    "plot_spectrogram",
    "plot_autocorrelation"
]
