import numpy as np
import pandas as pd
import biosignalsnotebooks as bsnb
import neurokit2 as nk
from scipy import signal

# wavelets
import pywt
from pywt import swt, iswt, wavedec, waverec
from skimage.restoration import denoise_wavelet
from copy import deepcopy

# importing local functions
from rndSignal.plotting.signal_plots import fft_graph_compare, plot_signal


def filter_ecg(sig, fs, f1 = 0.5, f2 = 40, order = 3,
               use_filtfilter = True, powerline = True,
               use_wavelets = True, display_fft = False,
               display_signal = False):
    """Wrapper function for biosignalsnotebooks.bandpass with recommended defaults for ECG preprocessing. 
    It includes extra feature of removing powerline noise.
    
    Parameters
    ----------
    sig : array-like
        Signal
    fs : float
        Sampling frequency
    f1 : int
        The lower cutoff frequency
    f2 : int
        The upper cutoff frequency
    order : int
        Butterworth filter order
    use_filtfilt : boolean
        If True, the signal will be filtered once forward and then backwards.
        The result will have zero phase and twice the order chosen.
    powerline : bool
        If True, 60hz frequency will be removed via band-stop/notch filter.
    use_wavelets : bool
        If True, apply wavelet transform to denoise signal.
    display_fft : bool
        If True, graph FFT of raw signal and clean signal to compare effect of filters.
    display_signal : bool
        If True, raw and clean signals are plotted for comparison.

    Returns
    ----------
    filtered_signal : array-like
        Filtered signal
    """
    # bandpass filters
    filtered_signal = bsnb.bandpass(sig, f1, f2, order, fs, use_filtfilter)
    
    # band-stop filter
    if powerline == True:
        filtered_signal = nk.signal_filter(filtered_signal, sampling_rate = fs, method="powerline", powerline=60)
       
    # wavelet transform
    if use_wavelets == True:
        filtered_signal = wavelet_denoise_dwt(filtered_signal)
        
    # display FFT
    if display_fft == True:
        fft_graph_compare(sig, filtered_signal, fs, f1, f2)
        
    # display raw vs. clean signals
    if display_signal == True:
        plot_signal([sig, filtered_signal], [fs], labels=["raw", "clean"])
    
    return filtered_signal


def filter_ppg(sig, fs, f1 = 0.5, f2 = 4, order = 3,
               use_filtfilter = True, powerline = True,
               use_wavelets = False, display_fft = False,
               display_signal = False):
    """Wrapper function for biosignalsnotebooks.bandpass with recommended defaults for PPG preprocessing.
    
    Parameters
    ----------
    sig : array-like
        Signal
    fs : float
        Sampling frequency
    f1 : int
        The lower cutoff frequency
    f2 : int
        The upper cutoff frequency
    order : int
        Butterworth filter order
    use_filtfilt : boolean
        If True, the signal will be filtered once forward and then backwards.
        The result will have zero phase and twice the order chosen.
    powerline : bool
        If True, 60hz frequency will be removed via band-stop/notch filter.
    use_wavelets : bool
        If True, apply wavelet transform to denoise signal.
    display_fft : bool
        If True, graph FFT of raw signal and clean signal to compare effect of filters.
    display_signal : bool
        If True, raw and clean signals are plotted for comparison.
        
    Returns
    -------
    filtered_signal : array-like
        Filtered signal
    """
    # bandpass filters
    filtered_signal = bsnb.bandpass(sig, f1, f2, order, fs, use_filtfilter)
    
    # band-stop filter
    if powerline == True:
        filtered_signal = nk.signal_filter(filtered_signal, sampling_rate = fs, method="powerline", powerline=60)
        
    # wavelet transform
    if use_wavelets == True:
        filtered_signal = wavelet_denoise_swt(filtered_signal)
        
    # display FFT
    if display_fft == True:
        fft_graph_compare(sig, filtered_signal, fs, f1, f2)
        
    # display raw vs. clean signals
    if display_signal == True:
        rndSignal.plot_signal([sig, filtered_signal], [fs], labels=["raw", "clean"])
        
    return filtered_signal


def wavelet_denoise_swt(sig, wavelet="db4", level=7): # default settings for PPG
    """Denoises signals with stationary wavelet transform, thresholding, and signal reconstruction.
    Default parameters are recommended defaults for PPG processing.
    
    Credits to BiosignalsPLUX for their SWT code
    (https://biosignalsplux.com/learn/notebooks/Categories/Other/bvp_analysis_rev.php)
    
    Parameters
    ----------
    sig : array-like
        Signal
    wavelet : string
        The mother wavelet to use
    level : int
        The number of wavelet decomposition levels to use

    Returns
    -------
    rec_signal : array-like
        Denoised signal
    """
        
    # pad to ensure that the number of signal samples is a power of 2.
    sig_length = len(sig)
    sig = np.pad(sig, (0, 2**int(np.ceil(np.log2(len(sig)))) - len(sig)), "constant")

    # applying SWT
    swt_orig_coeffs = swt(sig, wavelet=wavelet, level=level)
    swt_coeffs_copy = deepcopy(swt_orig_coeffs)

    # calculating thresholds per level
    for lvl in range(0, level):
        thr_avg_dt = np.mean(swt_orig_coeffs[lvl][1])
        thr_avg_sc_low = np.mean(swt_orig_coeffs[lvl][0]) - 3 * np.std(swt_orig_coeffs[lvl][0])
        thr_avg_sc_high = np.mean(swt_orig_coeffs[lvl][0]) + 3 * np.std(swt_orig_coeffs[lvl][0])

        # applying calculated thresholds to coefficients
        for coeff_nbr in range(0, len(swt_orig_coeffs[lvl][1])):
            if swt_orig_coeffs[lvl][1][coeff_nbr] > thr_avg_dt: # Motion artifact coefficients.
                swt_orig_coeffs[lvl][1][coeff_nbr] = 0

            if swt_orig_coeffs[lvl][0][coeff_nbr] < thr_avg_sc_low or swt_orig_coeffs[lvl][0][coeff_nbr] > thr_avg_sc_high: # Motion artifact coefficients.
                swt_orig_coeffs[lvl][0][coeff_nbr] = 0
            else: # Storage of noise artifact coefficients in a separate list.
                swt_coeffs_copy[lvl][0][coeff_nbr] = 0
    
    # reconstructing the signal with thresholded coefficients
    rec_signal = iswt(swt_orig_coeffs, wavelet=wavelet)
    
    return rec_signal[:sig_length]


def wavelet_denoise_dwt(sig, wavelet="db4", level=8, mode="hard"): # default settings for ECG
    """Wrapper function for scikit-image's denoise_wavelet function with recommended defaults for ECG preprocessing.
    Applies discrete wavelet transform (DWT) to denoise signals.
    
    ----------
    Parameters
    ----------
    sig : array-like
        Signal
    wavelet : string
        The mother wavelet to use
    level : int
        The number of wavelet decomposition levels to use
    mode : string
        The type of thresholding to use

    Returns
    -------
    rec_signal : array-like
        Denoised signal
    """
    sig_length = len(sig)
    
    # pad to ensure that the number of signal samples is a power of 2.
    sig = np.pad(sig, (0, 2**int(np.ceil(np.log2(len(sig)))) - len(sig)), "constant")
    
    denoised = denoise_wavelet(sig, wavelet=wavelet, mode=mode, wavelet_levels=level)
    
    return denoised[:sig_length]


# def fft_graph(sig_raw, sig_clean, fs, f1=None, f2=None): # matplotlib
#     """
#     Supplementary function that plots the Fast Fourier Transform (FFT) of the raw and clean signals.
#     Additionally, the corner frequencies of the filter applied to the clean signal is shown.
    
#     ----------
#     Parameters
#     ----------
#     sig_raw (array-like) - raw signal
#     sig_clean (string) - clean signal
#     fs (int) - sampling rate of the signals
#     f1 (float) - high-pass filter cut-off
#     f2 (float) - low-pass filter cut-off

#     ----------
#     Returns
#     ----------
#     None
#     """
#     fft_raw = np.abs(np.fft.fft(sig_raw - np.mean(sig_raw)))
#     fft_clean = np.abs(np.fft.fft(sig_clean - np.mean(sig_clean)))

#     fs = np.fft.fftshift(np.fft.fftfreq(len(sig_raw),d=1/fs))

#     fft_raw = np.fft.fftshift(fft_raw)
#     fft_clean = np.fft.fftshift(fft_clean)

#     plt.figure(figsize=(14,8))
#     plt.subplot(211)
#     plt.plot(fs, fft_raw, linewidth=1, c='g', label="raw signal FFT")
#     plt.xlim(0,f2*1.1)
#     plt.title("Clean Signal FFT")
#     if f1 != None:
#         plt.axvline(x=f1, c='r', label="corner frequency")
#     if f2 != None:
#         plt.axvline(x=f2, c='r')
#     plt.legend()

#     plt.subplot(212)
#     plt.plot(fs, fft_clean, linewidth=1, c='b', label="raw signal FFT")
#     plt.xlim(0,f2*1.1)
#     plt.xlabel("Frequency Hz")
#     plt.title("Clean Signal FFT")
#     if f1 != None:
#         plt.axvline(x=f1, c='r', label="corner frequency")
#     if f2 != None:
#         plt.axvline(x=f2, c='r')
#     plt.legend()
#     plt.show()
