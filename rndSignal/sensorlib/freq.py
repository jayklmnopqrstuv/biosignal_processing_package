"""
Tools for estimating frequency content.

"""

from __future__ import division, print_function

import numpy as np
import scipy.signal as signal


def fft_scaled(x, axis=-1, samp_freq=1.0, remove_mean=True):
    """
    Fully scaled and folded FFT with physical amplitudes preserved.

    Arguments
    ---------

    x: numpy n-d array
        array of signal information.

    axis: int
        array axis along which to compute the FFT.

    samp_freq: float
        signal sampling frequency in Hz.

    remove_mean: boolean
        remove the mean of each signal prior to taking the FFT so the DC
        component of the FFT will be zero.

    Returns
    --------

    (fft_x, freq) where *fft_x* is the full complex FFT, scaled and folded
    so that only positive frequencies remain, and *freq* is a matching
    array of positive frequencies.

    Examples
    --------

    A common use case would present the signals in a 2-D array
    where each row contains a signal trace. Columns would
    then represent time sample intervals of the signals. The rows of
    the returned *fft_x* array would contain the FFT of each signal, and
    each column would correspond to an entry in the *freq* array.

    """
    # Get length of the requested array axis.
    n = x.shape[axis]

    # Use truncating division here since for odd n we want to
    # round down to the next closest integer. See docs for numpy fft.
    half_n = n // 2

    # Remove the mean if requested
    if remove_mean:
        ind = [slice(None)] * x.ndim
        ind[axis] = np.newaxis
        x = x - x.mean(axis)[ind]

    # Compute fft, scale, and fold negative frequencies into positive.
    def scale_and_fold(x):
        # Scale by length of original signal
        x = (1.0 / n) * x[:half_n + 1]
        # Fold negative frequency
        x[1:] *= 2.0
        return x

    fft_x = np.fft.fft(x, axis=axis)
    fft_x = np.apply_along_axis(scale_and_fold, axis, fft_x)

    # Matching frequency array. The abs takes care of the case where n
    # is even, and the Nyquist frequency is usually negative.
    freq = np.fft.fftfreq(n, 1.0 / samp_freq)
    freq = np.abs(freq[:half_n + 1])

    return (fft_x, freq)


def correlate(x, y, mode):
    """
    Correlation between two vectors using a fast algorithm.

    """
    corr = signal.fftconvolve(x, y[::-1], mode=mode)
    return corr


def find_all_peaks_in_correlation(corr):
    """
    Find all peaks in a correlation function.

    """
    corr_pks = signal.argrelmax(corr, mode='clip')[0]
    return corr_pks


def find_dominant_peaks_in_correlation(corr):
    """
    Find the dominant and the next highest peak to the right of dominant.

    """
    corr_pks = find_all_peaks_in_correlation(corr)
    try:
        highest_corr_pk = corr_pks[np.argmax(corr[corr_pks])]
        pks_to_right = corr_pks[corr_pks > highest_corr_pk]
    except ValueError:
        highest_corr_pk = None
        next_highest_corr_pk = None
        return highest_corr_pk, next_highest_corr_pk
    try:
        next_highest_corr_pk = pks_to_right[np.argmax(corr[pks_to_right])]
    except ValueError:
        next_highest_corr_pk = None
    # print(highest_corr_pk, next_highest_corr_pk)
    return highest_corr_pk, next_highest_corr_pk


def fund_freq_by_corr(x, samp_freq=1.0, mode='same'):
    """
    Find the fundamental repetition rate in a signal using autocorrelation.

    """
    corr = correlate(x, x, mode=mode)
    highest, next_highest = find_dominant_peaks_in_correlation(corr)
    if highest is not None and next_highest is not None:
        fund_freq = float(samp_freq) / float(next_highest - highest)
    else:
        fund_freq = float(samp_freq) / len(x)
    return fund_freq
