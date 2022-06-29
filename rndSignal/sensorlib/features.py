import copy
import numpy as np
import pandas as pd

from .wav import (cwt as trans, Morlet)
from .cache import memory
from .utils import progress_printer


def cwt_fft(signal, dt, wavelet=Morlet(), dj=1.0 / 12.0,
            low_freq=None, high_freq=None, use_coi=False):
    """
    Computes the continuous wavelet transform and FFT for signal.

    Also returns arrays of frequency and time indices that match
    the requested frequency range and coi.

    Parameters
    ----------
    signal : 1D array
        Signal on which to compute cwts.
    dt : float
        Sampling period of signal in seconds.
    wavelet : Wavelet class
        Wavelet function to use in cwt.
    dj: float
        Spacing between discrete scales. Smaller values will result in
        better scale resolution, but slower calculation and plotting.
    low_freq : float
        Lowest frequency to include in returned freq_idxs array.
    high_freq : float
        Highest frequency to include in returned freq_idxs array.
    use_coi : bool
        Use the coi to determine time_idxs unaffected by edge effects.

    Returns
    -------
    res : dictionary
        Full results of cwt call.
    freq_idxs : 1D array
        If low_freq or high_freq are not None, array of indices into
        res['freqs'] that match the frequency range specified. Otherwise
        freq_idxs contains all in the indices in res['freqs'].
    time_idxs : 1D array
        If coi is true, array of time indices into signal that reflect the
        section of data that is unaffected by the coi. Otherwise time_idxs
        contains all indices in the input signal.

    """
    res = trans(signal, dt=dt, wavelet=wavelet, dj=dj, result='dictionary')
    if low_freq is None:
        low_freq = res['freqs'][-1]
    if high_freq is None:
        high_freq = res['freqs'][0]
    freq_idxs = np.arange(res['freqs'].size)[
        np.logical_and(res['freqs'] >= low_freq, res['freqs'] <= high_freq)]
    if use_coi:
        time_idxs = np.arange(res['coi'].size)[
            (1. / res['coi']) < low_freq]
    else:
        time_idxs = np.arange(res['coi'].size)
    return res, freq_idxs, time_idxs


def _create_cwts_ffts(frames, column, dt, wavelet, low_freq, high_freq,
                      use_coi, print_updates=False):
    """
    Computes the continuous wavelet transform of each frame.
    """
    if print_updates:
        print_progress = progress_printer()
    else:
        def print_progress(*args):
            pass
    print_progress('starting')
    cwts = []
    cwts_index = []
    ffts = []
    ffts_index = []
    freq_idxs = None
    time_idxs = None
    freqs = None
    for i, (idx, frame) in \
            enumerate(frames.groupby(level=frames.index.names[:-1])):
        if i % 1000 == 0:
            print_progress('processing frame {}'.format(i))
        signal = frame.ix[idx][column]
        if freq_idxs is None:
            res, freq_idxs, time_idxs = cwt_fft(
                signal, dt=dt, wavelet=wavelet,
                low_freq=low_freq, high_freq=high_freq, use_coi=use_coi)
            freqs = res['freqs'][freq_idxs]
        else:
            res = trans(signal, dt=dt, wavelet=wavelet, result='dictionary')
        fft = pd.Series(np.abs(res['signal_ft']), index=res['ftfreqs'])
        ffts.append(fft)
        ffts_index.append(idx)
        cwt = pd.DataFrame(np.abs(res['W'][freq_idxs][:, time_idxs]).T,
                           columns=freqs)
        cwt['time'] = signal.index[time_idxs]
        cwts.append(cwt)
        cwts_index.extend([idx] * len(time_idxs))
    print_progress('concatenating cwts')
    cwts = pd.concat(cwts, axis=0)
    print_progress('setting multiindex on cwts')
    cwts.index = pd.MultiIndex.from_tuples(cwts_index,
                                           names=frames.index.names[:-1])
    print_progress('appending time to cwts multiindex')
    cwts.set_index('time', append=True, inplace=True)
    print_progress('concatenating ffts')
    ffts = pd.concat(ffts, axis=1).T
    print_progress('setting multiindex on ffts')
    ffts.index = pd.MultiIndex.from_tuples(ffts_index,
                                           names=frames.index.names[:-1])
    print_progress('ending')
    return cwts, ffts


@memory.cache
def _create_cwts_ffts_cached(frames, column, dt, wavelet, low_freq, high_freq,
                             use_coi):
    return _create_cwts_ffts(frames, column, dt, wavelet, low_freq, high_freq,
                             use_coi)


def create_cwts_ffts(frames, column='Am', dt=1. / 48, wavelet=Morlet(),
                     low_freq=0.5, high_freq=24., use_coi=True, cache=True):
    if cache:
        return _create_cwts_ffts_cached(frames, column, dt, wavelet, low_freq,
                                        high_freq, use_coi)
    else:
        return _create_cwts_ffts(frames, column, dt, wavelet, low_freq,
                                 high_freq, use_coi)


def create_cwts_rolling_mean(cwts, window=96, overlap_divisor=2, print_updates=False):
    """
    Splits cwt frames into windows of fixed width, overlapping by half the
    width, and returns a dataframe of the mean of each window.

    We do not cache the results since the input, cwts, appears to change with
    each run of create_cwts_ffts.
    """
    if print_updates:
        print_progress = progress_printer()
    else:
        def print_progress(*args):
            pass

    print_progress('starting')
    res = []
    index = []
    h = int(window / overlap_divisor)
    for i, (idx, frame) in \
            enumerate(cwts.groupby(level=cwts.index.names[:-1])):
        if i % 1000 == 0:
            print_progress('processing frame {}'.format(i))
        df = frame.ix[idx].rolling(window=window, center=True).mean()[h::h]\
            .dropna().reset_index(drop=True)
        res.append(df)

        if isinstance(idx, tuple):
            idxs = [idx + (j,) for j in range(len(df))]
        else:
            idxs = [(idx, j) for j in range(len(df))]

        index.extend(idxs)
    print_progress('concatenating')
    res = pd.concat(res, axis=0)
    print_progress('setting multiindex')
    res.index = pd.MultiIndex.from_tuples(
        index, names=cwts.index.names[:-1] + ['window'])
    print_progress('ending')
    return res


def create_cwts_cycle_mean(cwts, cycles, print_updates=False):
    """
    Split cwts at step cycle boundaries and compute the mean inside boundaries.
    """
    if print_updates:
        print_progress = progress_printer()
    else:
        def print_progress(*args):
            pass

    print_progress('starting')
    cwt_means = []
    frame_id_idx = cycles.index.names.index('frame_id')
    frame_levels = cycles.index.names[:frame_id_idx + 1]
    cwt_frame_ids = cwts.index.get_level_values(frame_id_idx)
    cycle_frame_ids = cycles.index.get_level_values(frame_id_idx)
    unique_cycle_frame_ids = cycle_frame_ids.unique()
    cycle_times = np.vstack((cycles.index.get_level_values(-2),
                             cycles.index.get_level_values(-1))).T
    for i, frame_id in enumerate(unique_cycle_frame_ids):
        if i % 1000 == 0:
            print_progress('processing {}th frame {}'.format(i, frame_id))
        cwt = cwts.iloc[cwt_frame_ids == frame_id].reset_index(frame_levels,
                                                               drop=True)
        for start_time, end_time in \
                cycle_times[np.where(cycle_frame_ids == frame_id)]:
            cwt_means.append(cwt[start_time:end_time].mean())
    cwt_means = pd.concat(cwt_means, axis=1).T
    # cwt_means.index = cycles.index.droplevel([-1, -2])
    cwt_means.index = cycles.index
    return cwt_means


def join_cwts_and_cycles(df_cwt_cycle_means, df_cycles):
    """
    Concatenate cwt cycle mean features with step cycle features into a
    single feature set.
    """
    cwt_cols = ['cwt{:02d}'.format(col) for col in
                range(df_cwt_cycle_means.shape[1])]
    cyc_cols = ['cyc{:02d}'.format(col) for col in
                range(df_cycles.shape[1])]
    cols = np.array(cwt_cols + cyc_cols)
    X = pd.concat((df_cwt_cycle_means, df_cycles), axis=1)
    X.columns = cols
    return X


def transform_df(transformer, df):
    transformer = copy.copy(transformer)
    return pd.DataFrame(transformer.transform(df), index=df.index)
