"""
Tools for working with and manipulating time.
"""

import numpy as np
import pandas as pd


def sampling_frequency_hz(df):
    """
    Sampling rate in Hz of the DatetimeIndex of the dataframe.
    """
    if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
        return None

    if isinstance(df.index[1], tuple):
        sampling_interval = df.index[1][-1] - df.index[0][-1]
    else:
        sampling_interval = df.index[1] - df.index[0]
    freq = 1.0 / sampling_interval.total_seconds()
    return freq


def total_seconds(df):
    """
    Total duration of data in seconds.
    """
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        total = df.index[-1] - df.index[0]
        return total.total_seconds()
    else:
        return None


def elapsed_time_seconds(df):
    """
    Array of elapsed times since the beginning of recording.
    """
    if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
        time_epoch = df.index.astype(np.int64)
        elapsed = (time_epoch - time_epoch[0]) * 1.0e-9
        return elapsed
    else:
        return None


def resample_df(df, new_samp_freq, ds_method='bfill', us_method='time'):
    """
    Resample a dataframe at a new sampling frequency.

    Intelligently resample a dataframe, applying decimation or interpolation
    depending on the need to down-sample or up-sample, respectively. If the
    new sample frequency exactly matches the current sampling frequency,
    return a copy of the input dataframe for consistent behavior.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to be resampled. Must have a uniformly spaced DatetimeIndex.
    new_samp_freq : float
        New sampling frequency.
    ds_method : string
        Down-sampling method. See pandas documentation for `asfreq` method for
        valid strings. ds_method is only applied when the DataFrame will be
        down-sampled. Default is backfill.
    us_method : string
        Up-sampling method. See pandas documentation for `interpolate` method
        for valid strings. us_method is only applied when the DataFrame will
        be up-sampled. Default is linear in time.

    Returns
    -------
    Resampled DataFrame
    """
    # Get current sampling frequency
    old_samp_freq = sampling_frequency_hz(df)
    # Set new sampling period.
    samp_offset = pd.tseries.offsets.Micro(n=round(1.0e6 / new_samp_freq))

    # Down-sampling case
    if new_samp_freq < old_samp_freq:
        resampled = df.asfreq(samp_offset, method=ds_method)
    # Up-sampling case
    elif new_samp_freq > old_samp_freq:
        resampled = df.resample(
            samp_offset, label='left', closed='left').mean()
        # For some reason, the resampled data is actually offset in time a
        # little bit from the original dataframe. It's an oddity of pandas.
        # Remove the offset.
        offset = resampled.index[0] - df.index[0]
        resampled.index = resampled.index - offset
        # Interpolation to fill missing values
        resampled = resampled.interpolate(method=us_method)
    # No resampling required because resampling freq exactly matches old
    # sampling freq
    else:
        resampled = df.copy()

    return resampled
