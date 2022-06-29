"""
Implement step cycle segmentation as part of a
method for biometric identification using accelerometers.

Basic approach is to accept some segment of walking data of arbitrary
length. Divide that segment up into frames of about 10 seconds long
for step cycle segmentation. The 10 second window is a nice balance between
finding a good mean while still allowing the mean to wander gradually over
time.

Segments are assumed to arrive in the form of a pandas dataframe that
contains acceleration columns for 'Ax', 'Ay', 'Az', and 'Am' (magnitude) in
units of m/s^2. The dataframe must also have a DatetimeIndex at uniform
sampling frequency.

"""

from __future__ import print_function, division

import numpy as np
import pandas as pd
import scipy.signal as ss

from . import time as sbtime
from . import freq

import logging
logger = logging.getLogger(__name__)

# -----
# Step Cycle Segmentation
# -----


def frames_list(df, len_seconds=10):
    """
    Divide up a dataframe into a list of frames of certain duration.

    Parameters
    ----------
    df: DataFrame
        A DataFrame of walking data with DatetimeIndex as index.

    len_seconds: int
        An integer number of seconds for frame lengths

    Returns
    -------
    A list of pandas DataFrames each len_seconds long. List will be empty
    if input DataFrame is not long enough to contain a full frame.

    """
    d = pd.DateOffset(seconds=len_seconds)
    frames = []
    start = df.index[0]
    end = start + d
    while end < df.index[-1]:
        frames.append(df.ix[start:end])
        start = end
        end = start + d
    return frames


def smooth(x, sampling_freq_hz, corner_freq_hz=4.0):
    """
    Smooth step data with a low pass filter.

    Apply a 2nd order Butterworth filter forward and then backward
    through the step data to eliminate delay.

    Parameters
    ----------
    x: 1D numpy array
        Array containing the signal to be smoothed.
    sampling_freq_hz: float
        Sampling frequency of the signal in Hz.
    corner_freq_hz: float
        Corner frequency of the Butterworth filter in Hz.

    Returns
    -------
    filtered: 1D numpy array
        Array containing smoothed signal
    b, a: 1D numpy arrays
        Polynomial coefficients of the smoothing filter as returned from
        the Butterworth design function.

    """
    nyquist = sampling_freq_hz / 2.0
    f_c = np.array([corner_freq_hz, ], dtype=np.float64)  # Hz
    # Normalize by Nyquist
    f_c /= nyquist
    # Second order Butterworth low pass filter at corner frequency
    b, a = ss.butter(2, f_c, btype='low')
    # Apply the filter forward and backward to eliminate delay.
    filtered = ss.filtfilt(b, a, x)
    return (filtered, b, a)


def step_cycle_frequency(x, f_s, f_low=0.5, f_hi=1.5):
    """
    Determine the base step cycle frequency.

    Parameters
    ----------
    x: a 1D numpy array
        Array containing the signal, usually a smoothed version of it.
    f_s: float
        Sampling frequency of the signal in Hz.
    f_low: float
        Lowest expected fundamental rep frequency in Hz.
    f_high:
        Highest expected fundamental rep frequency in Hz.

    Returns
    -------
    cycle_freq: float
        Estimated frequency of full step cycles in Hz.

    """
    # Determine fundamental rep frequency
    fund_freq = freq.fund_freq_by_corr(x, f_s)
    # If fundamental rep frequency is higher than f_hi Hz then the fundamental
    # rep frequency likely represents single steps rather than full cycles.
    # If fundamental req frequency is lower than f_low Hz then the fundamental
    # has found something much slower than normal walking.
    if fund_freq > f_hi:
        cycle_freq = fund_freq / 2.0
    elif fund_freq < f_low:
        cycle_freq = fund_freq * 2.0
    else:
        cycle_freq = fund_freq
    return cycle_freq


def salience(x, maximum=True, clamp_to=None):
    """
    Compute the salience vector for a signal.

    The kth entry in the salience vector contains the minimum of these:
        * the number of array indices to the closest value in the array
          that is either greater than (if *maximum=True* or less than
          (if *maximum=False*) the value at the kth entry in the signal
          array, depending on the *maximum* parameter.
        * the value of `clamp_to`. If `clamp_to` is None, then `clamp_to`
          defaults to len(x)/2.

    By definition, the first and last value of the salience vector are 1
    since it is not possible to discern a peak at the endpoints of the signal.

    Parameters
    ----------
    x: 1D numpy array.
        The signal on which to compute salience.
    maximum: bool
        If True compute the max salience vector. If False compute
        the min salience vector.
    clamp_to: int
        Sets the largest returned salience value.

    Returns
    -------
    out: 1D numpy array.
        Salience vector.

    """
    if clamp_to is None:
        clamp_to = len(x) / 2
    out = np.ones_like(x, dtype=np.int)

    # sort the list to make comparing easier
    min_idx = np.argsort(x)
    if not maximum:
        min_idx = min_idx[::-1]

    # the last one would have the maximum salience
    out[min_idx[-1]] = clamp_to

    for i, idx in enumerate(min_idx[:-1]):
        start = i + 1
        # ignore equal values
        while x[idx] == x[min_idx[start]]:
            start += 1
            # if this is the end of the list all values should have the maximum salience
            if start == len(x):
                out[min_idx[i:]] = clamp_to
                out[0] = out[-1] = 1
                np.clip(out, 0, clamp_to, out=out)
                return out
        # calculate the salience
        out[idx] = np.min(np.abs(idx - min_idx[start:]))

    # By definition, first and last values of salience should be 1.
    out[0] = out[-1] = 1
    np.clip(out, 0, clamp_to, out=out)
    return out


def zero_crossings(x):
    """
    Find the indices of the zero crossings in the signal.

    Signal must be smooth to begin with. Returns None if no zero crossings
    are located.

    Parameters
    ----------
    x: 1D numpy array
        A somewhat smooth signal.

    Returns
    -------
    zc: 1D numpy array
        Indicies of the zero crossings.

    """
    zc = np.where(np.diff(np.sign(x)))[0]
    if len(zc) < 1:
        zc = None
    return zc


def centroids(y):
    """
    Find the centroids vector for a signal.

    Centroids vector is nominally zero but has spikes (both positive and
    negative) at the locations of area centroids between zero crossings.
    The height of the peaks reflects the area under the curve between zero
    crossings.

    Parameters
    ----------
    y: a 1D numpy array
        A somewhat smooth signal

    Returns
    -------
    out: 1D numpy array
        Centroids vector

    """
    out = np.zeros_like(y)
    zc = zero_crossings(y)
    if zc is not None:
        for i, c in enumerate(zc[:-1]):
            start = c
            end = zc[i + 1] + 1
            this_area = np.trapz(y[start:end])
            x = np.arange(start, end)
            this_w_area = np.trapz(x * y[start:end])
            this_centroid = int(float(this_w_area) / float(this_area))
            this_centroid = np.clip(this_centroid, start, end - 1)
            out[this_centroid] = this_area
    return out


def high_salient_peaks(max_sal, centroid, cycle_len, threshold=1, return_sals=False):
    """
    Find the indices of the strongest peaks using salience and centroids.

    Step data generally has a pattern of alternating peak heights that
    follow left foot and right foot steps. This function tries to find
    the higher peaks in the alternating pattern.

    Parameters
    ----------
    max_sal: 1D numpy array
        A max salience vector
    centroid: 1D numpy array
        A centroids vector
    cycle_len: int
        Length of the step cycle in samples
    threshold: int
        Peaks with height less than or equal to threshold are ignored
    return_sals: bool
        If true the location of nearest salient peaks will be returned instead of centroids

    Returns
    -------
    best_run: list
        Indices of the strongest signal peaks

    """
    # For smooth data, salience is generally 1 for all but the peaks.
    # Compare against a user supplied threshold to eliminate small peaks.
    max_sal_peak_idxs = np.where(max_sal > threshold)[0]
    max_cen_peak_idxs = np.where(centroid > threshold)[0]

    # Search through all centroids peaks as possible candidates for starting
    # segmentation. Look for more peaks in a window that corresponds to
    # cycle_len. Score candidate cycle boundaries by the sum of peak heights.
    # Pick the best scoring boundary group.

    window = int(cycle_len * 0.20)

    run_weight = 0
    best_idx = None
    best_run = []
    # calculate the nearest salient peak and weight of each node
    nearest_sals = {idx: max_sal_peak_idxs[np.abs(
        max_sal_peak_idxs - idx).argmin()] for idx in max_cen_peak_idxs}
    node_weights = {idx: max_sal[nearest_sals[idx]] + centroid[idx]
                    for idx in max_cen_peak_idxs}

    # dicts for memoization of the results
    next_nodes = {}
    previous_nodes = {}
    forward_costs = {None: 0}
    backward_costs = {None: 0}

    # functions to calculate cost of each path recursively with memoization
    def get_fwd_cost(node_idx):
        next_idx = next_nodes[node_idx]
        if next_idx in forward_costs:
            return forward_costs[next_idx]
        forward_costs[next_idx] = node_weights[next_idx] + get_fwd_cost(next_idx)
        return forward_costs[next_idx]

    def get_bckwd_cost(node_idx):
        prev_idx = previous_nodes[node_idx]
        if prev_idx in backward_costs:
            return backward_costs[prev_idx]
        backward_costs[prev_idx] = node_weights[prev_idx] + get_bckwd_cost(prev_idx)
        return backward_costs[prev_idx]

    # iterate once through all and fill the forward and backward paths at each node
    for idx in max_cen_peak_idxs:
        if idx + cycle_len + window > len(centroid) or not np.any(max_cen_peak_idxs > idx):
            next_nodes[idx] = None
        else:
            peaks_in_range = max_cen_peak_idxs[np.logical_and(
                max_cen_peak_idxs >= (idx + cycle_len - window),
                max_cen_peak_idxs <= (idx + cycle_len + window))]
            if peaks_in_range.size > 0:
                next_peak = peaks_in_range[np.argmax(centroid[peaks_in_range])]
            else:
                # Find closest peak ahead
                peaks_ahead = max_cen_peak_idxs[max_cen_peak_idxs > idx]
                nearest_peak_idx = np.abs(peaks_ahead - (idx + cycle_len)).argmin()
                next_peak = peaks_ahead[nearest_peak_idx]
            next_nodes[idx] = next_peak
        if idx not in previous_nodes:
            if idx - cycle_len - window < 0 or not np.any(max_cen_peak_idxs < idx):
                previous_nodes[idx] = None
            else:
                peaks_in_range = max_cen_peak_idxs[np.logical_and(
                    max_cen_peak_idxs >= (idx - cycle_len - window),
                    max_cen_peak_idxs <= (idx - cycle_len + window))]
                if peaks_in_range.size > 0:
                    prev_peak = peaks_in_range[np.argmax(centroid[peaks_in_range])]
                else:
                    # Find closest peak behind
                    peaks_behind = max_cen_peak_idxs[max_cen_peak_idxs < idx]
                    nearest_peak_idx = np.abs(peaks_behind - (idx - cycle_len)).argmin()
                    prev_peak = peaks_behind[nearest_peak_idx]
                previous_nodes[idx] = prev_peak

    # get the total cost of each path with memoization
    for idx in max_cen_peak_idxs:
        this_run = get_bckwd_cost(idx) + get_fwd_cost(idx) + node_weights[idx]
        if this_run > run_weight:
            run_weight = this_run
            best_idx = idx

    if not best_idx:
        return []

    # construct the best path
    p_node = previous_nodes[best_idx]
    while p_node:
        best_run.append(p_node)
        p_node = previous_nodes[p_node]
    best_run.reverse()
    best_run.append(best_idx)
    n_node = next_nodes[best_idx]
    while n_node:
        best_run.append(n_node)
        n_node = next_nodes[n_node]

    # return salient peaks if needed
    if return_sals:
        return [nearest_sals[node] for node in best_run]
    else:
        return best_run


def step_cycle_details(step_data, sampling_freq_hz, corner_freq_hz=4.0,
                       peaks=True):
    """
    Intermediate signal details for step cycle segmentation.

    This function is useful for accessing all the underlying details that
    resulted in a particular frame segmentation.

    Parameters
    ----------
    step_data : 1D numpy array
        Raw accelerometer magnitude from walking

    sampling_freq_hz : float
        Sampling frequency of raw data in Hz

    corner_freq_hz : float
        Corner frequency of the Butterworth smoothing filter in Hz

    peaks : bool
        if True, segment at peaks. Otherwise segment at valleys.

    Returns
    -------
    ret: dict
        Dictionary containing these keys:
        - 'mr': 1D numpy array. Mean removed signal
        - 'smoothed': 1D numpy array. Smoothed signal
        - 'cycle_freq': float. Full step cycle frequency in Hz
        - 'max_sal': 1D numpy array. Salience vector
        - 'centroids': 2D numpy array. Centroids vector
        - 'high_peaks': list. Indicies of the strongest signal peaks

    """
    # Remove the mean
    mr = step_data - step_data.mean()
    # Smooth
    smoothed, _, _ = smooth(mr, sampling_freq_hz, corner_freq_hz)
    # Compute step cycle frequency
    cycle_freq = step_cycle_frequency(smoothed, sampling_freq_hz)
    # Compute salience and centroids vectors
    if peaks:
        max_sal = salience(smoothed,
                           clamp_to=int(sampling_freq_hz / cycle_freq))
        cent = centroids(smoothed)
    else:
        max_sal = salience(-smoothed,
                           clamp_to=int(sampling_freq_hz / cycle_freq))
        cent = centroids(-smoothed)
    cycle_len = int((1.0 / cycle_freq) * sampling_freq_hz)
    high_peaks = high_salient_peaks(max_sal, cent, cycle_len)
    ret = {'mr': mr,
           'smoothed': smoothed,
           'cycle_freq': cycle_freq,
           'max_sal': max_sal,
           'centroids': cent,
           'high_peaks': high_peaks}
    return ret


def step_cycle_boundaries(step_data, sampling_freq_hz, corner_freq_hz=4.0,
                          peaks=True):
    """
    Find the step cycle boundaries from a frame of step_data.

    Apply the full step cycle segmentation process beginning from an array
    of accelerometer magnitude data.

    Parameters
    ----------
    step_data : 1D numpy array
        Roughly 10 seconds of accelerometer magnitude step data

    sampling_freq_hz : float
        Sampling frequency of raw data in Hz

    corner_freq_hz : float
        Corner frequency of the Butterworth smoothing filter in Hz

    peaks : bool
        if True, segment at peaks. Otherwise segment at valleys.

    Returns
    -------
    ret: dict
        Dictionary containing these keys:
        - 'cycle_boundary_idxs': list. Indicies of step cycle boundaries
        - 'num_cycles': int. Number of step cycles identifies in this frame
        - 'cycle_lengths_sec': 1D numpy array. Length of each cycle in seconds
        - 'cycle_freq': float. Full step cycle frequency in Hz

    """
    details = step_cycle_details(
        step_data, sampling_freq_hz, corner_freq_hz, peaks)
    cycle_freq = details['cycle_freq']
    high_peaks = details['high_peaks']
    # Step cycle boundaries are at high salient peaks
    cycle_boundary_idxs = high_peaks
    if cycle_boundary_idxs:
        # Compute step cycle stats for the frame
        num_cycles = len(cycle_boundary_idxs) - 1
        cycle_lengths_sec = np.diff(cycle_boundary_idxs) / sampling_freq_hz
    else:
        num_cycles = 0
        cycle_lengths_sec = np.empty(0)
    ret = {'cycle_boundary_idxs': cycle_boundary_idxs,
           'num_cycles': num_cycles,
           'cycle_lengths_sec': cycle_lengths_sec,
           'cycle_freq': cycle_freq}
    return ret


def step_cycle_boundaries_from_peaks_zc(peaks, zero_crossings):
    """
    Find the step cycle boundaries from the arrays of high salient peaks and
    zero crossing indices provided.

    Parameters
    ----------
    peaks: 1D numpy array
        Indices of strongest signal peaks
    zero_crossings: 1D numpy array
        Indices of zero crossings

    Returns
    -------
    cycle_boundary_idxs: 1D numpy array. Indicies of step cycle boundaries

    """
    boundaries = np.searchsorted(zero_crossings, peaks)
    boundaries = boundaries.compress(boundaries < len(zero_crossings))

    cycle_boundary_idxs = zero_crossings[boundaries]
    return cycle_boundary_idxs


def cycles_list(df, cycle_boundaries):
    """
    Return a list of dataframes containing individual step cycles.

    Parameters
    ----------
    df: DataFrame
        A frame of accelerometer data to segment into step cycles
    cycle_boundaries: 1D numpy array
        Indicies of step cycle boundaries

    Returns
    -------
    cycles: list. List of DataFrames, one for each identified step cycle

    """
    cycles = []
    for i, cycle_start in enumerate(cycle_boundaries[:-1]):
        cycles.append(df.ix[cycle_boundaries[i]:cycle_boundaries[i + 1]])
    return cycles


def elapsed_times_list(cycles_list):
    """
    Return a list of elapsed time arrays that match the dataframes in cycles_list.

    Parameters
    ----------
    cycles_list: a list of DataFrames
        Each DataFrame contains a full step cycle

    Returns
    -------
    times: list
        A list of elaped time vectors for each DataFrame in cycles_list

    """
    times = [sbtime.elapsed_time_seconds(df) for df in cycles_list]
    return times


def cycle_datetimes_list(cycles_list):
    """
    Get start and end datetimes for all cycles in cycles_list.

    Parameters
    ----------
    cycles_list : a list of DataFrames
        Each DataFrame contains a full step cycle.

    Returns
    -------
    dt_list : list
        A list of tuples of the form (cycle_start_datetime, cycle_end_datetime).

    """
    dt_list = []
    for cycle in cycles_list:
        start_dt = cycle.index[0]
        end_dt = cycle.index[-1]
        dt_list.append((start_dt, end_dt))
    return dt_list


# -----
# Step Cycle Normalization
# -----


def normalize_length(times, cycles, num_samples=100):
    """
    Return a list of step cycles normalized in length to be num_samples long.

    Parameters
    ----------
    times: list
        List of elapsed time vectors associated with step cycles
    cycles: list
        A list of DataFrames containing step cycles
    num_samples: int
        Number of samples to include in normalized cycles

    Returns
    -------
    out: list
        A list of 1D numpy arrays containing length normalized step cycles

    """
    out = []
    for time, cycle in zip(times, cycles):
        t_new = np.linspace(time[0], time[-1], num=num_samples)
        interp = np.interp(t_new, time, cycle['Am'].values)
        out.append(interp)
    return out


def normalize_amplitude(cycles):
    """
    Return a list of step cycles normalized in amplitude.

    Parameters
    ----------
    cycles: list
        A list of 1D numpy arrays containing step cycles

    Returns
    -------
    out: list
        A list of 1D numpy arrays contining amplitude normlized step cycles

    """
    out = []
    for cycle in cycles:
        out.append((cycle - np.mean(cycle)) / np.std(cycle, ddof=0))
    return out


def normalized_cycles(frames, num_samples=100, verbose=False, cycle_dt=False,
                      peaks=True, corner_freq_hz=4.0):
    """
    Return a 2D array of normalized step cycles from a list of frames.

    Each array row is a fully normalized step cycle.

    Parameters
    ----------
    frames : list
        List of DataFrames containing a fixed window of step
        data (on the order of 10 seconds) in which step cycles are
        to be located and normalized.

    num_samples : int
        Number of samples to include in normalized cycles.

    verbose : bool
        If True, print out diagnostics on removal of outlier step cycles.

    cycle_dt : bool
        If True, return start and end datetimes for each cycle.

    peaks : bool
        if True, segment at peaks. Otherwise segment at valleys.

    corner_freq_hz : float
        Corner frequency of the Butterworth smoothing filter in Hz

    Returns
    -------
    normed_cycles : 2D numpy array
        Array of step cycles. Each row is a cycle that is num_samples long.

    cycle_dt_list : list
        If cycle_dt is True, also return a list of tuples of the form
        (cycle_start_datetime, cycle_end_datetime) for cycles in normed_cycles.

    """
    normed_cycles = []
    cycle_dt_list = []
    for i, frame in enumerate(frames):
        f_s = sbtime.sampling_frequency_hz(frame)
        details = step_cycle_boundaries(
            frame['Am'].values, f_s, corner_freq_hz=corner_freq_hz, peaks=peaks)
        cycle_boundaries = details['cycle_boundary_idxs']
        if cycle_boundaries:
            cycle_freq = details['cycle_freq']
            cycle_lengths_sec = details['cycle_lengths_sec'].tolist()
            cycles = cycles_list(frame, cycle_boundaries)
            # Drop cycles that are too long or too short
            ideal_cycle_period = 1.0 / cycle_freq
            window = ideal_cycle_period * 0.2
            start_len = len(cycles)

            def in_window(cycle_len_sec):
                res = ((cycle_len_sec > (ideal_cycle_period - window)) and
                       (cycle_len_sec < (ideal_cycle_period + window)))
                return res

            cycles = [cycle for cycle, cl in zip(cycles, cycle_lengths_sec) if
                      in_window(cl)]
            removed = start_len - len(cycles)
            if removed and verbose:
                logger.info(
                    'Removed {} of {} cycles at frame {}'.format(
                        removed, start_len, i))
            # Interpolate down to num_samples
            times = elapsed_times_list(cycles)
            interp = normalize_length(times, cycles, num_samples)
            normed = normalize_amplitude(interp)
            normed_cycles += normed
            if cycle_dt:
                frame_cycle_dt = cycle_datetimes_list(cycles)
                cycle_dt_list += frame_cycle_dt
        else:
            continue
    if cycle_dt:
        return np.array(normed_cycles), cycle_dt_list
    else:
        return np.array(normed_cycles)


def average_cycle(cycles):
    """
    Return the average cycle from a 2D array of normalized step cycles.

    Parameters
    ----------
    cycles: 2D numpy array
        Array of step cycles where each row is a normalized step cycle.

    Returns
    -------
    mean: 1D numpy array
        The average step cycle

    """
    arr = np.array(cycles)
    return arr.mean(axis=0)


# -----
# Distance Measurement
# -----


def distance_between_cycles(cycles, reference_cycle):
    """
    Compute the distance between a 2D array of cycles and a reference cycle.

    Euclidean distance between vectors is used.

    Parameters
    ----------
    cycles: 2D numpy array
        Array of normalized step cycles
    reference_cycle: 1D numpy array
        A reference cycle against which to compare the cycles

    Returns
    -------
    distance: 1D numpy array
        Array of distances between cycles and the reference cycle

    """
    distance = np.linalg.norm(cycles - reference_cycle, axis=1)
    return np.array(distance, dtype=np.float64)


# -----
# Outlier Detection and Cleaning
# -----


def delete_outlier_cycles(cycles, reference_cycle, num_stdev=2.0):
    """
    Return a new array of cycles with outlier cycles removed.

    Input array of cycles is not modified.

    Parameters
    ----------
    cycles: 2D numpy array
        Normalized step cycles
    reference_cycle: 1D numpy array
        A reference cycle against which to compare the cycles
    num_stdev: float
        Number of standard deviations above which outlier cycles will
        will be removed from returned result.

    Returns
    -------
    cycles_cleaned: 2D numpy array
        Array of clean step cycles

    """
    dist = distance_between_cycles(cycles, reference_cycle)
    mean_dist = np.mean(dist)
    stdev_dist = np.std(dist, ddof=1)
    threshold = mean_dist + num_stdev * stdev_dist
    idxs_to_drop = np.where(dist > threshold)[0]
    cycles_cleaned = np.delete(cycles, idxs_to_drop, axis=0)
    return cycles_cleaned


# -----
# Automation
# -----


def process_frame(df, num_samples=100, verbose=False, cycle_dt=False,
                  peaks=True, corner_freq_hz=4.0):
    """
    Run the step cycle segmentation process on a single frame of data.

    Parameters
    ----------
    df : pandas DataFrame
        A single frame of step data containing at least an Am column.
        The index must be a DatetimeIndex.

    num_samples : int
        Number of samples to use in normalized cycles.

    cycle_dt : bool
        If True, return start and end datetimes for each cycle.

    peaks : bool
        if True, segment at peaks. Otherwise segment at valleys.

    corner_freq_hz : float
        Corner frequency of the Butterworth smoothing filter in Hz

    Returns
    -------
    normed_cycles : 2D numpy array
        An array of normalized step cycles. Each row is a step cycle that
        is num_samples long.

    cycle_dt_list : list
        If cycle_dt is True, also return a list of tuples of the form
        (cycle_start_datetime, cycle_end_datetime) for cycles in normed_cycles.

    """
    ret = normalized_cycles([df], num_samples, verbose, cycle_dt, peaks,
                            corner_freq_hz=corner_freq_hz)
    return ret


def process_segment(df, len_seconds=10, num_samples=100, num_stdev=2.0,
                    verbose=False, cycle_dt=False, peaks=True, corner_freq_hz=4.0):
    """
    Run the entire fingerprint creation process on a segment of step data.

    Return both raw normalized results and cleaned results after applying
    outlier removal rules.

    Parameters
    ----------
    df : pandas DataFrame
        A segment of step data.

    len_seconds : int
        The window length for frame sizes used in breaking up step data.

    num_samples : int
        Number of samples to use in normalized step cycles.

    num_stdev : float
        Number of standard deviations to use in outlier rejection.

    cycle_dt : bool
        If True, return start and end datetimes for each cycle.

    peaks : bool
        if True, segment at peaks. Otherwise segment at valleys.

    corner_freq_hz : float
        Corner frequency of the Butterworth smoothing filter in Hz

    Returns
    -------
    cycles : tuple
        A tuple containing:
            a 2D array of normalized step cycles.
            a 1D array of containing the average normalized step cycle.
            a 2D array of cleaned step cycles (after outlier cycle removal).
            a 1D array of containing the average clean step cycle.
        The tuple contains four None entries if no frames are found in the
        segment.

    cycle_dt_list : list
        If cycle_dt is True and normed_cycles contains cycles, also return
        a list of tuples of the form (cycle_start_datetime, cycle_end_datetime)
        for cycles in normed_cycles.

    """
    frames = frames_list(df, len_seconds)
    if not frames:
        return (None, None, None, None)
    else:
        ret = normalized_cycles(frames, num_samples, verbose, cycle_dt, peaks,
                                corner_freq_hz=corner_freq_hz)
        if cycle_dt:
            normed_cycles, cycle_dt_list = ret
        else:
            normed_cycles = ret
        # If no cycles were located, bail out now.
        if normed_cycles.size == 0:
            return (None, None, None, None)
        ave_normed_cycle = average_cycle(normed_cycles)
        cleaned_cycles = delete_outlier_cycles(
            normed_cycles, ave_normed_cycle, num_stdev)
        ave_cleaned_cycle = average_cycle(cleaned_cycles)
        if cycle_dt:
            return (normed_cycles, ave_normed_cycle,
                    cleaned_cycles, ave_cleaned_cycle), cycle_dt_list
        else:
            return (normed_cycles, ave_normed_cycle,
                    cleaned_cycles, ave_cleaned_cycle)


def _tuplify(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x,)


def process_dataset(df, resampling_freq_hz=None, resampling_method='bfill',
                    num_samples=100, verbose=False, cycle_dt=False, peaks=True,
                    corner_freq_hz=4.0, interp_method='time'):
    """
    Run the entire fingerprint creation process on a full dataset.

    Useful for extracting step cycles from train, test, or validate datasets.

    Parameters
    ----------
    df : pandas DataFrame
        A dataset derived from a call to `datasets.load_n_split`
    resampling_freq_hz : float or None
        Frequency in Hz at which to resample the time data in df. Set
        this to None to use the device native sampling rate.
    resampling_method : string
        Resampling fill method. Only applies when down-sampling is required.
        Method used to fill missing points during resampling. See
        `DataFrame.asfreq` documentation for valid method strings.
    num_samples : int
        Number of samples to use in length normalization of time data in df.
    verbose : boolean
        Print details of segmentation process.
    cycle_dt : bool
        If True, include start and end datetimes in the index for each cycle.
    peaks : bool
        if True, segment at peaks. Otherwise segment at valleys.
    corner_freq_hz : float
        Corner frequency of the Butterworth smoothing filter in Hz
    interp_method : string
        Resampling interpolation method. Only applies when up-sampling is
        required. See `DataFrame.interpolate` documentation for valid method
        strings.

    Returns
    -------
    cycles_df : pandas DataFrame
        A dataframe of step cycles for the data in df.
    """
    index_names = []
    all_cycles = []
    up_to_frames = df.index.names[:-1]
    for details, frame_df in df.groupby(level=up_to_frames):
        details = _tuplify(details)
        frame_df = frame_df.loc[details, :]
        # Resample if needed
        if resampling_freq_hz is not None:
            frame_df = sbtime.resample_df(
                frame_df, resampling_freq_hz,
                ds_method=resampling_method, us_method=interp_method)
        # Process frames
        ret = process_frame(
            frame_df, num_samples=num_samples, verbose=verbose,
            cycle_dt=cycle_dt, peaks=peaks, corner_freq_hz=corner_freq_hz)
        if cycle_dt:
            this_cycles, cycle_dt_list = ret
        else:
            this_cycles = ret
        # Bail on this frame if no cycles
        if this_cycles.size == 0:
            continue
        # Build up index
        if cycle_dt:
            this_index_names = [details + (i, dt[0], dt[1])
                                for i, dt in enumerate(cycle_dt_list)]
        else:
            this_index_names = [details + (i,)
                                for i in range(this_cycles.shape[0])]
        index_names.extend(this_index_names)
        all_cycles.append(this_cycles)
    all_cycles = np.vstack(all_cycles)

    if cycle_dt:
        cycles_index = pd.MultiIndex.from_tuples(
            index_names,
            names=up_to_frames + ['cycle_id', 'cycle_start', 'cycle_end'])
    else:
        cycles_index = pd.MultiIndex.from_tuples(
            index_names, names=up_to_frames + ['cycle_id'])
    cycles_df = pd.DataFrame(all_cycles, cycles_index)

    return cycles_df


# -----
# Step Cycle Segmentation again
# -----

def get_valleys(sample, order=1):
    """
    Get all valleys of a numerical data sample, including the left and right
    end points of valley lowlands.

    Parameters
    ----------
    sample: numpy.ndarray
        Numerical data sample
    order: int (optional)
        Passed to scipy.signal.argrelmin

    Returns
    -------
    valleys: numpy.ndarray
        Indexes of valleys in sample
    """
    # get relative minima
    valleys = ss.argrelmin(sample, order=order)[0]
    # augment these with long valleys (which argrelmin ignores)
    sample_diff = np.diff(sample)
    lowlands = np.where(sample_diff == 0.)[0]
    if len(lowlands) == 0:
        return valleys
    new_valleys = []
    eidx = 0
    for idx in lowlands:
        if idx < eidx:  # might have already checked idx in while loop below
            continue
        sidx = idx
        eidx = idx
        while eidx + 1 in lowlands:
            eidx += 1
        # don't set eidx = eidx + 1 here, since we need to preserve the value
        # of eidx for the if idx < eidx check in the next for loop iteration
        if (sidx == 0 or sample_diff[sidx - 1] < 0) and \
                (eidx + 1 == len(sample_diff) or sample_diff[eidx + 1] > 0):
            new_valleys.append(sidx)
            new_valleys.append(eidx + 1)
    valleys = np.append(valleys, new_valleys)
    valleys.sort()
    return valleys


def _diff_var(valleys, period):
    """
    Calculate mean variation about period.

    Paramters
    ---------
    valleys: array-like
        Indices of valleys
    period: int
        Expected mean of the difference in valleys

    Returns
    -------
    diff_var: float
        Variation about period
    """
    return ((np.diff(valleys) - period) ** 2).mean()


def add_valleys(valleys, valleys_new, period):
    """
    Greedily add new valleys to current list of valleys if doing so reduces
    diff_var

    Parameters
    ----------
    valleys: numpy.ndarray
        Indexes of valleys in sample
    valleys_new: numpy.ndarray
        Indexes of candidate valleys in sample
    period: int
        Expected period in number of data points

    Returns
    -------
    valleys: numpy.ndarray
        (Possibly updated) indexes of valleys in sample
    """
    # add valley_new if doing so reduces diff_var
    valleys_new = np.setdiff1d(valleys_new, valleys)
    while True:
        cands = []
        dv_vals = _diff_var(valleys, period)
        for valley_new in valleys_new:
            valleys_added = np.append(valleys, [valley_new])
            valleys_added.sort()
            dv_vals_add = _diff_var(valleys_added, period)
            # FIXME 0.95 is a magic number, the idea is that we don't add
            # valleys unless they change the diff_var significantly
            if dv_vals_add < 0.95 * dv_vals:
                cands.append([valley_new, dv_vals_add])
        if len(cands) == 0:
            break
        cands = np.array(cands)
        valley_new = int(cands[np.argmin(cands[:, 1]), 0])
        valleys = np.append(valleys, [valley_new])
        valleys.sort()
        valleys_new = valleys_new[valleys_new != valley_new]
    return valleys


def remove_valleys(valleys, period):
    """
    Greedily remove valleys from current list of valleys if doing so reduces
    diff_var

    Parameters
    ----------
    valleys: numpy.ndarray
        Indexes of valleys in sample
    period: int
        Expected period in number of data points

    Returns
    -------
    valleys: numpy.ndarray
        (Possibly updated) indexes of valleys in sample
    """
    # remove valley if doing so reduces diff_var
    while True:
        cands = []
        dv_vals = _diff_var(valleys, period)
        # don't allow removal of first or last valley
        for valley in valleys[1:-1]:
            valleys_removed = valleys[valleys != valley]
            dv_vals_rem = _diff_var(valleys_removed, period)
            if dv_vals_rem < dv_vals:
                cands.append([valley, dv_vals_rem])
        if len(cands) == 0:
            break
        cands = np.array(cands)
        valley = int(cands[np.argmin(cands[:, 1]), 0])
        valleys = valleys[valleys != valley]
    return valleys


def replace_valleys(s, valleys, valleys_new, period):
    """
    Greedily replace valleys in current list of valleys if doing so reduces
    diff_var

    Parameters
    ----------
    valleys: numpy.ndarray
        Indexes of valleys in sample
    valleys_new: numpy.ndarray
        Indexes of candidate valleys in sample
    period: int
        Expected period in number of data points

    Returns
    -------
    valleys: numpy.ndarray
        (Possibly updated) indexes of valleys in sample
    """
    # replace valley_delete with valley_new if doing so reduces diff_var
    valleys_new = np.setdiff1d(valleys_new, valleys)
    while True:
        cands = []
        dv_vals = _diff_var(valleys, period)
        for valley_new in valleys_new:
            dists = np.abs(valleys - valley_new)
            for valley_del in valleys[dists == dists.min()]:
                valleys_replaced = valleys.copy()
                valleys_replaced[valleys == valley_del] = valley_new
                dv_vals_rep = _diff_var(valleys_replaced, period)
                if dv_vals_rep < dv_vals:
                    cands.append([valley_new, valley_del, dv_vals_rep])
        if len(cands) == 0:
            break
        cands = np.array(cands)
        valley_new, valley_del = map(int, cands[np.argmin(cands[:, 2]), :2])
        valleys[valleys == valley_del] = valley_new
        valleys_new = valleys_new[valleys_new != valley_new]

    def is_downslope(vnew, vdel):
        return np.all(np.diff(s[vdel:vnew:np.sign(vnew - vdel)]) <= 0)

    for valley_new in valleys_new:
        dists = np.abs(valleys - valley_new)
        valleys_del = valleys[dists == dists.min()]
        is_downslopes = np.array([is_downslope(valley_new, vdel)
                                  for vdel in valleys_del])
        if is_downslopes.sum() == 1:
            valley_del = valleys_del[is_downslopes][0]
            valleys[valleys == valley_del] = valley_new
        if is_downslopes.sum() == 2:
            if s[valleys_del[0]] <= s[valleys_del[1]]:
                valley_del = valleys_del[0]
            else:
                valley_del = valleys_del[1]
            valleys[valleys == valley_del] = valley_new
    return valleys


def merge_valleys(s, valleys, valleys_new, period):
    """
    Merge canidate valleys into current list of valleys by greedily adding,
    replacing and removing valleys.

    Parameters
    ----------
    valleys: numpy.ndarray
        Indexes of valleys in sample
    valleys_new: numpy.ndarray
        Indexes of candidate valleys in sample
    period: int
        Expected period in number of data points

    Returns
    -------
    valleys: numpy.ndarray
        (Possibly updated) indexes of valleys in sample
    """
    valleys = add_valleys(valleys, valleys_new, period)
    valleys = replace_valleys(s, valleys, valleys_new, period)
    valleys = remove_valleys(valleys, period)
    return valleys


def get_valleys_lowpass(x, sample_rate, period, init_cutoff_freq=None,
                        final_cutoff_freq=None, num=4):
    """
    Get valleys from a numerical data sample by starting with the valleys of a
    lowpass filtered version of the sample, and iteratively amending the list
    of valleys with candidates from filtered versions of the sample with
    progressively larger lowpass cutoff frequencies. Candidates are chosen if
    they minimize the mean variation about the expected period.

    Parameters
    ----------
    x: numpy.ndarray
        1D numerical array
    sample_rate: float
        Sampling rate in hertz
    period: int
        Expected period in number of data points
    init_cutoff_freq: int, default None
        Initial corner frequency of low-pass Butterworth filter. If None, set
        to frequency implied by period.
    final_cutoff_freq: int, default None
        Final corner frequency of low-pass Butterworth filter. If None, set to
        init_cutoff_freq + num - 1.
    num: int, default 4
        Number of corner frequencies to use, evenly spaced from
        init_cutoff_freq to final_cutoff_freq. See numpy.linspace for details.

    Returns
    -------
    valleys: numpy.ndarray
        Indexes of valleys in sample
    res: dict
        Contains current list of valleys, candidate valleys from iteration, and
        the smoothed sample from iteration
    """
    if init_cutoff_freq is None:
        # set to frequency of period
        # FIXME magic number 1.2, it's here because sometimes you need a little
        # more wiggle to see the valleys. Another possible way to address this
        # problem is to decrease the order in get_valleys, but they doesn't
        # increase the wiggle to make the valleys more prominent
        init_cutoff_freq = 1.2 * sample_rate / float(period)
    if final_cutoff_freq is None:
        final_cutoff_freq = init_cutoff_freq + num - 1
    # set to nyquist frequency if larger
    final_cutoff_freq = min(final_cutoff_freq, sample_rate / 2)
    # initial cutoff frequency
    cutoff_freqs = np.linspace(init_cutoff_freq, final_cutoff_freq, num)
    s, _, _ = smooth(x, sample_rate, cutoff_freqs[0])
    # FIXME magic number order=period // 2
    valleys = remove_valleys(get_valleys(s, order=period // 2), period)
    res = {init_cutoff_freq: {'valleys': valleys,
                              'valleys_new': valleys,
                              'filtered': s}}
    # other cutoff frequencies
    for cutoff_freq in cutoff_freqs[1:]:
        s, _, _ = smooth(x, sample_rate, cutoff_freq)
        valleys_new = get_valleys(s)
        valleys = merge_valleys(s, valleys.copy(), valleys_new, period)
        res[cutoff_freq] = {'valleys': valleys,
                            'valleys_new': valleys_new,
                            'filtered': s}
    return valleys, res


def get_period(x, sample_rate, passband=[0.75, 1.25]):
    """
    Calculate the period of cycle whose frequency is expected within the
    specified passband. The signal x is bandpass-filtered using a second-order
    Butterworth filter, and the period is calculated between peaks in the
    correlogram.

    Parameters
    ----------
    x : ndarray
        1D numerical array
    sample_rate : float
        Sampling rate in hertz
    passband : array, length 2
        Passband for bandpass filter

    Returns
    -------
    period : int
        Length of cycle in number of samples
    """
    passband_normed = np.array(passband, dtype=np.float64) / sample_rate * 2.
    b, a = ss.butter(2, passband_normed, btype='bandpass')
    s = ss.filtfilt(b, a, x - x.mean())
    c = freq.correlate(s, s, mode='same')
    peaks = ss.argrelmax(c, mode='clip')[0]
    idx = np.argmax(c[peaks])
    return peaks[idx + 1] - peaks[idx]


def segment_sample(sample, sample_rate, col='Am', level='cycle_id',
                   passband=[0.75, 1.25], init_cutoff_freq=None,
                   final_cutoff_freq=None, num=4, segment_method=None,
                   **kwargs):
    """
    Segment samples into cycles.

    Parameters
    ----------
    sample : pandas.DataFrame
        Numerical data frame. Must have final index level 'time' and
        contain the column col.
    sample_rate : float
        Sampling rate in hertz
    col : str,  default 'Am'
        Column of sample to be used for segmentation
    level : str, default 'cycle_id'
        Name of new level in index
    passband : array, length 2
        Passband for bandpass filter
    init_cutoff_freq : int, default None
        Initial corner frequency of low-pass Butterworth filter
    final_cutoff_freq : int, default None
        Final corner frequency of low-pass Butterworth filter.
    num : int, default 4
        Number of corner frequencies to use, evenly spaced from
        init_cutoff_freq to final_cutoff_freq. See numpy.linspace for details.
    segment_method : {'previous_peak', 'next_cross', None}, default None
        Method used to compute segment boundaries from valleys.
        * None: segment at valleys
        * next_cross: for each valley, segment at the next (in time) crossing
          of 'thr' by the final_cutoff_freq-filtered signal
        * previous_peak: for each valley, segment at the previous (in time)
          peak of the final_cutoff_freq-filtered signal
    kwargs : dict
        Keyword arguments for segment methods

    Returns
    -------
    cycles : pandas.DataFrame
        Same as sample but with level added to the index, just in front of
        level 'time'
    period : int
        Expected period in number of data points
    res : dict
        Results of get_valleys_lowpass
    """
    x = sample[col].values
    period = get_period(x, sample_rate, passband=passband)
    valleys, res = get_valleys_lowpass(x, sample_rate, period,
                                       init_cutoff_freq=init_cutoff_freq,
                                       final_cutoff_freq=final_cutoff_freq,
                                       num=num)

    def next_cross(thr=9.81, **kwargs):
        y = res[sorted(res.keys())[-1]]['filtered']
        crossings = []
        for valley in valleys:
            cands = np.where(y[valley:valley + period] > thr)[0]
            if len(cands) > 0:
                crossings.append(valley + cands[0])
        return np.array(crossings)

    def previous_peak(**kwargs):
        y = res[sorted(res.keys())[-1]]['filtered']
        crossings = []
        for valley in valleys:
            sidx = max(0, valley - period)
            # FIXME magic number order=period // 4
            cands = ss.argrelmax(y[sidx:valley], order=period // 4)[0]
            if len(cands) > 0:
                crossings.append(sidx + cands[-1])
        return np.array(crossings)

    if segment_method is None:
        idxs = valleys
    else:
        idxs = locals().get(segment_method)(**kwargs)
    labels = [np.nan] + list(range(len(idxs) - 1)) + [np.nan]
    idxs = [0] + idxs.tolist() + [len(sample)]
    diff_idxs = np.diff(idxs)
    cycles = sample.copy()
    cycles[level] = np.repeat(labels, diff_idxs)
    cycles = cycles.set_index(level, append=True).swaplevel(level, 'time')
    return cycles, period, res


def get_cycle_boundaries(cycles, level='cycle_id'):
    g = cycles.reset_index('time').groupby(level=level).time
    firsts = g.first()
    ts = cycles.index.get_level_values('time')
    lidx = np.where(ts == g.last().iloc[-1])[0][0]
    return firsts.tolist() + [ts[lidx + 1]]
