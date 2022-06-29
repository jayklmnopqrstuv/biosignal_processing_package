import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import scipy.signal as ss
import scipy as sp
from scipy.interpolate import interp1d
from rndSignal.sensorlib.stepcycle import segment_sample
import rndSignal.sensorlib.stepcycle as stp
from scipy.interpolate import UnivariateSpline
#import psycopg2 #remove temporarily since not being utilized as of the moment
import pytz
#from . import cwt_signal_quality as sq
#import joblib as jl
#import gspread
#from oauth2client.service_account import ServiceAccountCredentials
from scipy.stats import pearsonr
from rndSignal.sensorlib.features import cwt_fft
import rndSignal.sensorlib.freq as slf
from rndSignal.preprocessing.signal_filter import filter_ppg
import warnings

#PREPROCESSING
def cascade_low_high_pass_filter(x, sampling_freq_hz, hpass_cor_freq_hz=0.5,
                                 lpass_cor_freq_hz=3.0,cascade = False):
    """
    This is a cascaded butterworth high pass filter
    and low pass filter.

    input:
        x - is a 1d array of raw ppg signal. Signal to be filtered.
        sampling_freq_hz - is the sampling rate of the raw ppg signal in Hz.
        hpass_cor_freq_hz - is the high pass filter cut-off frequency in Hz.
        low_pass_filter - is the low pass filter cut-off frequency in Hz.
    return:
        filtered_x - is 1d array filtered raw ppg signal.
    """
    if cascade:
        nyquist = sampling_freq_hz / 2.0
        f_c_high = np.array([hpass_cor_freq_hz, ], dtype=np.float64)  # Hz
        # Normalize by Nyquist
        f_c_high /= nyquist
        # Second order Butterworth low pass filter at corner frequency
        b, a = ss.butter(2, f_c_high, btype='high') # second order butterworth

        f_c_low = np.array([lpass_cor_freq_hz, ], dtype=np.float64)  # Hz
        # Normalize by Nyquist
        f_c_low /= nyquist
        # Sixth order Butterworth low pass filter at corner frequency
        bh, ah = ss.butter(6, f_c_low, btype='low') # sixth order butterworth

        filtered_x = ss.filtfilt(bh, ah, ss.filtfilt(b, a, x))

        return filtered_x
        
    else:
        filtered_x = filter_ppg(x,sampling_freq_hz)
        
        return filtered_x


def time_index_from_capture(start_time, n, fs):
    """
    Create a DatetimeIndex from capture start time and sampling frequency.

    It's useful to have datetimes for capture datapoints since the index
    for all the beats uses timestamps.
    """
    start_time_nanos = start_time.value
    time_deltas_nanos = np.int64(np.arange(n) / fs * 1.0e9)

    times_nanos = time_deltas_nanos + start_time_nanos
    index = pd.to_datetime(times_nanos)
    return index


def sg_derivatives(x, window_length = 7, polyorder = 3, deriv = 1):
    """
    This is derivative function.

    input:
        x - is a 1d array. Signal to be differentiated.
    return:
        1d array differentiated signal.
    """

    return ss.savgol_filter(x, window_length = window_length,
                            polyorder = polyorder , deriv=deriv)


def systolic_peak_ppg(ppg, sampling_freq_hz, low_pass_cor_f_hz = 1.85):
    """
    computes the indices of systolic peaks.

    input:
        ppg - 1d array filtered ppg signal.
        sampling_freq_hz - sampling frequency of ppg signal in Hz.
        fast_cor_f_hz - low pass filter higher cut-off frequency in Hz.
        slow_cor_f_hz - low pass filter lower cut-off frequency in Hz.
    return:
        systolic_index - 1d array of indices of systolic peaks.
        window_span - List of span indices. Inside of which a systolic peak is located.
        lpf_fast - higher cut-off frequency low pass filtered signal.
        lpf_slow - lower cut-off frequency low pass filtered signal.
    """

    lpf_fast = low_pass_filter(ppg, sampling_freq_hz, low_pass_cor_f_hz)
    lpf_slow = np.zeros(len(ppg))
    window_span = window_index_pairs(lpf_fast, lpf_slow)
    systolic_index =  np.array([start+np.argmax(ppg[start:end]) for start, end in window_span])

    return systolic_index, window_span, lpf_fast, lpf_slow


def trough_index_ppg(ppg, systolic_index):

    """
    computes the indices of ppg valleys.

    input:
        ppg - 1d array filtered ppg signal.
        systole_index - indices of systolic peaks
    return:
        1d array of indices of ppg valleys.
    """

    ppg_sys_to_sys_segment = [[systolic_index[n], systolic_index[n+1]]\
                              for n in range(len(systolic_index)-1)]
    trough_index = [k[0]+np.argmin([ppg[k[0]:k[1]]]) \
                    for k in ppg_sys_to_sys_segment]

    return np.array(trough_index)


def ppg_beat_cycle_id(ppg, trough_idx):

    idx = [0] + trough_idx.tolist() + [len(ppg)]
    diff_idx = np.diff(idx)
    labels = [np.nan] + list(range(len(trough_idx)-1)) + [np.nan]
    cycle_id = np.repeat(labels, diff_idx)
    return cycle_id


def preprocess(x, sampling_freq_hz, hpass_cor_freq_hz=0.5, lpass_cor_freq_hz=3.,
               timestamp = pd.Timestamp('2000-1-1'), cycle_method = '1',
               cycle_index = None, filter_signal = True, cascade = False):

    """
    This is a pre-process to ppg features computation. It computes
    the filtered ppg, first derivative ppg, second derivative ppg and
    cycle id for the second derivative ppg.

    input:
        x - is a 1d array raw ppg signal.
    return:
        df - a pandas dataframe the contains the
            columns:
            Am - raw ppg signal
            PPG - filtered ppg signal
            FDPPG - first derivative ppg signal
            SDPPG - second derivative ppg signal

            index:
            time - time index of ppg signal
            cycle_id - cycle id of second derivative ppg signal.

    """
    if filter_signal:
        ppg = cascade_low_high_pass_filter(x, sampling_freq_hz,
                                            hpass_cor_freq_hz=hpass_cor_freq_hz,
                                            lpass_cor_freq_hz=lpass_cor_freq_hz,
                                            cascade = cascade)
    else:  
        ppg = x
    fdppg = sg_derivatives(ppg, deriv=1)
    sdppg = sg_derivatives(ppg, deriv=2)

    df = pd.DataFrame(np.array([x,ppg,fdppg,sdppg]).T,
                      columns=["Am","PPG","FDPPG","SDPPG"])

    systolic_index = systolic_peak_ppg(ppg, sampling_freq_hz)[0]
    ppg_trough_index = trough_index_ppg(ppg, systolic_index)

    df["time"] = time_index_from_capture(timestamp, len(x), sampling_freq_hz)
    if cycle_index is None:
        if cycle_method =='1': #joseph's method
            df["cycle_id"] =  ppg_beat_cycle_id(ppg, ppg_trough_index)
            df.set_index(["cycle_id", 'time'], inplace = True)
        elif cycle_method =='2': #using the segment sample method
            passband, cut_off_freq, num = ([0.5, 1.], 7, 6)
            #see ppg_processing_experimentation for the seg params
            df.set_index("time", inplace = True)
            df,_,_ = segment_sample(df, sampling_freq_hz,level="cycle_id",
                                col='PPG', passband=passband,
                                final_cutoff_freq=cut_off_freq, num=num)
        else:
            raise ValueError("Invalid cycle method")
    else:#this will be used for segmenting red signal using ir's segmentation
        if len(cycle_index) != len(ppg):
            raise ValueError("Length of cycle index is not equal to \
                             length of signal")
        else:
            df["cycle_id"] = cycle_index
            df.set_index(["cycle_id", "time"], inplace = True)

    return df


#5 FEATURES
#1. and 2. heart rate and heart rate variability functions
def low_pass_filter(x, sampling_freq_hz, lowpass_corner_freq_hz, order = 6):
    """
    This is a butterworth low pass filter.

    input:
        x - 1d array signal to be filtered.
    return:
        filtered_x - filtered signal
    """

    nyquist = sampling_freq_hz / 2.0
    f_c_low = np.array([lowpass_corner_freq_hz, ], dtype=np.float64)  # Hz
    # Normalize by Nyquist
    f_c_low /= nyquist
    # Second order Butterworth low pass filter at corner frequency
    b, a = ss.butter(order, f_c_low, btype='low') # sixth order butterworth

    filtered_x = ss.filtfilt(b, a, x)
    return filtered_x


def window_index_pairs(fast, slow):
    """
    computes the index of window span based on the intersection
    of 2 input signals.

    input:
        fast - 1d array high frequency signal
        slow - 1d array low frequenct signal
    return:
        window - list of span indices
    """
    positive_crossing = []
    negative_crossing = []
    for i in range(len(fast)-1):
        if fast[i] <= slow[i] and fast[i+1] > slow[i+1]:
            positive_crossing.append(i)
        if fast[i] >= slow[i] and fast[i+1] < slow[i+1] and len(positive_crossing)>0:
            negative_crossing.append(i)

    window = list(zip(positive_crossing,negative_crossing))
    return window


def systolic_to_systolic_bpm(systolic_index, sampling_freq_hz):
    """
    computes the heart rate based on the systolic peaks.

    input:
        systolic_time - indices of the systolic peaks.
        sampling_freq_hz - sampling rate of the ppg.
    return:
        systolic_bpm - heart rate in bpm.
        hrv - heart rate variability.
        rms - root mean squared of hear rate
        time_per_cycle - time per systolic peak interval.
    """
    systolic_difference = np.diff(systolic_index)
    time_per_cycle = systolic_difference/(sampling_freq_hz*60) # minutes per cycle
    try: systolic_bpm = 1.0/(np.median(time_per_cycle))
    except: systolic_bpm = 0.0

    hrv = np.std(time_per_cycle*60*1000) # milliseconds
    rms = np.sqrt(np.mean((time_per_cycle*60*1000)**2))
    return systolic_bpm, hrv, rms, time_per_cycle


def heart_rate_good_beats(beats_df, fs):
    #filter good beats from beats_df
    good_beat_window = beats_df[beats_df['pred_label'] == 1.0]['index'].values

    #compute for the durations of good beats on the capture
    #require 60% of the ppg beats to be good to compute for heart rate
    if np.mean(beats_df["pred_label"].values)>=0.60:
        beat_duration_values_good = np.array([(y-x) for (x,y) in good_beat_window])
        heart_rate = (len(good_beat_window)/np.sum(beat_duration_values_good))*fs*60
        hrv_std = np.std(beat_duration_values_good*1000/fs)
        hrv_rms = np.sqrt(np.mean(beat_duration_values_good*1000/fs)**2)
        return (heart_rate, hrv_std, hrv_rms)
    else:
        return (np.nan, np.nan, np.nan)

#3. oxygen saturation functions

def ppg_envelope(ppg, x):
    """
    computes the ppg envelopen based on either the systolic peaks
    or the ppg valleys.

    input:
        ppg - 1d array filtered ppg signal.
        x - index of either the systolic peaks or ppg valleys.
    return:
        new_y - 1d array of ppg envelope.
    """
    x_last_idx = len(ppg)-1
    y = ppg[x]

    x_ = [0] + x.tolist() + [x_last_idx]
    y_ = [ppg[x[0]]] + y.tolist() + [ppg[x[-1]]]

    new_length = len(ppg)
    new_x = np.linspace(0, len(ppg)-1, new_length)
    new_y = sp.interpolate.UnivariateSpline(x_, y_, k=2)(new_x)

    return new_y


def compute_spo2_from_acdcratio(acdc_ir, acdc_red, calib_curve=None):
    if not calib_curve:
        m,b = (-42.56460065019621, 127.41112059601953)
    else:
        m,b = calib_curve

    R = acdc_red/acdc_ir
    spo2 = m*R+b

    if spo2>100:
        return 1.0
    elif spo2<0:
        return 0.0
    else:
        return spo2/100.



#4. blood pressure
#functions used for computing the features for predicting
#blood pressure: systolic upstroke time, diastolic time,
#time delay

def sys_peak_beats(df, fs):
    """
    df: should have "cycle_id" index and "PPG" column
    modifies the output returned from pu.systolic_peak_ppg function
    to return systolic peaks of the ppg beats
    """
    sys_peaks = systolic_peak_ppg(df["PPG"].values,fs)[0]

    indices = np.zeros(len(df))
    indices[sys_peaks] = 1
    indices *= np.arange(len(df))
    fin_sys_peaks = (df.reset_index(["time", "cycle_id"])
                      .groupby("cycle_id")
                     .apply(lambda x: indices[x.index[:]])
                      .apply(lambda x: x[x>0][0] if np.any(x) else np.nan))
    return fin_sys_peaks.values


def diastolic_peak_loc(df):
    """
    detects diastolic peaks per ppg cycle using zero crossings in
    first derivative ppg
    df must have "cycle_id" index, "PPG", "FDPPG" columns
    """

    def zero_cross(signal):
        return np.where(np.diff(np.sign(signal)))[0] +1 #get the index at where it changed its sign

    df_copy = df.copy()
    inds = zero_cross(df_copy["FDPPG"])
    zc = np.zeros(len(df_copy))
    zc[inds] = 1
    zc *= np.arange(len(df_copy))
    df_copy["zc"] = zc

    diastolic_index = (df_copy.reset_index(["cycle_id", "time"])
                         .groupby("cycle_id")
                         .apply(lambda x: (x["zc"][x["PPG"]>0]).values)
                         .apply(lambda x: x[x>0])
                         .apply(lambda x: np.append(x, [np.nan]*(3-len(x)))[2] if len(x)<3 else x[2]))

    return diastolic_index.values


def systolic_upstroke_time(df_ir, fs):
    """
    df_ir: dataframe for infrared signal, must contain
    indexes "cycle_id" and "time".
    Computes the mean(in seconds) systolic upstroke time, time from start
    of the ppg cycle up to systolic peak of all the ppg cycles
    """
    systolic_index = sys_peak_beats(df_ir, fs)
    start_index = (df_ir.reset_index(["cycle_id", "time"])
                        .groupby("cycle_id").apply(lambda x: x.index[0]).values)
    valid = np.where(~np.isnan(start_index*systolic_index)*systolic_index>start_index)[0]
    if np.any(valid):
        systolic_time = df_ir.index.get_level_values(1)[systolic_index[valid].astype(int)]
        start_time = df_ir.index.get_level_values(1)[start_index[valid].astype(int)]
        return np.mean([systolic_time[i]-start_time[i] for i in range(len(valid))]).total_seconds()
    else:
        return np.nan #no valid pair of peaks found


def diastolic_time(df_ir, fs):
    """
    df_ir: dataframe for infrared signal, must contain
    indexes "cycle_id" and "time".
    Computes the mean(in seconds) diastolic time, time from systolic peak
    up to the end of the ppg cycle of all the ppg cycles
    """
    systolic_index = sys_peak_beats(df_ir, fs)
    end_index = (df_ir.reset_index(["cycle_id", "time"])
                 .groupby("cycle_id").apply(lambda x: x.index[-1]).values)
    valid = np.where(~np.isnan(systolic_index*end_index)*end_index>systolic_index)[0]
    if np.any(valid):
        systolic_time = df_ir.index.get_level_values(1)[systolic_index[valid].astype(int)]
        end_time = df_ir.index.get_level_values(1)[end_index[valid].astype(int)]
        return np.mean([end_time[i] - systolic_time[i] for i in range(len(valid))]).total_seconds()
    else:
        return np.nan #no valid pair of peaks found


def time_delay(df_ir, fs):
    """
    df_ir: dataframe for infrared signal, must contain
    indexes "cycle_id" and "time".
    Computes for the mean(in seconds) time delay, time from systolic peak
    to diastolic peak of all the ppg cycles
    """
    systolic_index = sys_peak_beats(df_ir, fs)
    diastolic_index = diastolic_peak_loc(df_ir)

    valid = np.where(~np.isnan(diastolic_index*systolic_index)*diastolic_index>systolic_index)[0]
    if np.any(valid):
        systolic_time = df_ir.index.get_level_values(1)[systolic_index[valid].astype(int)]
        diastolic_time = df_ir.index.get_level_values(1)[diastolic_index[valid].astype(int)]
        return np.mean([diastolic_time[i] - systolic_time[i] for i in range(len(valid))]).total_seconds()
    else:
        return np.nan #no valid pair of peaks found



#5. respiration rate
def respiration_rate_brpm(respiration_signal, sampling_freq_hz):
    """
    compute the respiration rate and time interval per respiration cycle.

    input:
        respiration_signal - 1d array of filtered respiration signal. Respiration
                            signal can be the low pass filter of the following signals:
                                - systolic envelope
                                - trough envelope
                                - low pass filter of filtered ppg signal
        sampling_freq_hz - sampling rate of ppg signal.
    return:
        respiration_rate - breaths per minute
        time_per_cycle - 1d array of time interval between respiration.
    """
    window_span = window_index_pairs(respiration_signal,np.zeros(len(respiration_signal)))
    respiration_peak_idx =  [start+np.argmax(respiration_signal[start:end]) for start, end in window_span]
    respiration_rate,_,_,time_per_cycle = systolic_to_systolic_bpm(respiration_peak_idx, sampling_freq_hz)
    return respiration_rate, time_per_cycle


def respiration_process(signal_dict,fs):
    """
    process the multiple input respiration signal. compute the respiration rate and
    time interval per respiration cycle on each of the input signal.

    input:
        signal_dict - a dictionary of input respiration signal. Respiration
                     input signal can be the low pass filter of the following signals:
                         - systolic envelope
                         - trough envelope
                         - low pass filter of the filtered ppg signal
                    example:
                        dict_resp = {"systolic"     : systolic_envelope,
                                     "trough"       : trough_envelope,
                                     "ppg_low_pass" : ppg
                                    }
        fs - sampling frequency of ppg signal in Hz.
    return:
        dictionary of the ppg respiration signal, respiration rate in breaths per minute,
        time interval per respiration cycle.

    """
    respiration_signal = {}
    respiration_per_min = {}
    respiration_time_per_cycle = {}

    for key in signal_dict:
        sig = signal_dict[key]
        smooth = cascade_low_high_pass_filter(sig,fs, 0.2, 0.4, cascade = True)
        rate_, time_ = respiration_rate_brpm(smooth, fs)
        respiration_signal[key] = smooth
        respiration_per_min[key] = rate_
        respiration_time_per_cycle[key] = time_

    respiration_dict = {"signal"     : respiration_signal,
                        "rate"       : respiration_per_min,
                        "cycle_time" : respiration_time_per_cycle}

    return respiration_dict


def get_l_middle_values(x,l):
    """
    extract the l count middle values from an array.

    input:
        x - 1d array
        l - length of middle values to be extracted from x
    return:
        1d array of middle values with length l
    """
    s = len(x)
    m_x = s//2
    m_l = l//2
    start = m_x - m_l
    return np.array(list(x[start:start+l])*1)


def kalmanf(cycle_time_dict):

    """
    Sensor fusion using Kalman filter. Fuse the different sources
    of respiration rate.
    input:
        dictionary of respiration cycle time from
        different source of respiration signal.
        example:
            {"ppg_low_pass"  : np.array([...]),
             "sytolic"       : np.array([...]),
             "trough"        : np.array([...])
            }
        where each 1d array values are the time interval per
        respiration cycle in breaths per minute.
    return:
        dictionary of sensor fusion.
            - respiration rate in breaths per minute
            - the measurements
            - the kalman estimates
            - kalman gain
            - error of estimate
    """

    val = cycle_time_dict.values()
    key = list(cycle_time_dict.keys())

    l = min([len(u) for u in val])

    q_l =  [get_l_middle_values(t,l) for t in val]

    est = np.mean([np.mean(e) for e in val])    # initial estimate
    e_est = np.max([np.var(e) for e in val])    # error in the estimate
    e_process = 0.000001                        # process error
    e_mea = np.array([np.var(x) for x in val])  # error in the measurement
    mea = np.array(q_l)                         # measurement

    h = np.ones(len(key))

    est_list = []
    kg_list = []
    e_est_list = []

    for i in range(len(np.transpose(q_l))):
        e_est = e_est + e_process
        kg = e_est*h/(e_est*h+e_mea)               # kalman gain
        est = est + np.dot(kg,(mea.T[i] - h*est))  # updated estimate
        e_est = (1-kg)*e_est                       # updated error in the estimate

        est_list.append(est)
        kg_list.append(kg)
        e_est_list.append(e_est)

    kalman_dict = {"resp_per_min"   : 1./est,
                   "measurement"    : dict(zip(key,mea)),
                   "estimate"       : np.array(est_list),
                   "kalman_gain"    : dict(zip(key,np.transpose(kg_list))),
                   "error_estimate" : dict(zip(key,np.transpose(e_est_list)))}

    return kalman_dict


def respiration_rate_kalman(ppg, systolic_envelope, trough_envelope, fs):
    """
    Estimate the respiration rate from 3 respiration signals
        1. Sytolic Envelope
        2. Trough Envelope
        3. Low pass filter of PPG

    Input:
        ppg - 1D Array of Filtered PPG
        systolic_envelope - 1D Array of systolic envelope
        trough_envelope - 1D Array of trough envelope
        fs - Sampling rate of PPG

    Return:
        respiration_process_dict - dictionary of the ppg respiration signal,
                                   respiration rate in breaths per minute,
                                   time interval per respiration cycle for
                                   each of the respiration signals.

        respiration_kalman_fused_dict - dictionary of kalman sensor fusion.
                                            - respiration rate in breaths per minute
                                            - the measurements
                                            - the kalman estimates
                                            - kalman gain
                                            - error of estimate

        respiration_rate - estimate of respiration rate in breaths per minute
                           from sensor fusion of 3 respiration signals.

    """

    respiration_signal_dict = {"systolic"     : systolic_envelope,
                               "trough"       : trough_envelope,
                               "ppg_low_pass" : ppg}

    respiration_process_dict = respiration_process(respiration_signal_dict, fs)
    respiration_cycle_time = respiration_process_dict['cycle_time']
    respiration_kalman_fused_dict = kalmanf(respiration_cycle_time)
    respiration_rate = respiration_kalman_fused_dict['resp_per_min']

    return respiration_process_dict, respiration_kalman_fused_dict, respiration_rate


def getwindow(signal, time_sig, window_duration = 16, window_interval=1, f_s = 4):
    """
    Function that divides a signal into windows

    INPUT
        signal:          1d array containing the signal
        time_sig:        1d array containing the time of the signal
        window_duration: duration of one window in seconds
        window_interval: interval at which the window will slide (default value: 1sec)
                         signal must contain values divisible by the window_interval
        f_s:             sampling frequency; unit is in Hz

    OUTPUT
        signal_segments: array containing array of values per window
        time_segments:   array containing array of time per window

    """
    time_signal = np.arange(len(signal))*1/f_s #relative to zero
    start_index = np.arange(len(signal))[(time_signal%window_interval) == 0]
    end_index = start_index + window_duration*f_s
    end_index = end_index[end_index<=len(signal)]
    start_index = start_index[:len(end_index)]

    indices = start_index[:, None] + np.arange(end_index[0])
    signal_segments = signal[indices]
    time_segments = time_sig[indices]

    return signal_segments, time_segments


def stdize(signal):
    """
    INPUT:
        signal:         (1d array) signal to standardize

    OUTPUT:
        signal_stdize:  (1d array) standardized signal
    """
    signal_stdize = (signal-np.nanmean(signal))/np.std(signal)
    return signal_stdize


def resample_interpolate(signal, time_signal, new_fs = 4):
    """
    INPUT:
        signal:       (1-dimensional array) - signal
        time_signal:  (1-dimensional array) - time signal
        new_fs:       (int) - new sampling rate of the signal

    OUTPUT:
        sig_new:      (1-dimensional array) - new signal
        time_new:     (1-dimensional array) - new time signal
    """
    new_dt = 1/new_fs
    func = interp1d(time_signal, signal)
    time_new = np.arange(time_signal[0], int(time_signal[-1]), new_dt)
    #in case new_dt is float, roundoffs may affect
    sig_new = func(time_new)
    return sig_new, time_new


def respiration_rate_riv(ppg_sig, fs):
    """
    INPUT:
        ppg_sig:    (1-dimensional array) - ppg signal
        fs:         (int) sampling rate of the ppg signal

                NOTE: ppg_sig is assumed to have 100% signal quality,
                meaning it contains all good beats.

    OUTPUT:
        iv_smooth:  (1-dimensional array) - smoothed respiratory
                    induced intensity variation
        fv_smooth:  (1-dimensional array) - smoothed respiratory
                    induced frequency variation
        av_smooth:  (1-dimensional array) - smoothed respiratory
                    induced amplitude variation
        resp_rate:  (float) - respiration rate
    """

    #preprocessing
    #repiration range considered: min of 12 and max of 30 breaths per minute(0.2-0.5 Hz)
    #based on young individuals age 3 yr old to elders with age 80 yrs old and above.
    #Source: https://en.wikipedia.org/wiki/Respiratory_rate#Normal_range

    #To preserve the respiration frequency, we will use a high pass filter with cut off of 0.15 Hz
    filt_signal = cascade_low_high_pass_filter(ppg_sig, fs, hpass_cor_freq_hz=0.15, cascade = True)

    #systolic peaks and trough indices will be used in the calculation of respiratory
    #induced variations
    sys_peaks = systolic_peak_ppg(filt_signal, fs)[0]
    tro_peaks = trough_index_ppg(filt_signal, sys_peaks)

    sys_peaks_mag = filt_signal[sys_peaks]
    tro_peaks_mag = filt_signal[tro_peaks]

    #Note that all the respiratory induced variations are resampled at 4 Hz
    resampling_freq = 4
    #respiratory induced intensity variation
    iv, time = resample_interpolate(sys_peaks_mag[1:],
                                  (sys_peaks*(1/fs))[1:], #time_signal is relative to zero
                                  new_fs = resampling_freq)
    #respiratory induced frequency variation
    fv, time = resample_interpolate(np.diff(sys_peaks, n=1)*(1/fs),
                                  (sys_peaks*(1/fs))[1:],
                                  new_fs = resampling_freq)
    #respiratory induced amplitude variation
    av, time = resample_interpolate(sys_peaks_mag[1:]-tro_peaks_mag,
                                  (sys_peaks*(1/fs))[1:],
                                  new_fs = resampling_freq)

    #smoothen the rivs
    h_cutoff = 0.15
    l_cutoff = 0.6
    iv_smooth = cascade_low_high_pass_filter(iv, resampling_freq,
                             hpass_cor_freq_hz=h_cutoff, lpass_cor_freq_hz=l_cutoff, cascade = True)
    fv_smooth = cascade_low_high_pass_filter(fv, resampling_freq,
                             hpass_cor_freq_hz=h_cutoff, lpass_cor_freq_hz=l_cutoff,cascade = True)
    av_smooth = cascade_low_high_pass_filter(av, resampling_freq,
                             hpass_cor_freq_hz=h_cutoff, lpass_cor_freq_hz=l_cutoff,cascade = True)

    #Windowing with 16 sec duration per segment moving at 1 sec interval
    iv_win, time_win = getwindow(iv_smooth, time, window_duration=16,
                                 window_interval=1, f_s=resampling_freq)
    fv_win, time_win = getwindow(fv_smooth, time, window_duration=16,
                                 window_interval=1, f_s=resampling_freq)
    av_win, time_win = getwindow(av_smooth, time, window_duration=16,
                                 window_interval=1, f_s=resampling_freq)


    #fundamental freq is the respiration rate estimate
    resprate_iv_ff = np.array([slf.fund_freq_by_corr(stdize(iv_win[i]), samp_freq = 4)*60
                              for i in range(len(iv_win))])
    resprate_fv_ff = np.array([slf.fund_freq_by_corr(stdize(fv_win[i]), samp_freq = 4)*60
                               for i in range(len(fv_win))])
    resprate_av_ff = np.array([slf.fund_freq_by_corr(stdize(av_win[i]), samp_freq = 4)*60
                               for i in range(len(av_win))])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        resp_rate = np.nanmean([resprate_av_ff, resprate_iv_ff, resprate_fv_ff])

    return iv_smooth, fv_smooth, av_smooth, resp_rate

#6. continuous wavelet transform signal quality

def cwt_features_per_beat_mean_signal(pd_df, fs, nmf):
    x = pd_df['PPG'].values
    index_pd_df = pd_df.index
    res, freq, time = cwt_fft(x, dt=1.0/fs, low_freq=0.1, high_freq=6)
    cwt_magnitude = np.abs(res['W'][freq][:, time])
    freqs = res['freqs'][freq]
    df = pd.DataFrame(cwt_magnitude.T, index=pd_df.index)
    df_mean = df.groupby(level='cycle_id').mean()
    idx = df_mean.index
    features =nmf.transform(df_mean)
    features_df = pd.DataFrame(features, index = idx)

    return features_df


def cwt_proba_signal_df(signal_df, systolic_idx, fs, nmf, rf):
    df_ppg = signal_df[['PPG']]
    df_ppg.reset_index(inplace=True)
    df_ppg = df_ppg.dropna().set_index('cycle_id')[['PPG']]
    W = cwt_features_per_beat_mean_signal(df_ppg, fs, nmf)
    proba_idx = systolic_idx[1:-1]
    proba_ = rf.predict_proba(W.values).T[1]
    return proba_idx, proba_


#2ND DERIV PPG PEAKS AND RATIOS
def a_loc_sdppg(sdppg, sampling_freq_hz, fast_cor_f_hz = 2.0 , slow_cor_f_hz = 0.6):
    """
    computes the indices of 'a' peaks in second derivative ppg.

    input:
        sdppg - 1d array of second derivative ppg.
        sampling_freq_hz - sampling rate of ppg signal in Hz.
        fast_cor_f_hz - higher cut-off frequency in Hz.
        slow_cor_f_hz - lower cut-off frequency in Hz.
    return:
        index of 'a'
        squared of sdppg
        low pass filtered signal with higher cut-off Hz.
        low pass filtered signal with lower cut-off Hz.
        indices of span .
    """
    squared_ = sdppg**2
    lpf_fast = low_pass_filter(squared_,sampling_freq_hz, fast_cor_f_hz)
    lpf_slow = low_pass_filter(squared_,sampling_freq_hz, slow_cor_f_hz)
    window_span = window_index_pairs(lpf_fast, lpf_slow)

    arg =  [start+np.argmax(sdppg[start:end]) for start, end in window_span]

    return np.array(arg), squared_, lpf_fast, lpf_slow, window_span


def sdppg_cycle_segment(sdppg, a_x):
    """
    segment the second derivative ppg per cycle.

    input:
        sdppg - second derivative of ppg.
        a_x - index of second derivative 'a' peaks
    return:
        cycle_id - id per segment
    """
    idx = [0] + a_x.tolist() + [len(sdppg)]
    diff_idx = np.diff(idx)
    labels = [np.nan] + list(range(len(a_x)-1)) + [np.nan]
    cycle_id = np.repeat(labels, diff_idx)
    return cycle_id


def sdppg_critical_points(sdppg_cycle, df_index):
    """
    computes the b, c, d, e, f points in 1 second derivative
    ppg cycle.

    input:
        sdppg_cycle - 1d array of a segment in second derivative ppg.
        df_index - index of whole second derivative ppg signal.
    return:
        aw - index of critical points in the segment.
    """
    maxima = ss.argrelmax(sdppg_cycle)[0]
    minima = ss.argrelmin(sdppg_cycle)[0]
    idxs = np.sort(maxima.tolist() + minima.tolist())
    if len(idxs)>0:
        cps = sdppg_cycle[idxs]
        cpx = df_index[idxs]
    else:
        cps=[np.nan]
        cpx=[np.nan]

    cps_nan = np.append(cps[0:5], np.zeros(5-len(cps[0:5]))+np.nan)
    cpx_nan = np.append(cpx[0:5], np.zeros(5-len(cpx[0:5]))+np.nan)

    a =  sdppg_cycle[0]
    a_x = df_index[0]

    aw =  np.concatenate([np.array([a]), cps_nan, np.array([a_x]), cpx_nan])
    #critical_p =  np.concatenate([np.array([a_x]), cpx_nan])

    return aw #critical_p


def peaksloc_ratios_sdppg1(df, sampling_freq_hz):

    df_ = df.copy()

    sdppg = df_["SDPPG"].values


    a_x = a_loc_sdppg(sdppg, sampling_freq_hz)[0]
    sdppg_cycle_id = sdppg_cycle_segment(sdppg, a_x)
    df_["cycle_id_sd"] = sdppg_cycle_id

    df_["index"] = range(len(df))
    df_.reset_index("time", inplace = True)
    df_.set_index(["cycle_id_sd", "time"], inplace = True)

    sdppg_cycles = df_[["SDPPG", "index"]].groupby(level = "cycle_id_sd")

    sdppg_peaks = sdppg_cycles.apply(lambda x: sdppg_critical_points(x["SDPPG"].values,
                                                                     x["index"].values)).values
    a_, b_, c_, d_, e_, f_, \
    a_x, b_x, c_x, d_x, e_x, f_x = [np.array([x[t] for x in sdppg_peaks]) for t in range(12)]

    b_a = np.nanmedian(np.divide(b_,a_))
    c_a = np.nanmedian(np.divide(c_,a_))
    d_a = np.nanmedian(np.divide(d_,a_))
    e_a = np.nanmedian(np.divide(e_,a_))
    f_a = np.nanmedian(np.divide(f_,a_))

    a_x = a_x[~np.isnan(a_x)].astype(int)
    b_x = b_x[~np.isnan(b_x)].astype(int)
    c_x = c_x[~np.isnan(c_x)].astype(int)
    d_x = d_x[~np.isnan(d_x)].astype(int)
    e_x = e_x[~np.isnan(e_x)].astype(int)
    f_x = f_x[~np.isnan(f_x)].astype(int)

    peaksloc = [a_x, b_x, c_x, d_x, e_x, f_x]
    ratios = [b_a, c_a, d_a, e_a, f_a]

    #df.drop("index", axis = 1, inplace = True)
    return  (peaksloc, ratios)


def peaksloc_ratios_sdppg2(df, sampling_freq_hz, col = "PPG", passband = [0.5, 1.],
                    cut_off_freq = 7, num = 6):
    """
    Locate the indices of peaks a,b,c,d,e and f in sdppg
    using the segment_sample() function
    Input:
    df: dataframe with columns "PPG" and "SDPPG"
    """
    cycles_df, period, res = segment_sample(df, sampling_freq_hz,level = "cycle_id",
                                            col = col,passband=passband,
                                            final_cutoff_freq=cut_off_freq, num=num)
    cycles_df = cycles_df.reset_index()
    pkval_indices = np.sort(np.append(ss.argrelmax(np.array(cycles_df["SDPPG"]))[0],
                                      ss.argrelmin(np.array(cycles_df["SDPPG"]))[0]))
    indicator = np.zeros(len(cycles_df)).astype(int)
    indicator[pkval_indices] = 1
    cycles_df["indicator"] = indicator

    pks_ = cycles_df.groupby("cycle_id").apply(lambda x: x.index[x.indicator.astype(bool)][:6])
    pks_ = pks_.apply(lambda x: np.append(x, [np.nan]*(6-len(x))))

    #confirm if the index of peak a is the index of the maximum peak
    #if not ignore the peaks of that cycle
    bool_ = pks_.apply(lambda x: x[0])!= \
            cycles_df.groupby("cycle_id")["SDPPG"].apply(lambda x: np.argmax(x[:len(x)//2]))
    if np.any(bool_):
        pks_.loc[bool_] = [[np.nan]*6]*sum(bool_)

    pks_df = pd.DataFrame(np.array([i for i in pks_.values]), columns = ["a", "b", "c", "d", "e", "f"])
    pksloc = [pks_df[col].values for col in pks_df]
    ratios_array = [df.reset_index()["SDPPG"][pks_ind].values/ \
                    df.reset_index()["SDPPG"][pksloc[0]].values \
                            for pks_ind in pksloc[1:] ]
    ratios = [np.NaN if np.all(i!=i) else np.nanmedian(i) for i in ratios_array]

    cycles_df.drop("indicator", axis = 1, inplace = True)
    cycles_df.set_index(["cycle_id", "time"], inplace = True)
    return (cycles_df,(pksloc, ratios))



#FUNCTIONS FOR PPG BEATS

def normalize_cycle_length(df, sampling_freq_hz, len_new = 64, spline = 3,
                           level = False, scale = None, time_idx = False):
    """
    df: dataframe, should contain "cycle_id" and "time" as indices
        and "PPG" as column for filtered ppg
    sampling_freq_hz: sampling frequency
    len_new: new length of the interpolated signal
    spline: integer, used for spline interpolation
    level: boolean, if the endpoints of the beat should be leveled
    scale: perform a scaling on the beat
           - None: default, no scaling applied
           - "normalize": each beat will be bounded from 0 to 1
           - "standardize": each beat will have mean of 0 and standard deviation of 1
    time_idx: boolean, if the new time index for each interpolated  beat will be returned
    """
    def sr_new_offsets(len_sig):
        return round(1e3/ (len_new / len_sig * sampling_freq_hz))

    def spline_interpolate(signal, len_new):
        spline_func = UnivariateSpline(np.arange(len(signal)), signal, k=spline)
        return spline_func(np.linspace(0, len(signal), len_new))

    signal =  df.groupby(level = "cycle_id")\
                .apply(lambda x: spline_interpolate(x["PPG"].values, len_new)
                                 if len(x["PPG"].values)>spline \
                                 else np.array([np.nan]))
    #will only interpolate if the length of the beat is greater than spline

    index = sorted(list(map(lambda x: (x[0], x[-1]),\
                            df.reset_index().groupby("cycle_id").indices.values())))
    index = [i if (i[1]-i[0])+1>spline else None for i in index]
    dict_res = {"PPG_beat": signal, "index": index}
    if level:
        signal = signal.apply(lambda x: level_startend(x))
    if scale:
        if scale=="normalize":
            scaled_signal = signal.apply(lambda x: (x - np.min(x))/(np.max(x)-np.min(x))\
                                         if ~np.isnan(x).all() else x)
        elif scale=="standardize": #each beat will have a mean 0 and std dev of 1
            scaled_signal = signal.apply(lambda x: (x - np.mean(x))/np.std(x)\
                                         if ~np.isnan(x).all() else x)
        dict_res["PPG_beat_scale"] = scaled_signal
    if time_idx:
        #old way
        #time_index =  df.groupby(level = "cycle_id")\
        #                .apply(lambda x: x.reset_index("cycle_id")\
        #                        .resample(pd.offsets.Milli(sr_new_offsets(len(x)-1)))\
        #                        .mean().index[:len_new])
        #new way
        #http://stackoverflow.com/questions/37964100/creating-numpy-linspace-out-of-datetime
        time_index = df.reset_index().groupby("cycle_id")["time"]\
                       .apply(lambda x: (x.values[0], x.values[-1]))\
                       .apply(lambda x: (pd.Timestamp(x[0]), pd.Timestamp(x[1])))\
                       .apply(lambda x: np.linspace(x[0].value, x[1].value, len_new))\
                       .apply(lambda x: pd.to_datetime(x))
        dict_res["time"] = time_index
    return pd.DataFrame(dict_res)


def level_startend(signal):
    """
    Given a ppg beat signal, it fits a line from endpoint to endpoint and use that line
    to level the start and end values of the signal
    """
    x = np.arange(len(signal))
    line = signal[-1]+((signal[-1]-signal[0])/(x[-1]-x[0]))*(x-x[-1])
    diff = line -min(signal[0], signal[-1])
    return signal-diff



#function for loading the data from  amiigo database
def timestamp_converter(timestamp, timezone = "Asia/Manila"):

    time = pytz.timezone(timezone)
    timestamp = timestamp.astimezone(time)

    return pd.to_datetime(timestamp)
def ms_to_timestamp(ms_int):
    """
    ms_int: integer representing milliseconds elapsed
    """
    timestamp = pd.to_datetime(ms_int, unit = 'ms')
    timestamp = pytz.utc.localize(timestamp)
    return timestamp
def expand(df_):

    #expand "beats" column
    #features taken from "beats" column
    cols = ["spo2", "bpm", "b/a", "hrv"]
    func_mean = lambda x: x.apply(lambda col: np.nanmean(col), axis = 0).to_dict()
    verify_col = lambda x: pd.DataFrame(x)[[name for name in cols if name in pd.DataFrame(x).columns]]
    feat_beats = df_["beats"].apply(lambda x: func_mean(verify_col(x)) if x else x).apply(lambda x: pd.Series(x))
    feat_beats.columns = [i+"_beats"for i in feat_beats.columns]

    #expand "captures" column
    choose_index = lambda x: np.argmin(pd.isnull(pd.DataFrame(x)).apply(lambda x: sum(x), axis = 1))
        #choose index with the most info
    captures_df_ = pd.concat([pd.DataFrame(i).ix[[choose_index(i)]] if len(pd.DataFrame(i))>1 else pd.DataFrame(i)
                              for i in df_["captures"]]).reset_index()
    captures_df_["phtime"] = captures_df_["index"].map(
        lambda x: timestamp_converter(ms_to_timestamp(int(x)), "Asia/Manila"))

    #append feat_beats and captures_data_forced to df_
    datacapbeat_forced = pd.concat([df_, captures_df_, feat_beats], axis = 1)

    return datacapbeat_forced

''' remove since this is not being used as of the moment
def load_data():
    try:
        conn = psycopg2.connect(dbname='amiigo_platform_sherpa',
                            user='savvy_read',
                            host='10.1.0.141',
                            password='savvy_read',
                            port = '15610')
    except:
        print("failed to connect")
        return None

    q = """select * from passive_passivedata where is_manual='True' order by created_at"""
    passive_data = pd.read_sql_query(q, conn)

    #remove rows samples with no captures data
    #passive_data = passive_data[~pd.isnull(passive_data["captures"])]
    passive_data.dropna(subset = ["captures"], inplace = True)
    passive_data.reset_index(drop = True, inplace = True)
    passive_data = expand(passive_data)


    passive_data["_vec.ir"] = passive_data["_vec.ir"].apply(lambda x: np.array(x))
    passive_data["_vec.red"] = passive_data["_vec.red"].apply(lambda x: np.array(x))

    #remove rows with _vec.ir and _vec.red columns containing Nans
    notNan = passive_data["_vec.ir"].map(lambda x: np.all(x))*\
        passive_data["_vec.red"].map(lambda x: np.all(x))
    passive_data = passive_data[notNan==notNan]

    #flipping the signals
    passive_data["_vec.ir"] = passive_data["_vec.ir"].apply(lambda x: -x)
    passive_data["_vec.red"] = passive_data["_vec.red"].apply(lambda x: -x)

    passive_data.set_index("created_at", inplace = True)
    return passive_data


#load the recorded data from google spreadsheet
#https://github.com/burnash/gspread

def load_excel(oauth_cred):
    scope = ['https://spreadsheets.google.com/feeds']
    credentials = ServiceAccountCredentials.from_json_keyfile_name(oauth_cred, scope)

    url = "https://docs.google.com/spreadsheets/d/1Ziz9hYojbfm_rhlPg0OHUveeU5-NfHAPodPfFd4BAXM/edit#gid=0"
    gc = gspread.authorize(credentials)
    sheet = gc.open_by_url(url).sheet1

    df = pd.DataFrame(sheet.get_all_values())
    df.columns = df.ix[0]
    df.drop(0, inplace = True)
    df.reset_index(drop = True, inplace  = True)

    df["Cuff Blood Pressure_SV"] = pd.to_numeric(df["Cuff Blood Pressure"]\
                                        .apply(lambda x: x.split("/")[0] if x else ""), errors="coerce")
    df["Cuff Blood Pressure_DV"] = pd.to_numeric(df["Cuff Blood Pressure"]\
                                        .apply(lambda x: x.split("/")[1] if x else ""),errors="coerce")
    df["Heart Rate"] = pd.to_numeric(df["Heart Rate"], errors = "coerce")
    df["Fingertip SpO2"] = pd.to_numeric(df["Fingertip SpO2"], errors = "coerce")
    df["Respiration Rate"] = pd.to_numeric(df["Respiration Rate"], errors="coerce")
    df["Amiigo PPG BPM"] = pd.to_numeric(df["Amiigo PPG BPM"], errors="coerce")
    df["Calculated SpO2"] = pd.to_numeric(df["Calculated SpO2"], errors = "coerce")

    return df
'''

## Additional features for beat classification:
def diffheights_troughtopeak_ratio(df):
    diff = df.groupby(level = "cycle_id")\
                    .apply(lambda x: x["PPG"].values)\
                    .apply(lambda x: (x[0], np.max(x), x[-1]))\
                    .apply(lambda x: (x[1]-x[0], x[1]-x[2]))\
                    .apply(lambda x: np.abs(x[1]-x[0]))\
                    .values
    meddiff = np.nanmedian(diff)
    return diff/meddiff

def amplitudes_ratio(df):
    heights = df.groupby(level = "cycle_id")\
                .apply(lambda x: x["PPG"].values)\
                .apply(lambda x: (max(x)-min(x)))\
                .values
    medheights = np.nanmedian(heights)
    return heights/medheights

def duration_ratio(df):
    duration = df.groupby(level = "cycle_id")\
                 .apply(lambda x: x.index.get_level_values(1))\
                 .apply(lambda x: (x[-1] - x[0]).total_seconds())\
                 .values
    medduration = np.nanmedian(duration)
    return duration/medduration

def max_corr_coeff(df):
    normbeats = normalize_cycle_length(df, 64)["PPG_beat"]
    normbeats_filt = np.vstack(normbeats.ix[[~np.isnan(i).any() for i in normbeats]])
    medbeat = np.median(normbeats_filt, axis = 0)
    return np.array([np.max(ss.correlate(medbeat/np.sum(abs(medbeat)),
                                             i/np.sum(abs(i)))) for i in normbeats])

def corr_coeff_pearsonsr(df):
    df_new = normalize_cycle_length(df, 64)
    df_new_filt = df_new.ix[[~np.isnan(i).any() for i in df_new["PPG_beat"]]]
    median_beat = np.nanmedian(np.vstack(df_new_filt['PPG_beat']), axis = 0)
    corr = df_new.groupby(level = 'cycle_id')\
                .apply(lambda x: (x['PPG_beat'].values))\
                .apply(lambda x: pearsonr(x[0], median_beat))\
                .apply(lambda x: x[0])\
                .values
    return corr

def variance_ratio(df):
    variance = df.groupby(level = "cycle_id")\
                .apply(lambda x: x["PPG"].values)\
                .apply(np.var)\
                .values
    medvariance = np.median(variance)
    return variance/medvariance

def starttopeak_slope_ratio(df):
    slope = df.groupby(level = "cycle_id")\
                .apply(lambda x: x["PPG"].values)\
                .apply(lambda x: (np.max(x), np.argmax(x), x[0]))\
                .apply(lambda x: abs((x[0]-x[2])/x[1]))\
                .values

    medslope = np.nanmedian(slope)

    return slope/medslope

def peaktoend_slope_ratio(df):
    slope = df.groupby(level = "cycle_id")\
                .apply(lambda x: x["PPG"].values)\
                .apply(lambda x: (np.max(x), np.argmax(x), x[-1], len(x)))\
                .apply(lambda x: abs((x[0]-x[2])/(x[1] - x[3])))\
                .values
    medslope = np.nanmedian(slope)

    return slope/medslope

def amplitude_duration_ratio(df):
    amp_dur = df.groupby(level = "cycle_id")\
                .apply(lambda x: x["PPG"].values)\
                .apply(lambda x: (np.max(x), np.min(x), len(x)))\
                .apply(lambda x: abs(x[0]-x[1])/x[2])\
                .values
    medamp_dur = np.nanmedian(amp_dur)

    return amp_dur/medamp_dur

def beat_area_ratio(df):
    area = df.groupby(level = "cycle_id")\
                .apply(lambda x: (x["PPG"].values, np.min(x["PPG"].values)))\
                .apply(lambda x: x[0] - x[1])\
                .apply(np.sum)\
                .values
    medarea = np.median(area)
    return area/medarea
def energy_ratio(df):
    energy = df.groupby(level = "cycle_id")\
                .apply(lambda x: x["PPG"].values)\
                .apply(lambda x: x**2)\
                .apply(np.sum)\
                .values
    medenergy = np.median(energy)

    return energy/medenergy
