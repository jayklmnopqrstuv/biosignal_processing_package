# Import libraries
import biosignalsnotebooks as bsnb
import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy.integrate as integr
from rndSignal.signal_quality.snr import snr
from rndSignal.plotting.ecg_plotting import ecg_plot_bpm
from rndSignal.plotting.signal_plots import plot_epoch_hr
from rndSignal.preprocessing.signal_transform import epoch_signal
from neurokit2.signal import signal_rate, signal_sanitize
from neurokit2 import ecg_peaks

def hrv_parameters(sig, sr, signal = False, in_seconds = False):
    """This function extracts HRV parameters from time, poincare, and frequency domains.
    
    Parameters
    ----------
    sig : array
        The filtered ECG signal data.
    sr : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    signal : boolean
        If True, then the data argument contains the set of the ECG acquired samples.
    in_seconds : boolean
        If the R peaks list defined as the input argument "data" contains the sample numbers where
        the R peaks occur, then in_seconds needs to be False.
        
    Returns
    -------
    out : dict
        Dictionary with HRV parameters values, with keys:
        
        - *MaxRR* : Maximum RR interval

        - *MinRR* : Minimum RR interval

        - *AvgRR* : Average RR interval

        - *MaxBPM* : Maximum RR interval in BPM

        - *MinBPM* : Minimum RR interval in BPM

        - *AvgBPM* : Average RR interval in BPM

        - *SDNN* : Standard deviation of the tachogram

        - *SD1* : Square root of half of the sqaured standard deviation of the differentiated tachogram

        - *SD2* : Square root of double of the squared SD1 minus the SD2 squared

        - *SD1/SD2* : quotient between SD1 and SD2

        - *NN20* : Number of consecutive heartbeats with a difference larger than 20 ms

        - *pNN20* : Relative number of consecutive heartbeats with a difference larger than 20 ms

        - *NN50* : Number of consecutive heartbeats with a difference larger than 50 ms

        - *pNN50* : Relative number of consecutive heartbeats with a difference larger than 50 ms

        - *ULF_Power* : Power of the spectrum between 0 and 0.003 Hz

        - *VLF_Power* : Power of the spectrum between 0.003 and 0.04 Hz

        - *LF_Power* : Power of the spectrum between 0.04 and 0.15 Hz

        - *HF_Power* : Power of the spectrum between 0.15 and 0.40 Hz

        - *LF_HF_Ratio* : Quotient between the values of LF_Power and HF_Power

        - *Total_Power* : Power of the whole spectrum
    """

    ecg_snr = snr(sig, sr, value = True)
    if(ecg_snr < 0):
        raise Exception("The signal is not of good quality.")
    
    out_dict = {}

    # Generation of tachogram
    # Uses modifed function below
    tachogram_data, tachogram_time = _tachogram(sig, sr, signal = signal, in_seconds = in_seconds, 
                                                out_seconds = True)

    # Ectopy Removal
    tachogram_data_NN, tachogram_time_NN = bsnb.remove_ectopy(tachogram_data, tachogram_time)

    # ================================== Time Parameters ==========================================
    # Maximum, Minimum and Average RR Interval
    out_dict["MaxRR"] = np.max(tachogram_data_NN)
    out_dict["MinRR"] = np.min(tachogram_data_NN)
    out_dict["AvgRR"] = np.average(tachogram_data_NN)

    # Maximum, Minimum and Average Heart Rate
    max_hr = 1 / out_dict["MinRR"]  # Cycles per second.
    out_dict["MaxBPM"] = max_hr * 60  # BPM

    min_hr = 1 / out_dict["MaxRR"]  # Cycles per second.
    out_dict["MinBPM"] = min_hr * 60  # BPM

    avg_hr = 1 / out_dict["AvgRR"]  # Cyles per second.
    out_dict["AvgBPM"] = avg_hr * 60  # BPM

    # SDNN
    out_dict["SDNN"] = np.std(tachogram_data_NN)

    # ================================ Poincaré Parameters ========================================
    # Auxiliary Structures
    tachogram_diff = np.diff(tachogram_data)
    sdsd = np.std(tachogram_diff)

    # Poincaré Parameters
    out_dict["SD1"] = np.sqrt(0.5 * np.power(sdsd, 2))
    out_dict["SD2"] = np.sqrt(2 * np.power(out_dict["SDNN"], 2) - np.power(out_dict["SD1"], 2))
    out_dict["SD1/SD2"] = out_dict["SD1"] / out_dict["SD2"]

    # ============================= Additional Parameters =========================================
    tachogram_diff_abs = np.fabs(tachogram_diff)

    # Number of RR intervals that have a difference in duration, from the previous one, of at least
    # 20 ms
    out_dict["NN20"] = sum(1 for i in tachogram_diff_abs if i > 0.02)
    out_dict["pNN20"] = int(float(out_dict["NN20"]) / len(tachogram_diff_abs) * 100)  # % value

    # Number of RR intervals that have a difference in duration, from the previous one, of at least
    # 50 ms
    out_dict["NN50"] = sum(1 for i in tachogram_diff_abs if i > 0.05)
    out_dict["pNN50"] = int(float(out_dict["NN50"]) / len(tachogram_diff_abs) * 100)  # % value

    # =============================== Frequency Parameters ========================================
    # Auxiliary Structures
    freqs, power_spect = bsnb.psd(tachogram_time, tachogram_data)  # Power spectrum

    # Frequency Parameters
    freq_bands = {"ulf_band": [0.00, 0.003], "vlf_band": [0.003, 0.04], "lf_band": [0.04, 0.15],
                  "hf_band": [0.15, 0.40]}
    power_band = {}
    total_power = 0

    band_keys = freq_bands.keys()
    for band in band_keys:
        freq_band = freq_bands[band]
        freq_samples_inside_band = [freq for freq in freqs if freq_band[0] <= freq <= freq_band[1]]
        power_samples_inside_band = [power_val for power_val, freq in zip(power_spect, freqs) if
                                     freq_band[0] <= freq <= freq_band[1]]
        power = np.round(integr.simps(power_samples_inside_band, freq_samples_inside_band), 5)

        # Storage of power inside band
        power_band[band] = {}
        power_band[band]["Power Band"] = power
        power_band[band]["Freqs"] = freq_samples_inside_band
        power_band[band]["Power"] = power_samples_inside_band

        # Total power update
        total_power = total_power + power

    out_dict["ULF_Power"] = power_band["ulf_band"]["Power Band"]
    out_dict["VLF_Power"] = power_band["vlf_band"]["Power Band"]
    out_dict["LF_Power"] = power_band["lf_band"]["Power Band"]
    out_dict["HF_Power"] = power_band["hf_band"]["Power Band"]
    out_dict["LF_HF_Ratio"] = power_band["lf_band"]["Power Band"] / power_band["hf_band"]["Power Band"]
    out_dict["Total_Power"] = total_power

    return out_dict

def _tachogram(data, sample_rate, signal = False, in_seconds = False, out_seconds = False):
    """This function generates the ECG Tachogram. It uses Neurokit's algorithm for detecting R peaks
    for ECG signal.
    
    Parameters
    ----------
    data : list
        ECG signal or R peak list. When the input is a raw signal the input flag signal should be
        True.
    sample_rate : int
        Sampling frequency.
    signal : boolean
        If True, then the data argument contains the set of the ECG acquired samples.
    in_seconds : boolean
        If the R peaks list defined as the input argument "data" contains the sample numbers where
        the R peaks occur, then in_seconds needs to be False.
    out_seconds : boolean
        If True then each sample of the returned time axis is expressed in seconds.
        
    Returns
    -------
    out : list, list
        List of tachogram samples. List of instants where each cardiac cycle ends.
    """

    if signal is False:  # data is a list of R peaks position
        data_copy = data
        time_axis = np.array(data)
        if out_seconds is True and in_seconds is False:
            time_axis = time_axis / sample_rate
    else:  # data is a ECG signal
        # Detection of R peaks
        instant_peaks, rpeaks = nk.ecg_peaks(data, sample_rate, correct_artifacts = True)
        r_peaks = rpeaks['ECG_R_Peaks']
        r_peaks = r_peaks / sample_rate
        time_axis = r_peaks

    # Generation of Tachogram
    tachogram_data = np.diff(time_axis)
    tachogram_time = (time_axis[1:] + time_axis[:-1]) / 2

    return tachogram_data, tachogram_time

def _nk_process_ecg(ecg_signal, sampling_rate, method="neurokit"):
    """Miscellaneous function to process the ECG signal and
    calculate the instantaneous heart rate derived from the peaks
    using Neurokit2
    
    """
    # Sanitize input
    ecg_signal = signal_sanitize(ecg_signal)
    
    #Remove this option
    #ecg_cleaned = ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method)
    
    # R-peaks
    instant_peaks, rpeaks, = ecg_peaks(
        ecg_cleaned=ecg_signal, sampling_rate=sampling_rate, method=method, correct_artifacts=True
    )

    rate = signal_rate(rpeaks, sampling_rate=sampling_rate, desired_length=len(ecg_signal))

    #quality = ecg_quality(ecg_signal, rpeaks=None, sampling_rate=sampling_rate)

    signals = pd.DataFrame({"ECG_Raw": ecg_signal, "ECG_Rate": rate})

    '''
    # Additional info of the ecg signal
    delineate_signal, delineate_info = ecg_delineate(
        ecg_cleaned=ecg_signal, rpeaks=rpeaks, sampling_rate=sampling_rate
    )

    cardiac_phase = ecg_phase(ecg_cleaned=ecg_signal, rpeaks=rpeaks, delineate_info=delineate_info)

    signals = pd.concat([signals, instant_peaks, delineate_signal, cardiac_phase], axis=1)

    # Rpeaks location and sampling rate in dict info
    info = rpeaks
    info['sampling_rate'] = sampling_rate
    '''
    
    return signals


def ecg_heart_rate(sig_list, sr, show = False):
    """Wrapper function for Neurokit2's hrv_time() function used for generating 
    heart rate information for ECG data.
    
    Parameters
    ----------
    sig_list : list
        ECG signal
    sr : int
        Sampling rate of the signals
    show : bool
        If true, features generated are visualized.
    
    Returns
    -------
    ECG Heart Rate : list
        contains the instantaneous heart rate
    hrv_time_features : DataFrame
        Pandas DataFrame that contains heart rate features:
        
        - *mean_bpm*: Average heart rate of the input ecg signal
        
        - *sd_bpm*: Standard deviation of the input ecg signal
        
        - *max_bpm*: Maximum bpm from the estimated bpm values of the ecg signal
        
        - *min_bpm*: Minimum bpm from the estimated bpm values of the ecg signal
    
    """
      
    nk_proc_ecg= _nk_process_ecg(sig_list, sr)

    mean_bpm = np.nanmean(nk_proc_ecg["ECG_Rate"])
    sd_bpm = np.nanstd(nk_proc_ecg["ECG_Rate"])
    max_bpm = np.nanmax(nk_proc_ecg["ECG_Rate"])
    min_bpm = np.nanmin(nk_proc_ecg["ECG_Rate"])
    
    if show:
        ecg_plot_bpm(nk_proc_ecg,sr)
    nk_proc_ecg= nk_proc_ecg[["ECG_Raw","ECG_Rate"]]   
    
    heart_rate_features = pd.DataFrame(
        {'mean_bpm': mean_bpm,'sd_bpm': sd_bpm,'max_bpm': max_bpm, 'min_bpm' : min_bpm },
        index = [0]
    )
    
    return(
        list(nk_proc_ecg["ECG_Rate"]), 
        heart_rate_features
    )

def ecg_epoch_hr(sig_list, epoch_list, sample_rate, apply_filter  = False, plot = False):
    """Calculate and display the instantaneous heart rate of the ecg signal
    based on Neurokit function
   

    Parameters
    ----------
    sig_list : list
        Unsegmented ecg signal
    epoch_list : list
        Corresponding epoch/event of the signal point (should have the same length with sig_list) 
    sample_rate : int
        Sampling frequency of the signal
    plot: boolean
        Create a visualization of the epoch instantaneous heart rate
    apply_filter: boolean
        If True, function will perform filtering of the signal first
    
    Returns
    -------
    _ecg_hr: pd.DataFrame
        Contains columns of the ecg signal, ECG Heart Rate and the epoch the signal point belongs to
    
    """   
    _epoch_dict = epoch_signal(sig_list,epoch_list,signal_type = "ECG", sample_rate = sample_rate,apply_filter = apply_filter, plot = plot)
    sig_list = [_epoch_dict.get(ep) for ep in _epoch_dict.keys()]
    sig_list = np.concatenate(sig_list).ravel()
    
    _ecg_hr = _nk_process_ecg(sig_list,sample_rate)
    _ecg_hr["Epoch"] = epoch_list
    
    if plot:
        plot_epoch_hr(_ecg_hr["ECG_Rate"],_epoch_dict,sample_rate)
    
    return _ecg_hr    

