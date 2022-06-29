import neurokit2 as nk
import pandas as pd
import numpy as np
from neurokit2.signal.signal_formatpeaks import _signal_from_indices
from neurokit2.misc import as_vector
from rndSignal.plotting.ppg_plotting import ppg_plot_bpm
from rndSignal.savvyppg.ppg_module import ppgSignal
from rndSignal.plotting.signal_plots import plot_epoch_hr
from rndSignal.preprocessing.signal_transform import epoch_signal

 
def _return_ppg_signal(_obj):
    """Miscellaneous function to convert the input ppg signal to a numpy array
    since other functions require an input of type `np.array`.
    It checks if  if the input is a module type which is the output after using the class `ppgSignal()` 
    for processing the ppg signal, a pandas DataFrame, or a numpy array.
    
    Parameters
    ----------
    _obj : object
        contains PPG signal 
        
    Returns
    -------
    sig : np.array
        converted PPG signal
    
    """
    if isinstance(_obj,ppgSignal):
        sig = _obj.data_ppg.tolist()
        sig = np.array(sig)
    elif isinstance(_obj, pd.DataFrame):
        sig = _obj['PPG'].tolist()
        sig = np.array(sig)
    elif isinstance(_obj,np.ndarray):
        sig = _obj
    else:
        sig = np.array(_obj)

    return sig


def _nk_process_ppg(ppg_sig,sr):
    """Miscellaneous function to process the input ppg signal using Neurokit2 function
    The cleaning part of the function has been omitted since it has been assumed that 
    input signal has been cleaned by `filter_ppg` or `ppgSignal()`
    Refer to: https://neurokit2.readthedocs.io/en/latest/_modules/neurokit2/ppg/ppg_process.html
    """
    sig = as_vector(ppg_sig)
    
    #Omit cleaning since data is assumed to be cleaned using filter_ppg() or ppgSignal()
    #ppg_cleaned = nk.ppg_clean(sig, sampling_rate=sampling_rate) 
    
    info = nk.ppg_findpeaks(sig, sampling_rate=sr)
    info['sampling_rate'] = sr  # Add sampling rate in dict info

    # Mark peaks
    peaks_signal = _signal_from_indices(info["PPG_Peaks"], desired_length=len(sig))

    # Rate computation
    rate = nk.signal_rate(info["PPG_Peaks"], sampling_rate=sr, desired_length=len(sig))

    # Prepare output
    signals = pd.DataFrame(
        {"PPG_Raw": ppg_sig, "PPG_Rate": rate, "PPG_Peaks": peaks_signal}
    )

    return signals#, info
    
    #return nk.ppg_process(sig, sampling_rate = sr)[0]


def ppg_heart_rate(savvyppg_obj, sr, show = False):
    """Wrapper function for Neurokit2's hrv_time() function used for generating 
    heart rate information for PPG data.
    
    Parameters
    ----------
    savvyppg_obj : object
        PPG signal
    sr : int
        Sampling rate of the signals
    show : bool
        If true, features generated are visualized.
    
    Returns
    -------
    PPG Heart Rate : list
      contains the instantaneous heart rate
    hrv_time_features : DataFrame
        Pandas DataFrame that contains heart rate features:
        
        - *mean_bpm*: Average heart rate of the input ppg signal
        
        - *sd_bpm*: Standard deviation of the input ppg signal
        
        - *max_bpm*: Maximum bpm from the estimated bpm values of the ppg signal
        
        - *min_bpm*: Minimum bpm from the estimated bpm values of the ppg signal
    
    """
    
    sig = _return_ppg_signal(savvyppg_obj)   
    
    nk_proc_ppg = _nk_process_ppg(sig, sr)
    #peaks_idx = nk_proc_ppg.index[nk_proc_ppg["PPG_Peaks"] == 1].tolist()

    mean_bpm = np.nanmean(nk_proc_ppg["PPG_Rate"])
    sd_bpm = np.nanstd(nk_proc_ppg["PPG_Rate"])
    max_bpm = np.nanmax(nk_proc_ppg["PPG_Rate"])
    min_bpm = np.nanmin(nk_proc_ppg["PPG_Rate"])
    
    if show:
        ppg_plot_bpm(nk_proc_ppg,sr)
    nk_proc_ppg = nk_proc_ppg[["PPG_Raw","PPG_Rate"]]   
    
    heart_rate_features = pd.DataFrame(
        {'mean_bpm': mean_bpm,'sd_bpm': sd_bpm,'max_bpm': max_bpm, 'min_bpm' : min_bpm },
        index = [0]
    )
    
    return(
        #nk_proc_ppg.rename(columns = {"PPG_Raw": "PPG"}), 
        list(nk_proc_ppg["PPG_Rate"])
        heart_rate_features
    )
        
     
def ppg_hrv_time(savvyppg_obj,sr,show=False):
    """Wrapper function for Neurokit2's hrv_time() function used for generating 
    time domain HRV metrics for PPG data.
    
    Parameters
    ----------
    savvyppg_obj : object
        PPG signal
    sr : int
        Sampling rate of the signals
    show : bool
        If true, features generated are visualized.
    
    Returns
    -------
    hrv_time_features : DataFrame
        Pandas DataFrame that contains time domain HRV metrics:
        
        - *RMSSD*: The square root of the mean of the sum of successive differences between 
          adjacent RR intervals. It is equivalent (although on another scale) to SD1, and
          therefore it is redundant to report correlations with both (Ciccone, 2017).
        
        - *MeanNN*: The mean of the RR intervals.
        
        - *SDNN*: The standard deviation of the RR intervals.
        
        - *SDSD*: The standard deviation of the successive differences between RR intervals.
        
        - *CVNN*: The standard deviation of the RR intervals (SDNN) divided by the mean of the RR
          intervals (MeanNN).
        
        - *CVSD*: The root mean square of the sum of successive differences (RMSSD) divided by the
          mean of the RR intervals (MeanNN).
        
        - *MedianNN*: The median of the absolute values of the successive differences between RR intervals.
        
        - *MadNN*: The median absolute deviation of the RR intervals.
        
        - *HCVNN*: The median absolute deviation of the RR intervals (MadNN) divided by the median
          of the absolute differences of their successive differences (MedianNN).
        
        - *IQRNN*: The interquartile range (IQR) of the RR intervals.
        
        - *pNN50*: The proportion of RR intervals greater than 50ms, out of the total number of RR intervals.
        
        - *pNN20*: The proportion of RR intervals greater than 20ms, out of the total number of RR intervals.
        
        - *TINN*: A geometrical parameter of the HRV, or more specifically, the baseline width of
          the RR intervals distribution obtained by triangular interpolation, where the error of least
          squares determines the triangle. It is an approximation of the RR interval distribution.
          
        - *HTI*: The HRV triangular index, measuring the total number of RR intervals divded by the
          height of the RR intervals histogram.
    """
    sig = _return_ppg_signal(savvyppg_obj) 
    #peaks = _nk_process_ppg(sig, sr)
    peaks = nk.ppg_findpeaks(sig, sampling_rate=sr)
    hrv_time_features = nk.hrv_time(peaks, sampling_rate=sr, show=show)
    
    return hrv_time_features


def ppg_hrv_freq(savvyppg_obj,sr,show=False):
    """Wrapper function for Neurokit2's hrv_frequency() function used for generating 
    frequency domain HRV metrics for PPG data.
    
    Parameters
    ----------
    savvyppg_obj : object
        PPG signal
    sr : int
        Sampling rate of the signals
    show : bool
        If true, features generated are visualized.
    
    Returns
    -------
    hrv_freq_features : DataFrame
        Pandas DataFrame that contains frequency domain HRV metrics:
        
        - *ULF*: The spectral power density pertaining to ultra low frequency band i.e., .0 to .0033 Hz
          by default.
        
        - *VLF*: The spectral power density pertaining to very low frequency band i.e., .0033 to .04 Hz
          by default.
        
        - *LF*: The spectral power density pertaining to low frequency band i.e., .04 to .15 Hz by default.
        
        - *HF*: The spectral power density pertaining to high frequency band i.e., .15 to .4 Hz by default.
        
        - *VHF*: The variability, or signal power, in very high frequency i.e., .4 to .5 Hz by default.
        
        - *LFn*: The normalized low frequency, obtained by dividing the low frequency power by
          the total power.
          
        - *HFn*: The normalized high frequency, obtained by dividing the low frequency power by
          the total power.
          
        - *LnHF*: The log transformed HF.
    """
    sig = _return_ppg_signal(savvyppg_obj)
    #peaks = _nk_process_ppg(sig, sr)
    peaks = nk.ppg_findpeaks(sig, sampling_rate=sr)
    hrv_freq_features = nk.hrv_frequency(peaks, sampling_rate=sr, show=show)
    
    return hrv_freq_features
    
    
def ppg_hrv_nonlinear(savvyppg_obj,sr, show = False):
    """Wrapper function for Neurokit2's hrv_non-linear() function used for generating 
    nonlinear HRV metrics for PPG data.
    
    Parameters
    ----------
    savvyppg_obj : object
        PPG signal
    sr : int
        Sampling rate of the signals
    show : bool
        If true, features generated are visualized.
    
    Returns
    -------
    hrv_nonlinear_features : DataFrame
        Pandas DataFrame that contains non-linear HRV metrics:
        
        - Characteristics of the Poincaré Plot Geometry:
            
          - *SD1*: SD1 is a measure of the spread of RR intervals on the Poincaré plot
            perpendicular to the line of identity. It is an index of short-term RR interval
            fluctuations, i.e., beat-to-beat variability. It is equivalent (although on another
            scale) to RMSSD, and therefore it is redundant to report correlations with both
            (Ciccone, 2017).

          - *SD2*: SD2 is a measure of the spread of RR intervals on the Poincaré plot along the
            line of identity. It is an index of long-term RR interval fluctuations.

          - *SD1SD2*: the ratio between short and long term fluctuations of the RR intervals
            (SD1 divided by SD2).

          - *S*: Area of ellipse described by SD1 and SD2 (``pi * SD1 * SD2``). It is
            proportional to *SD1SD2*.

          - *CSI*: The Cardiac Sympathetic Index (Toichi, 1997), calculated by dividing the
            longitudinal variability of the Poincaré plot (``4*SD2``) by its transverse variability (``4*SD1``).

          - *CVI*: The Cardiac Vagal Index (Toichi, 1997), equal to the logarithm of the product of
            longitudinal (``4*SD2``) and transverse variability (``4*SD1``).

          - *CSI_Modified*: The modified CSI (Jeppesen, 2014) obtained by dividing the square of
            the longitudinal variability by its transverse variability.

        - Indices of Heart Rate Asymmetry (HRA), i.e., asymmetry of the Poincaré plot (Yan, 2017):

          - *GI*: Guzik's Index, defined as the distance of points above line of identity (LI)
            to LI divided by the distance of all points in Poincaré plot to LI except those that
            are located on LI.

          - *SI*: Slope Index, defined as the phase angle of points above LI divided by the
            phase angle of all points in Poincaré plot except those that are located on LI.

          - *AI*: Area Index, defined as the cumulative area of the sectors corresponding to
            the points that are located above LI divided by the cumulative area of sectors
            corresponding to all points in the Poincaré plot except those that are located on LI.

          - *P*: Porta's Index, defined as the number of points below LI divided by the total
            number of points in Poincaré plot except those that are located on LI.

          - *SD1d and SD1a*: short-term variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).

          - *C1d and C1a*: the contributions of heart rate decelerations and accelerations
            to short-term HRV, respectively (Piskorski,  2011).

          - *SD2d and SD2a*: long-term variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).

          - *C2d and C2a*: the contributions of heart rate decelerations and accelerations
            to long-term HRV, respectively (Piskorski,  2011).

          - *SDNNd and SDNNa*: total variance of contributions of decelerations
            (prolongations of RR intervals) and accelerations (shortenings of RR intervals),
            respectively (Piskorski,  2011).

          - *Cd and Ca*: the total contributions of heart rate decelerations and
            accelerations to HRV.

       - Indices of Heart Rate Fragmentation (Costa, 2017):

          - *PIP*: Percentage of inflection points of the RR intervals series.

          - *IALS*: Inverse of the average length of the acceleration/deceleration segments.

          - *PSS*: Percentage of short segments.

          - *PAS*: IPercentage of NN intervals in alternation segments.

        - Indices of Complexity:

          - *ApEn*: The approximate entropy measure of HRV, calculated by `entropy_approximate()`.

          - *SampEn*: The sample entropy measure of HRV, calculated by `entropy_sample()`.

          - *MSE*: The multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.

          - *CMSE*: The composite multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.

          - *RCMSE*: The refined composite multiscale entropy measure of HRV, calculated by `entropy_multiscale()`.

          - *CorrDim*: The correlation dimension of the HR signal, calculated by `fractal_correlation()`.

          - *DFA*: The detrended fluctuation analysis of the HR signal, calculated by `fractal_dfa()`.

    """
    sig = _return_ppg_signal(savvyppg_obj) 
    #peaks = _nk_process_ppg(sig, sr)
    peaks = nk.ppg_findpeaks(sig, sampling_rate=sr)
    hrv_nonlinear_features = nk.hrv_nonlinear(peaks, sampling_rate = sr,show = show)
    
    return hrv_nonlinear_features

       
    
def ppg_epoch_hr(sig_list, epoch_list, sample_rate, apply_filter  = False, plot = False):
    """Calculate and display the instantaneous heart rate of the ppg signal
    based on Neurokit function. Each signal point (stored as sig_list) should have
    corresponding epoch/event (stored as epoch_list).
   

    Parameters
    ----------
    sig_list : list
        Unsegmented ppg signal
    epoch_list : list
        Corresponding epoch/event of each signal point (should have the same length with sig_list) 
    sample_rate : int
        Sampling frequency of the signal
    plot: boolean
        Create a visualization of the epoch instantaneous heart rate
    apply_filter: boolean
        If True, function will perform filtering of the signal first
    
    Returns
    -------
    _ppg_hr: pd.DataFrame
        Contains columns of the ppg signal, PPG Heart Rate and the epoch the signal point belongs to
    
    """   
    _epoch_dict = epoch_signal(sig_list,epoch_list,signal_type = "PPG", sample_rate = sample_rate,apply_filter = apply_filter, plot = plot)
    sig_list = [_epoch_dict.get(ep) for ep in _epoch_dict.keys()]
    sig_list = np.concatenate(sig_list).ravel()
    
    _ppg_hr = _nk_process_ppg(sig_list,sample_rate)
    _ppg_hr["Epoch"] = epoch_list
    
    if plot:
        plot_epoch_hr(_ppg_hr["PPG_Rate"],_epoch_dict,sample_rate)
    
    return _ppg_hr    
    
    
