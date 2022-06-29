import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal
from sklearn.preprocessing import minmax_scale
import warnings
import itertools
import math
from rndSignal.signal_quality import deepbeat_model
from rndSignal.preprocessing.signal_transform import signal_downsample

deepbeat = deepbeat_model._load_model()
# Define SQI thresholds

deepbeat_sqi_threshold = 0.5
template_sqi_threshold = 0.5
 
        
def _ppg_sqi_dict(output_type,sqi,signal_len):
    """Create values that can be plotted with the signal to indicate level of signal quality
    # To do, once plotting functions have been finalized
    
    """
    pass
    
    
def ppg_signal_quality(ppg_signal, sample_rate, method = "deepbeat",output_type = "categorical", **kwargs):
    """Assess the input signal quality of an input ppg signal based 
    from various methods. Input ppg signal is assumed to have been preprocessed - noise has 
    been filtered out and motion artifacts have been reduced.
    
    Parameters
    ----------
    ppg_signal : list
        1 dimensional ppg signal
    sample_rate : float
        Sampling frequency of the ppg signal
    method : string
        Indicates method to assess signal quality
    output_type :string 
        signal assessment result if `categorical` or `continuous`
    method : string
        - *deepbeat* (continuous/categorical) (default)
          documentation: https://arxiv.org/ftp/arxiv/papers/2001/2001.00155.pdf
          A deep learning approach to assess and categorize a segmented ppg signal 
          if Excellent, Acceptable, or Noise/Not Acceptable
          Trained model requires a ppg length of 25 seconds      
        - *template_matching* (continuous/categorical) 
          adapting methods implemented by Li & Huang: 
          Li, H. & Huang, S. (2020). A High-Efficiency and Real-Time Method for Quality Evaluation of PPG Signals. 
          Retrieved from https://iopscience.iop.org/article/10.1088/1757-899X/711/1/012100/pdf
          Modification: SSF method was replaced with the Neurokit2 peak detection method by Elgendi
        - *SNR* (continuous) (not yet included)
        - *sSQI* (continuous): (not yet included)
    
    Returns
    -------
    sqi_info: str/list
        This may vary per model:
    
        - deepbeat : str
            categorical value ("Excellent", "Acceptable", "Not Acceptable")
        
        - template_matching : list
            
            - sqi_vals_continuous : list
                SQI value for each beat detected
              
            - sqi_vals_categorical : list
                SQI category for each beat detected(Acceptable/Not Acceptable) based from 
                `template_sqi_threshold` value
            
            - sqi_mean : float
                Average value of sqi_vals_continuous
            
            - sqi_category : string
                Categorize if the input segment is Acceptable/Not acceptable using the `sqi_mean` 
    """
    
    
    if method == 'deepbeat':
        sqi_info = _ppg_signal_quality_deepbeat(
            ppg_signal,
            sample_rate,
            "categorical"
        )
        return sqi_info
    if method == 'template_matching':
        sqi_info = _ppg_signal_quality_matching(ppg_signal,sample_rate,output_type)
        return sqi_info
    
    
    if method == 'adaptive_template_matching':
        print("This method may not work well for short signals, try using `savvyppg.ppgSignal()` and visualize the signal using `ppg_plot_quality()`")
        ppg_template = kwargs.get('template_signal')
        sqi_info = _ppg_signal_quality_matching(
            ppg_signal,sample_rate,
            output_type, adaptive = True, 
            template_signal = ppg_template
        )
        return sqi_info


def _ppg_signal_quality_deepbeat(ppg_signal,sample_rate,output_type, **kwargs):
    """Main method to perform signal quality calculation using deep learning approach(DeepBeat)
    
    Parameters
    ----------
    ppg_signal : list
        Input ppg signal, should have duration = 25 seconds
    sample_rate : float
        Sampling frequency of the ppg signal
    
    Returns
    -------
    sqi_calc : list
        Probabilities of "Excellent", "Acceptable", "Not Acceptable"
    sqi_cat : string
        Signal quality classification
    """
    
    #Trained model only accepts 25 second ppg signal
    if len(ppg_signal)/sample_rate != 25:
        return ValueError("Input signal is not equal to 25 seconds")
    # perform downsampling
    ppg_signal = signal_downsample(ppg_signal,sample_rate, downsample_rate = 32)
    # normalize signal
    ppg_signal = minmax_scale(ppg_signal, feature_range=(0, 1), axis=0, copy=True)
    # reshape for the model
    ppg_signal = [[ppg_signal.reshape(-1, 1)]]
    
    predictions_qa, predictions_r = deepbeat.predict(ppg_signal)
    if output_type == 'continuous': 
        #sqi_calc = predictions_r[0][0]
        #return sqi_calc
        raise ValueError("Continuous values are not yet supported by this method")
        
        
    else:
        sqi_calc = predictions_r[0][0]
        val = np.argmax(predictions_qa, axis=1)
        if val == 2:
            sqi_cat = "Excellent"
        elif val == 1:
            sqi_cat = "Acceptable"
        else:
            sqi_cat = "Not Acceptable"
            
        return sqi_cat

    
def _ppg_matching_template(simulate = False, method = 'mean',**kwargs):
    """Create a ppg template from either simulated or a ppg segment
    Note: This is a miscellaneous function to create a reference signal 
    """
    
    sample_rate = kwargs.get('sample_rate')
    if simulate:
        ppg_signal = nk.ppg_simulate(
            duration = kwargs.get('duration'), 
            sampling_rate = sample_rate,
            heart_rate = kwargs.get('heart_rate')
        )
    else:
        ppg_signal = kwargs.get('ppg_signal')
        
    if kwargs.get('downsample'):
        ppg_signal = signal_downsample(ppg_signal,
                                    sample_rate, 
                                    downsample_rate = kwargs.get('downsample_rate')
                                   )
 
    ppg_segments = _ppg_segments_from_peaks(ppg_signal, sample_rate, 
                                            kwargs.get('floor_idx'),
                                            kwargs.get('ceil_idx')
                                           )
    ppg_segments = _ppg_padding_segments(ppg_segments)
    ppg_template = _ppg_template_from_segments(ppg_segments)
   
    return ppg_segments, ppg_template

def _ppg_template_from_segments(ppg_segments, method = 'mean'):
    """Create a reference template from the segments
    
    Parameters
    ----------
    ppg_segments : array
        Segmented ppg beats with equal length
    method : string
        Method to calculate the ppg template
    
    Returns
    -------
    ppg_template : list
        Calculated ppg beat from the segments 
    """
    
     # include other methods in the future
    if method == 'mean':
        ppg_template = np.nanmean(ppg_segments, axis = 0)
    return ppg_template


def _ppg_padding_segments(ppg_segments, fillval = np.nan):
    """Pad the segments to have equal length
    
    Parameters
    ----------
    ppg_segments : array
        Segmented ppg beats
    fillval : method
        Default = nan values
        
    Returns
    -------
    ppg_out : array
        PPG beats with equal length
    
    """
    ppg_out = np.array(list(itertools.zip_longest(*ppg_segments,fillvalue=np.nan))).T
    return ppg_out


def _ppg_segments_from_peaks(ppg_signal,sampling_rate,*idx):
    """Segment the ppg signals based from the peaks detected.
    
    Parameters
    ----------
    ppg_signal : list
        Input signal (downsampled) for segmentation
    sampling_rate : float
        Sampling frequency
    *idx: - ppg segment length
        idx[0](int): floor_idx, number of points to be added before the detected peak 
        idx[0=1](int): floor_idx, number of points to be added after the detected peak
        
    Returns
    -------
    ppg_segments : array
        Segmented from `ppg_signal` with length defined by *idx
    
    """
    
    if None in idx:
        floor_idx = 20
        ceil_idx = 40
    else: 
        floor_idx = idx[0]
        ceil_idx = idx[1]
    ppg_df, ppg_info = nk.ppg_process(ppg_signal,sampling_rate =sampling_rate)
    # Create segments based from the peaks
    peaks_idx = ppg_df.loc[ppg_df['PPG_Peaks'] == 1].index.tolist()
    ppg_clean = ppg_df.PPG_Clean.tolist()
    ppg_segments =[]
    for i in peaks_idx:
        tmp_arr = np.array(ppg_clean[i-floor_idx:i+ceil_idx])
        ppg_segments.append(tmp_arr)
    return np.array(ppg_segments)
        
    
def _ppg_calculate_template_correlation(ppg_segments,reference_signal):
    """Calculate the normalized cross correlation of the reference signal and per ppg beat
    then calculate the signal quality index. The overall SQI of the input ppg segment is 
    obtained by getting the mean of all the SQIs.
    
    Parameters
    ----------
    ppg_segments : array
        Array of the ppg beats segmented
    reference_signal : list
        The template signal 
    
    Returns
    -------
    sqi_vals_continuous : list
        Calculated SQI for each segment
    sqi_vals_categorical : list
        Calculated SQI for each segment
    sqi_mean : float
        Average SQI value
    """
    # Check if ppg_segments length = reference_signal
    if len(reference_signal) != max([len(i) for i in ppg_segments]):
        return ValueError("Reference signal doesnt have the same dimension with the ppg segments!")
    sqi_vals_continuous = []
    sqi_vals_categorical = []
  
    for segment in ppg_segments:
        reference_norm = (reference_signal - np.mean(reference_signal)) / (np.std(reference_signal) * len(reference_signal))
        segment_norm = (segment - np.nanmean(segment)) / (np.nanstd(segment))
        segment_norm = segment_norm[~np.isnan(segment_norm)]
        sqi_calc = np.correlate(reference_norm, segment_norm)[0]
        sqi_vals_continuous.append(sqi_calc)
        sqi_cat = "Acceptable" if sqi_calc> template_sqi_threshold else "Not Acceptable"
        sqi_vals_categorical.append(sqi_cat)
 
    sqi_mean = np.nanmean(sqi_vals_continuous)
    sqi_category = "Acceptable" if sqi_mean > template_sqi_threshold else "Not Acceptable"
    return sqi_mean,sqi_category,sqi_vals_continuous,sqi_vals_categorical,  
    
    
def _ppg_signal_quality_matching(ppg_signal,sample_rate, output_type, adaptive = False, downsample_rate = 100, **kwargs):
    """Main method to perform signal quality calculation using template matching
    
    Parameters
    ----------
    ppg_signal : list
        Input ppg signal, should be greater than 10 seconds
    sample_rate : float
        Sampling frequency of the ppg signal
    adaptive : bool
        If True, it will use an adaptive method of creating a template ppg signal 
        from the previous input
    downsample_rate : int 
        Default = 100Hz (if not equal to 100Hz, floor_idx and ceil_idx should be modified)
    **kwargs:
        previous_segments(array): contains list of the previous segments/reference signals that will be
                                  used for adaptive template matching. This should have the same dimension
                                  with the current ppg segments
        floor_idx(int): number of signal points to be included before the peak of a beat
        ceil_idx(idx): number of signal points to be included after the peak of a beat

    Returns
    -------
    sqi_vals_continuous : list
        SQI value for each beat detected
    sqi_vals_categorical : list
        SQI category for each beat detected(Acceptable/Not Acceptable) based from 
        `template_sqi_threshold` value
    sqi_mean : float
        Average value of sqi_vals_continuous
    sqi_category : string
        Categorize if the input segment is Acceptable/Not acceptable using the `sqi_mean` 
    """
    
    if len(ppg_signal)/sample_rate < 10: #acceptable ppg duration to implement this method 
        raise ValueError("Input signal is too short")
    if downsample_rate != 100: warnings.warn(f"Modify 'floor_idx' and 'ceil_idx' by passing these arguments in the function")
    ppg_signal = signal_downsample(ppg_signal,sample_rate, downsample_rate = downsample_rate)
    ppg_segments = _ppg_segments_from_peaks(ppg_signal, downsample_rate, kwargs.get('floor_idx'),kwargs.get('ceil_idx'))
    ppg_segments = _ppg_padding_segments(ppg_segments)
    
    if not adaptive:
        # create reference signal
        ppg_template = _ppg_template_from_segments(ppg_segments)
        return _ppg_calculate_template_correlation(ppg_segments,ppg_template)
        
    else:
        if kwargs.get('previous_segments'):
            ppg_previous_segments = kwargs.get("previous_segments").append(ppg_segments)
            ppg_template = _ppg_template_from_segments(ppg_previous_segments)
            return (_ppg_calculate_template_correlation(ppg_segments,ppg_template), ppg_template)
        else:
            return ValueError(f'No reference segments, include a list of previous ppg segments downsampled at {downsample_rate} Hz')
    
    
        

    
        
    

    
    
    
    
