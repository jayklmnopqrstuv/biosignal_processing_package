import numpy as np
from scipy.signal import resample
import warnings
from rndSignal.plotting.signal_plots import plot_segments, plot_signal
from rndSignal.preprocessing.signal_filter import filter_ppg, filter_ecg
from rndSignal.plotting.signal_plots import plot_epoch


def signal_downsample(sig, sample_rate,show_plot = False, **kwargs):
    """Transform the signal by downsampling using `scipy.signal.resample`. 
    Resample x to num samples using Fourier method along the given axis.
    
    Parameters
    ----------
    sig : list
        1 dimensional signal
    sample_rate : float
        Sampling frequency of the signal
    show_plot : bool
        If True, it will show plot of the original and downsampled signal
    **kwargs: should only select one arg for downsampling
        downsample_rate (float): target sampling frequency after downsampling
        downsample_points (integer): total number of signal points after downsampling
        downsample_factor (integer): divides the sampling frequency by n factor
        
    Returns
    -------
    downsampled_signal: list
        signal after downsampling
    """
  
    
    if kwargs.__len__() == 0:
        raise NameError("No defined 'dowsample_rate' or 'downsample_points' or 'downsampling_factor'")
    if kwargs.__len__() > 1:
        raise NameError("Define only one from 'dowsample_rate' or 'downsample_points' or 'downsampling_factor' parameters")
   
    if 'downsample_points' in kwargs:
        samps = kwargs.get('downsample_points')
        if samps > len(sig):
            return ValueError("The resulting length of signal after downsampled should be less than the length of the input signal!")  
        ds_rate = len(sig)/samps
        
    if 'downsample_rate' in kwargs: 
        ds_rate = kwargs.get('downsample_rate')
        if ds_rate > sample_rate:
            return ValueError("Downsampling rate should be less than sample rate!")
            
        secs = len(sig)/sample_rate
        samps = secs * ds_rate
        if samps.is_integer():
            samps = int(samps)
        else:
            samps = int(round(samps))
            ds_rate = samps/secs
            warnings.warn(f"Downsample rate returns a non-integer sample points, setting new downsample rate to {ds_rate}Hz")
      
        
    if 'downsample_factor' in kwargs:
        ds_factor = kwargs.get('downsample_factor')
        R_factor = sample_rate/ds_factor
        secs = len(sig)/sample_rate
        samps = secs * R_factor
        if samps.is_integer():
            samps = int(samps)
        else:
            samps = int(round(samps))
            warnings.warn(f"Downsample factor returns a non-integer sample points, setting a new downsample factor")
        ds_rate = sample_rate/ds_factor
        
        
    downsampled_signal = resample(sig, samps)
    if show_plot:
        plot_signal([sig,downsampled_signal],[sample_rate,ds_rate], labels = ["raw","downsampled"], 
                    x_axis_label = 'Time (seconds)', grid_plot = True, grid_lines = 2, grid_columns = 1)
        
    return downsampled_signal
        

def segment_signal(sig, sample_rate, window_time = 25, show_plot = False, overlap = False, **kwargs):
    """Segment the signal
    
    Parameters
    ----------
    sig : list
        1 dimensional signal
    sample_rate : float
        Sampling frequency of the signal
    window_time : float
        Duration of each segment (seconds)
    show_plot : boolean
        If True, it will show plot of the segments as box annotations
    overlap : boolean
        If True, method will be a sliding window.
    **kwargs:
        overlap_time: Sliding window time (seconds)
    
    Returns
    -------
    segments: list
        chunks of the signal (list) after segmentation stored in a list
    """
    if overlap:
        if kwargs.__len__() == 0:
            raise ValueError("No defined 'overlap_time'")
        step = kwargs.get('overlap_time') * sample_rate
    else:
        step = window_time * sample_rate
    size = sample_rate * window_time 
     
    
    segments = [sig[i : i + size] for i in range(0, len(sig), step)]
    lengths = map(len,segments)
    if len(set(map(len,segments)))!=1:
        print("There are segments with different window length")
  
    if show_plot:
        plot_segments(sig, segments, sample_rate, overlap_time = kwargs.get('overlap_time'))
        
    return segments


def epoch_signal(sig_list,epoch_list,signal_type = None, sample_rate = None, plot = False, apply_filter = False):
    """Create a dictionary of the epoch and the corresponding signal points
  
    Parameters
    ----------
    sig_list : list
        Unsegmented signal
    epoch_list : list
        Corresponding epoch/event of the signal point (should have the same length with sig_list) 
    sample_rate : int
        Sampling frequency of the signal
    signal_type: string
        Specify the type of signal, current acceptable signal type: 'ECG' and 'PPG'.
        This argument is needed if 'apply_filter' = True
    plot: boolean
        Create a visualization of the epoch
    apply_filter: boolean
        If True, function will perform filtering of the signal first
    
    Returns
    -------
    unique_epoch: dictionary
        Dictionary of the unique events (ordered) and the signal points
    
    """   
    # Transform the epoch names (by order)
    if plot and sample_rate is None:
        return ValueError("Need to input sample_rate")
    if apply_filter and sample_rate is None:
        return ValueError("Need to input sample_rate")
    
    if apply_filter and signal_type is None:
        return ValueError("Define if signal is `ECG` or `PPG`")
    
    if apply_filter:
        if signal_type.upper() == "ECG":
            sig_list = filter_ecg(sig_list,sample_rate)
        elif signal_type.upper() == "PPG":
            sig_list = filter_ppg(sig_list,sample_rate)
        else:
            return ValueError("Unknown `signal_type`, choose between PPG or ECG")
            
    epoch_arr = []
    current_epoch = 0
    unique_epoch = {}
    n = 0
    for i in epoch_list:
        if i != current_epoch:
            n = n + 1
            current_epoch = i
        else:
            pass
        epoch_arr.append(str(n)+ "_" + i)

    _idx =[epoch_arr.index(x) for x in sorted(set(epoch_arr))]
    
    for i in range(len(_idx)):
        try:
            unique_epoch[epoch_arr[_idx[i]]] = sig_list[_idx[i]:_idx[i+1]]
        except:
            unique_epoch[epoch_arr[_idx[i]]] = sig_list[_idx[i]:]
            continue

    if plot:
        plot_epoch(sig_list,unique_epoch,sample_rate)
          
    return unique_epoch    
    
def main():
    pass
    
if __name__ == "__main__":
    main()
