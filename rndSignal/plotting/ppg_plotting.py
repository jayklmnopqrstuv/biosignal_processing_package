import warnings
from itertools import cycle
import operator
# Base packages used in OpenSignals Tools Notebooks for plotting data
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import BoxAnnotation, LinearAxis, Range1d
output_notebook(hide_banner=True)
import pandas as pd
import numpy as np

# Package for ensuring operations in arrays
#from numpy import average, linspace, random
import biosignalsnotebooks as bsnb
from biosignalsnotebooks.visualise import plot, opensignals_kwargs, opensignals_style, opensignals_color_pallet


def _process_beats_df(beats_df):
    """Miscellaneous function to summarize the `.beats_df` after processing the ppg signal using
    `savvyppg.ppgSignal()`
    
    Parameters
    ----------
    beats_df : DataFrame
        summarized beats information after using `savvyppg.ppgSignal()` to process the ppg signal 
        
    Returns
    -------
    idx: list
        position of the detected beats
    proba: list
        list of bool that determine if the beat detected is abnormal or not
    pearson_coeff: list
        list of the correlation values of the detected beats
    """
    idx = beats_df["index"].to_list()
    proba = beats_df["pred_label"].to_list()
    pearson_coeff = beats_df["morpfeat_corr_coeff_pearsonsr"].to_list()
    
    return(idx,proba,pearson_coeff)   


def _ppg_max_beat(ppg_signal,sample_rate,beats_df):
    """Miscellaneous function to detect the peaks of the signal
    
    Parameters
    ----------
    ppg_signal : list
        ppg signal
    sample_rate: int
        sampling frequency of the signal
    beats_df : DataFrame
        summarized beats information after using `savvyppg.ppgSignal()` to process the ppg signal
        
    Returns
    -------
    peaks_time: list
        list of time (seconds) of the peak value of the detected beats
    peaks_time: list
        list of the peak value of the detected beats
    peaks_idx: list
        positions of the peak value of the detected beats
    
    
    """
    peaks_vals = []
    peaks_time = []
    peaks_idx = []
    idx, _, _= _process_beats_df(beats_df)
    sig = ppg_signal.tolist()
    for i in idx:
        pt = max(sig[i[0]:i[1]])
        pt_idx = (sig[i[0]:i[1]]).index(pt)       
        pt_time = (pt_idx + i[0]) /sample_rate
        peaks_idx.append(i[0] + pt_idx)
        peaks_vals.append(pt)
        peaks_time.append(pt_time)
    
    return peaks_time,peaks_vals, peaks_idx
        
    
def ppg_plot_quality(ppg_signal,beats_df, sample_rate, labels = ["ppg","pearson_coeff"], savvyppg = True, **kwargs):    
    """Plot the signal quality by detecting the abnormal beats and showing the pearson 
    correlation values
    
    Parameters
    ----------
    ppg_signal : list
        ppg signal
    beats_df : DataFrame
        summarized beats information after using `savvyppg.ppgSignal()` to process the ppg signal
    sample_rate: int
        sampling frequency of the signal
    labels : list
        labels that will be used in plotting
    savvyppg: bool
        defaults to True, function only accepts results from `savvyppg.ppgSignal()` 
        
    Returns
    -------
    Plot
        An interactive bokeh plot of the input ppg signal and the signal quality (beat level)
        
    """
    
   
    if savvyppg:
        idx, proba, pearson_coeff = _process_beats_df(beats_df)
        time_x = bsnb.generate_time(ppg_signal, sample_rate)

    
        list_figures = []
        list_figures.append(
            figure(x_axis_label = 'Time (s)', y_axis_label = 'signal',
                **bsnb.opensignals_kwargs("figure")
            )
        )
        list_figures[-1].line(time_x,ppg_signal,color = "darkslategray", line_width = 2)
        cnt = 0
        time_x = []
        color_box = ["red","green","white"]
        for i in idx: 
            start_time= i[0]/sample_rate
            end_time = i[1]/sample_rate
            try:
                _proba = int(proba[cnt])
            except:
                _proba = 2
            box_annotation = BoxAnnotation(left = start_time, right = end_time, fill_color = color_box[_proba], fill_alpha = 0.2)           
            list_figures[-1].add_layout(box_annotation)
            cnt = cnt + 1
            time_x.append(((i[0] + i[1])/2)/sample_rate)
        list_figures[-1].line(time_x,pearson_coeff, 
                        color = "blue",
                        line_dash=[4, 4], line_width = 2)
        bsnb.opensignals_style(list_figures, toolbar = "above")
        show(list_figures[-1])
    else:
        print("Method only supports `savvyppg` output for now")
        

def ppg_plot_beats(ppg_signal,beats_df, sample_rate, savvyppg = True):
    """Visualize the detected beats
    
    Parameters
    ----------
    ppg_signal : list
        ppg signal
    beats_df : DataFrame
        summarized beats information after using `savvyppg.ppgSignal()` to process the ppg signal
    sample_rate: int
        sampling frequency of the signal
    savvyppg: bool
        defaults to True, function only accepts results from `savvyppg.ppgSignal()` 
        
    Returns
    -------
    Plot
        An interactive bokeh plot of the input ppg signal and the detected beats
    
    """
 
    if savvyppg:
        idx, _, _= _process_beats_df(beats_df)
        time_x = bsnb.generate_time(ppg_signal, sample_rate)
   
        list_figures = []
        list_figures.append(
            figure(x_axis_label = 'Time (s)', y_axis_label = 'signal',
                **bsnb.opensignals_kwargs("figure")
            )
        )
        list_figures[-1].line(time_x,ppg_signal,color = "darkslategray", line_width = 2)
        cnt = 0
        for i in idx: 
            start_time= i[0]/sample_rate
            end_time = i[1]/sample_rate
            
            box_annotation = BoxAnnotation(left = start_time, right = end_time, 
                                           fill_color = bsnb.opensignals_color_pallet(), fill_alpha = 0.2)
            list_figures[-1].add_layout(box_annotation)
            cnt = cnt + 1
            
        peaks_time,peaks_vals,_ = _ppg_max_beat(ppg_signal,sample_rate,beats_df)
        list_figures[-1].circle(peaks_time, peaks_vals, radius = 0.3,fill_color = "green")
        bsnb.opensignals_style(list_figures, toolbar = "above")
        show(list_figures[-1])
        
    else:
        print("Method only supports `savvyppg` output for now")       
 
    
def ppg_plot_bpm(ppg_df,sample_rate):
    """Plot the ppg signal and the heart rate
    This is called from ppg_features.ppg_heart_rate(..., show = True)
    
    Parameters
    ----------
    ppg_df : DataFrame
        processed ppg signal from `ppg_features.nk_process_ppg` 
    sample_rate: int
        sampling frequency of the signal
        
    Returns
    -------
    Plot
        An interactive bokeh plot of the input ppg signal and the bpm values
    
    """
    
    
    time_x = bsnb.generate_time(ppg_df["PPG_Peaks"].to_list(), sample_rate)
    
    
    fig_list = []
    list_figures_ppg = []
    list_figures_bpm = []
    
    list_figures_ppg.append(
        figure(
            title="PPG Signal and Peaks", 
            x_axis_label = 'Time(s)', 
            **bsnb.opensignals_kwargs("figure")
        )
    )
    list_figures_bpm.append(
        figure(
            title="Heart Rate",
            x_axis_label = 'Time(s)', 
            y_axis_label = 'BPM',
            **bsnb.opensignals_kwargs("figure")
        )
    )
    
    peaks_time = ppg_df.index[ppg_df["PPG_Peaks"] == 1].tolist()
    peaks_time = [i/sample_rate for i in peaks_time]
    peaks_vals = ppg_df[ppg_df["PPG_Peaks"] == 1]
    peaks_vals = peaks_vals["PPG_Raw"].tolist()
    mean_bpm = np.nanmean(ppg_df["PPG_Rate"])
    
    list_figures_ppg[-1].line(time_x,ppg_df["PPG_Raw"].to_list(),**bsnb.opensignals_kwargs("line"))
    list_figures_ppg[-1].circle(peaks_time, peaks_vals, radius = 0.3,fill_color = "green")
    
    
    
    list_figures_bpm[-1].line(time_x,ppg_df["PPG_Rate"],**bsnb.opensignals_kwargs("line"))
    list_figures_bpm[-1].line(time_x,[mean_bpm for i in range(len(time_x))],**bsnb.opensignals_kwargs("line"))
 


    bsnb.opensignals_style(list_figures_ppg, toolbar = "above")
    bsnb.opensignals_style(list_figures_bpm, toolbar = "above")


    grid = gridplot([list_figures_ppg,list_figures_bpm],toolbar_location='above', **bsnb.opensignals_kwargs("gridplot"))
    show(grid)
    
    

    
    
