import warnings
import numpy as np
from itertools import cycle
# Base packages used in OpenSignals Tools Notebooks for plotting data
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
from bokeh.models import BoxAnnotation, LinearAxis, Range1d
output_notebook(hide_banner=True)
from matplotlib import pyplot as plt
from scipy import signal
from statsmodels.graphics import tsaplots

# Plotly
import plotly
import plotly.graph_objects as go
import plotly.tools as tls
from plotly.subplots import make_subplots

# Package for ensuring operations in arrays
#from numpy import average, linspace, random
import biosignalsnotebooks as bsnb
from biosignalsnotebooks.visualise import plot, opensignals_kwargs, opensignals_style, opensignals_color_pallet


def plot_signal(sig, sample_rate, time_arr = None, labels = None, grid_plot = False, plotly = False, **kwargs):
    """Create a bokeh or plotly plot of signals

    Parameters
    ----------
    sig : array
        Input signal/s passed as a list
    sample_rate : list
        Sampling frequency of each input signal passed as a list
    time_arr : list
        User defined x-axis for each signal
    labels : list
        Legend for each signal plotted
    plotly: bool
        If True, it will return a plotly format line graph
    grid_plot : bool
        If False (default), signals will be plotted on the same graph. If True, each signal will be plotted 
        separately (need to pass kwargs)
    kwargs:
        x_axis_label(string) : label of the x-axis
        y_axis_label(string) : label of the y-axis
        title(string): title of the plot
        grid_lines(int) : number of row plots  (grid_plot should be True)
        grid_columns(int): number of column plots (grid_plot should be True)

        
    Returns
    -------
    Plot 
        An interactive plot of the signal.
        
    """

    time_x = [] #Generate x axis values
    if len(sample_rate) == 1 and len(sig) != 1:
        sample_rate = [sample_rate[0] for i in range(len(sig))]        
    for i in range(len(sig)):
        try:
            time_x.append(time_arr[i])
        except:
            time_x.append(bsnb.generate_time(sig[i], sample_rate[i]))
    if labels is not None:
        #check if same with _signal length
        if len(labels) != len(sig): 
            raise ValueError("Argument `labels` doesn't have equal length with the number of input signals")
    
    if not plotly: 
    
        if not grid_plot:

            _fig = bsnb.plot(time_x, sig, 
                legend_label=labels, 
                y_axis_label=kwargs.get('y_axis_label'), 
                x_axis_label=kwargs.get('x_axis_label'),
                title = kwargs.get('title')
            )
            
        else:
            _fig = bsnb.plot(time_x, sig, 
                legend_label=labels, 
                y_axis_label=kwargs.get('y_axis_label'), 
                x_axis_label=kwargs.get('x_axis_label'),
                grid_plot= True, 
                grid_lines= kwargs.get('grid_lines'), 
                grid_columns=kwargs.get('grid_columns'),
                title = [kwargs.get('title')] + ["" for i in range(len(sig)-1)]
            )
    else:
        
        if not grid_plot:
            _fig = go.Figure()
            for i in range(len(sig)):
                try: _label = labels[i]
                except: _label = None
                _fig.add_trace(go.Scatter(x=time_x[i], y=sig[i], mode='lines',name = _label))
                  
            _fig.update_layout(
                xaxis_title= kwargs.get('x_axis_label'),
                yaxis_title= kwargs.get('y_axis_label'),
                title = kwargs.get('title'),
                legend_title= "signals" )
        
        else:
            _fig = make_subplots(
                rows= kwargs.get('grid_lines'), 
                cols= kwargs.get('grid_columns'),
                x_title= kwargs.get('x_axis_label'),
                y_title= kwargs.get('y_axis_label')
            )
            plt_n = 0
            for _i in range(kwargs.get('grid_lines')):   
                for _j in range(kwargs.get('grid_columns')):
                    try: _label = labels[plt_n]
                    except: _label = None
                    _fig.add_trace(go.Scatter(x=time_x[plt_n], y=sig[plt_n],mode='lines',name = _label), row = _i+1, col = _j+1)
                    plt_n = plt_n + 1          
                
            _fig.update_layout(
                title = kwargs.get('title'),
                legend_title= "signals")
                    
    return _fig
                    
        
        

def plot_filtered(signal_raw,signal_clean, sample_rate, labels = ["Raw","Filtered"], downsample = False, plotly = False, **kwargs):
    """Create a plot for the filtered signals.
    (this can be passed in the signal_filter.filter_ppg(..., plot = True) or signal_filter.filter_ecg(..., plot = True)

    Parameters
    ----------
    signal_raw : list
        Unfiltered signal
    signal_clean : list
        Filtered signal
    sample_rate : float
        Sampling frequency of thesignal
    downsample : bool
        If True, perform downsampling using signal_transform.signal_downsampling (refer to function 
        for the set of kwargs)
    plotly: bool
        If True, it will return a plotly format line graph
        
    Returns
    -------
    Plot
        An interactive plot comparing the raw and filtered signals.
    """
    '''
    Temporarily disable, preprocessing may be done separately before passing the input signals to this function
    if downsample:
        if kwargs.get('downsample_rate'):
            signal_raw = signal_downsample(signal_raw,sample_rate, downsample_rate = kwargs.get('downsample_rate'))
            signal_clean = signal_downsample(signal_clean,sample_rate, dowsample_rate = kwargs.get('downsample_rate'))
        if kwargs.get('downsample_factor'):
            signal_raw = signal_downsample(signal_raw,sample_rate, downsample_rate = kwargs.get('downsample_factor'))
            signal_clean = signal_downsample(signal_clean,sample_rate, downsample_rate = kwargs.get('downsample_factor'))
        if kwargs.get('downsample_points'):
            signal_raw = signal_downsample(signal_raw,sample_rate, downsample_rate = kwargs.get('downsample_points'))
            signal_clean = signal_downsample(signal_clean,sample_rate, downsample_rate = kwargs.get('downsample_points'))
    '''
    if len(signal_raw) == len(signal_clean):       
        fig = plot_signal([signal_raw,signal_clean],[sample_rate],labels = labels, x_axis_label = 'Time (seconds)', plotly = plotly)
    else:
        warnings.warn(f"Raw signal and filtered signal have different length, plotting separately")
        fig = plot_signal([signal_raw,signal_clean],[sample_rate],labels = labels, x_axis_label = 'Time (seconds)', grid_plot = True, 
                    grid_lines = 2, grid_columns = 1, plotly = plotly)
    
    return fig
    
    
def plot_segments(sig, seg_sig, sample_rate, overlap_time = None,  **kwargs):
    """Plot the segmented signal
    (this can be passed in the signal_transform.segment_signal(..., plot = True)

    Parameters
    ----------
    sig : list
        Unsegmented signal
    seg_sig : array
        List of segmented signal (generated from signal_transform.segment_signal()) 
    overlap_time : float
        Sliding window time 
    
    Returns
    -------
    Plot
        An interactive bokeh plot of the raw signal with box annotation of the segments
    """
    time_x = bsnb.generate_time(sig, sample_rate)
    list_figures = []
    list_figures.append(
        figure(x_axis_label = 'Time (s)', y_axis_label = 'Time (s)', 
               **bsnb.opensignals_kwargs("figure")
        )
    )
    list_figures[-1].line(time_x, sig,**bsnb.opensignals_kwargs("line"))
    start_time = 0
    cnt = 0
    color_box = ["red","yellow","blue"]
    pool = cycle(color_box)
    for seg in seg_sig:
        time_x =  bsnb.generate_time(seg, sample_rate)  
        end_time = start_time + time_x[-1]
        box_annotation = BoxAnnotation(left = start_time, right = end_time, fill_color = next(pool), fill_alpha = 0.1)
        list_figures[-1].add_layout(box_annotation)
        if overlap_time is None:
            start_time = end_time
        else: 
            start_time = start_time + overlap_time
    
    bsnb.opensignals_style(list_figures, toolbar = "above")
    show(list_figures[-1])

def plot_epoch(sig, epoch, sample_rate):
    """Plot the segmented signal based on epochs
    (this can be passed in the signal_transform.epoch_signal(..., plot = True)

    Parameters
    ----------
    sig : list
        Unsegmented signal
    sample_rate: integer
        Sampling frequency
    epoch : dictionary
        Dictionary of epoch and the segmented signal (generated from signal_transform.epoch_signal()) 
    
    Returns
    -------
    Plot
        An interactive bokeh plot of the raw signal with box annotation of the segments
    """
    time_x = bsnb.generate_time(sig, sample_rate)
    list_figures = []
    list_figures.append(
        figure(x_axis_label = 'Time (s)', y_axis_label = 'Time (s)', 
               **bsnb.opensignals_kwargs("figure")
        )
    )
    list_figures[-1].line(time_x, sig,legend_label = str([i.split('_')[1] for i in epoch.keys()]),**bsnb.opensignals_kwargs("line"))
    start_time = 0
    cnt = 0
    color_box = ["red","blue","green","purple","brown","orange","gray","yellow","black"]
    _unique_vals = sorted(list(set([i.split('_')[1] for i in epoch.keys()])))
    #pool = cycle(color_box)
    for ep in epoch.keys():
        
        time_x =  bsnb.generate_time(epoch.get(ep), sample_rate)  
        _ep = ep.split("_")[1]
        end_time = start_time + time_x[-1]
        box_annotation = BoxAnnotation(left = start_time, right = end_time, fill_color = color_box[_unique_vals.index(_ep)], fill_alpha = 0.1)
        #box_annotation = BoxAnnotation(left = start_time, right = end_time, tags = [ep], fill_color = bsnb.opensignals_color_pallet(), fill_alpha = 0.2)
        list_figures[-1].add_layout(box_annotation)
        start_time = end_time

    
    bsnb.opensignals_style(list_figures, toolbar = "above")
    show(list_figures[-1])
    
def plot_epoch_hr(hr_sig, epoch, sample_rate):
    """Plot the instantaneous hr of the segmented signal based on epochs
   

    Parameters
    ----------
    hr_sig : list
        instantaneous heart rate (should be the same length with the total number of signal points
        in the `epoch` dictionary)
    epoch : dictionary
        Dictionary of epoch and the segmented signal (generated from signal_transform.epoch_signal()) 
    sample_rate: integer
        Sampling frequency
        
    Returns
    -------
    Plot
        An interactive bokeh plot of the raw signal with box annotation of the segments
    """
    time_x = bsnb.generate_time(hr_sig, sample_rate)
    list_figures = []
    list_figures.append(
        figure(x_axis_label = 'Time (s)', y_axis_label = 'Time (s)', 
               **bsnb.opensignals_kwargs("figure")
        )
    )
    list_figures[-1].line(time_x, hr_sig,legend_label = str([i.split('_')[1] for i in epoch.keys()]),**bsnb.opensignals_kwargs("line"))
    start_time = 0
    cnt = 0
    color_box = ["red","blue","green","purple","brown","orange","gray","yellow","black"]
    _unique_vals = sorted(list(set([i.split('_')[1] for i in epoch.keys()])))
    #pool = cycle(color_box)
    for ep in epoch.keys():
        
        time_x =  bsnb.generate_time(epoch.get(ep), sample_rate)  
        _ep = ep.split("_")[1]
        end_time = start_time + time_x[-1]
        box_annotation = BoxAnnotation(left = start_time, right = end_time, fill_color = color_box[_unique_vals.index(_ep)], fill_alpha = 0.1)
        #box_annotation = BoxAnnotation(left = start_time, right = end_time, tags = [ep], fill_color = bsnb.opensignals_color_pallet(), fill_alpha = 0.2)
        list_figures[-1].add_layout(box_annotation)
        start_time = end_time

    
    bsnb.opensignals_style(list_figures, toolbar = "above")
    show(list_figures[-1])

    
def fft_graph_compare(sig_raw, sig_clean, fs, f1, f2): # bokeh plotting
    """Supplementary function that plots the Fast Fourier Transform (FFT) of the raw and clean signals.
    Additionally, the corner frequencies of the filter applied to the clean signal is shown.
    
    Parameters
    ----------
    sig_raw : array-like
        Raw signal
    sig_clean : string
        Clean signal
    fs : int
        Sampling rate of the signals
    f1 : float
        High-pass filter cut-off
    f2 : float
        Low-pass filter cut-off
    
    Returns
    -------
    None
    """
    fft_raw = np.abs(np.fft.fft(sig_raw - np.mean(sig_raw)))
    fft_clean = np.abs(np.fft.fft(sig_clean - np.mean(sig_clean)))

    freq_arr = np.fft.fftshift(np.fft.fftfreq(len(sig_raw),d=1/fs))
    idx = np.where(freq_arr == 0)[0][0]
    freq_arr = freq_arr[idx:]
    fft_raw = np.fft.fftshift(fft_raw)
    fft_clean = np.fft.fftshift(fft_clean)
    fft_raw = fft_raw[idx:]
    fft_clean = fft_clean[idx:]

    fig_list = []
    list_figures_raw = []
    list_figures_clean = []
    list_figures_raw.append(
        figure(
            x_range=(0, f2*1.1),
            title="Raw Signal", 
            x_axis_label = 'Frequency (Hz)', 
            **bsnb.opensignals_kwargs("figure")
        )
    )
    list_figures_clean.append(
        figure(x_range=(0, f2*1.1),
        title="Clean Signal",
        x_axis_label = 'Frequency (Hz)', 
        **bsnb.opensignals_kwargs("figure")
        )
    )
    list_figures_raw[-1].line(freq_arr,fft_raw,**bsnb.opensignals_kwargs("line"))
    list_figures_clean[-1].line(freq_arr,fft_clean,**bsnb.opensignals_kwargs("line"))
    box_annotation = BoxAnnotation(left = f1, right = f2, fill_color = 'red', fill_alpha = 0.1)
    list_figures_raw[-1].add_layout(box_annotation)
    list_figures_clean[-1].add_layout(box_annotation)  

    bsnb.opensignals_style(list_figures_raw, toolbar = "above")
    bsnb.opensignals_style(list_figures_clean, toolbar = "above")

    grid = gridplot([list_figures_raw,list_figures_clean],toolbar_location='above', **bsnb.opensignals_kwargs("gridplot"))
    show(grid)
    
    
def fft_graph(sig, fs): # bokeh plotting
    """
    Function that plots the Fast Fourier Transform (FFT) of a given signal.
    
    Parameters
    ----------
    sig :array-like
        Signal
    fs : int
        Sampling rate of the signals

    Returns
    -------
    None
    """
    fft_sig = np.abs(np.fft.fft(sig - np.mean(sig)))
    freq_arr = np.fft.fftshift(np.fft.fftfreq(len(sig),d=1/fs))
    idx = np.where(freq_arr == 0)[0][0]
    freq_arr = freq_arr[idx:]
    fft_sig = np.fft.fftshift(fft_sig)
    fft_sig = fft_sig[idx:]

    fig_list = []
    list_figures= []
    list_figures.append(
        figure(
            x_range=(0, f2*1.1),
            title="Signal FFT", 
            x_axis_label = 'Frequency (Hz)', 
            **bsnb.opensignals_kwargs("figure")
        )
    )

    list_figures[-1].line(freq_arr,fft_sig,**bsnb.opensignals_kwargs("line"))
    bsnb.opensignals_style(list_figures, toolbar = "above")
    grid = gridplot([list_figures],toolbar_location='above', **bsnb.opensignals_kwargs("gridplot"))
    show(grid)

def plot_crosscorr(sig1, sig2, fs, normalize=True, return_crosscorr=False, plotly=False, title=True):
    """A function based on numpy.correlate which plots the cross correlation of the 2 given signals.
    
    ** implementation of functions for signals of different lengths still needs to be checked
    
    Parameters
    ----------
    sig1 : array-like
        First of 2 signals to be cross-correlated
    sig2 : array-like
        Second of 2 signals to be cross-correlated
    fs : float
        Sampling frequency
    normalize: bool
        If true, correlation values normalized to [-1, 1]
    return_crosscorr : bool
        If set to True, the values associated to the calculated cross-correlation is returned
    plotly : bool
        If true, return Plotly figure instead of Matplotlib
    tile : bool
        If true, return figure with title (default: "Cross-correlation")
    
    Returns
    -------
    crosscorr : 1-D ndarray
        Cross-correlation values (with various lags) of the 2 signals given
        Only returned if return_crosscorr=True.
        
    fig : plot figure object
        A figure object associated with the cross-correlation plot.
        If plotly = True, then it returns a Plotly figure while a Matplotlib figure otherwise.
    """

    # normalize
    if normalize:
        sig1 = (sig1 - np.mean(sig1)) / (np.std(sig1) * len(sig1))
        sig2 = (sig2 - np.mean(sig2)) /  np.std(sig2)

    # calculate cross correlation
    crosscorr = np.correlate(sig1, sig2, mode="full")

    # plotting
    plt.figure(figsize=(10,5))
    t = np.linspace(1, len(crosscorr), len(crosscorr))
    plt.plot(t, crosscorr)
    plt.xlabel("Time Lag (seconds)")
    plt.ylabel("Correlation Coefficient")
    
    if title:
        plt.title("Cross Correlation")

    # xticks
    zero_lag_i = sig1.shape[0] - 1
    xticks = np.linspace(0, len(crosscorr)-1, len(crosscorr))
    xticks_translated = [xtick-zero_lag_i for xtick in xticks] # to make 0 center
    pos_xticks = xticks_translated[zero_lag_i::int(np.floor(len(xticks_translated)/10))]
    neg_xticks = xticks_translated[zero_lag_i::-int(np.floor(len(xticks_translated)/10))]
    xticks_10_translated = neg_xticks[1::][::-1]+pos_xticks
    xticks_10 = [xtick+zero_lag_i for xtick in xticks_10_translated] # translated back

    lags = [lag for lag in range(-int(len(sig1)/fs-1), int(len(sig2)/fs-1)-1)]
    zero_i = lags.index(0)
    pos_lags = lags[zero_i::int(np.floor(len(lags)/10))]
    neg_lags = lags[zero_i::-int(np.floor(len(lags)/10))]
    pos_lags = pos_lags[:len(pos_xticks)] # for same length for mapping
    neg_lags = neg_lags[:len(neg_xticks)] # for same length for mapping
    lags_seconds = neg_lags[1::][::-1]+pos_lags

    plt.xticks(xticks_10, lags_seconds)
    
    if plotly == True:
        # pull x and y data from mpl fig
        x = t
        y = crosscorr

        # layout
        title = True # TEMP
        if title:
            layout = go.Layout(
                title = 'Cross-correlation',
                yaxis = dict(title = 'Correlation Coefficient'), # y-axis label
                xaxis = dict(title = "Time Lag (seconds)"), # x-axis label
            )
        else:
            layout = go.Layout(
                yaxis = dict(title = 'Correlation Coefficient'), # y-axis label
                xaxis = dict(title = "Time Lag (seconds)"), # x-axis label
            )

        # plot figure
        fig = go.Figure(data=go.Scatter(x=x, y=y,
                                        mode='lines+markers',
                                        name='lines+markers'),
                        layout=layout)

        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = xticks_10,
                ticktext = lags_seconds
            )
        )
        
        fig.update_layout(
            autosize=False,
            width=800,
            height=600,)
    else:
        fig = plt.gcf() # get figure
        
    # optional return
    if return_crosscorr:
        return crosscorr, fig
    else:
        return fig

def plot_spectrogram(sig, fs, ylim=None, return_spectrogram=False, plotly = False, title=True):
    """A function based on scipy.signal.spectrogram which plots a given signal's spectrogram.
    
    Parameters
    ----------
    sig : array-like
        Signal
    fs : float
        Sampling frequency
    ylim : list (len=2)
        List with the first element being the lower y-axis limit of the spectrogram and the second element
        being the higher y-axis limit. (CURRENTLY ONLY WORKS FOR plotly=False).
    return_spectrogram : bool
        If set to True, the values associated to the calculated spectrogram is returned
    plotly : bool
        If true, return Plotly figure instead of Matplotlib
    title : bool
        If true, return figure with title (default: "Spectrogram")
    
    Returns
    -------
    f : 1-D ndarray
        Array of sample frequencies.
        Only returned if return_spectrogram=True.
        
    t : 1-D ndarray
        Array of segment times.
        Only returned if return_spectrogram=True.
        
    Sxx : 2-D ndarray
        Spectrogram of x. By default, the last axis of Sxx corresponds to the segment times.
        Only returned if return_spectrogram=True.
        
    fig : plot figure object
        A figure object associated with the signal spectrogram plot.
        If plotly = True, then it returns a Plotly figure while a Matplotlib figure otherwise.
    """
    
    N = len(sig)
    length_sec = N/fs

    # setting xticks for time
    if length_sec <= 10:
        time = np.arange(0, length_sec)
    elif length_sec <= 20:
        time = np.arange(0, length_sec, 5)
    elif length_sec <= 200:
        time = np.arange(0, length_sec, 10)
    else:
        time = np.arange(0, length_sec, 20)

    # calculate spectrogram
    f, t, Sxx = signal.spectrogram(sig, fs)
    
    # plotting spectrogram (matplotlib)
    if plotly == False:
        plt.figure(figsize=(10,5))
        plt.pcolormesh(t, f, Sxx, shading="gouraud")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.xticks(time)
        
        if title:
            plt.title("Spectrogram")

        if ylim != None:
            plt.ylim(ylim[0], ylim[1])
            
        fig = plt.gcf()
            
    # plotting spectrogram (plotly)
    else:
        # plotly spectrum: https://stackoverflow.com/questions/38701969/dynamic-spectrum-using-plotly
        trace = [go.Heatmap(
            x= t,
            y= f,
            z= 10*np.log10(Sxx),
#             colorscale='Jet',
            )]
        
        if title:
            layout = go.Layout(
                title = 'Spectrogram',
                yaxis = dict(title = 'Frequency (Hz)'), # x-axis label
                xaxis = dict(title = 'Time (seconds)'), # y-axis label
            )
        else:
            layout = go.Layout(
                yaxis = dict(title = 'Frequency (Hz)'), # x-axis label
                xaxis = dict(title = 'Time (seconds)'), # y-axis label
            )
        fig = go.Figure(data=trace[0], layout=layout)
        
        fig.update_layout(
            autosize=False,
            width=800,
            height=600,)
    
    if return_spectrogram:
        return f, t, Sxx, fig
    else:
        return fig

def plot_autocorrelation(sig, fs, max_lag=5, plotly=False, title=True):
    """A function based on statsmodels.graphics.tsaplots which plots a given signal's autocorrelation
    with different lag values.
    
    Parameters
    ----------
    sig : array-like
        Signal
    fs : float
        Sampling frequency
    max_lag : int
        Maximum lag (in seconds) to calculate a signal's autocorrelation on.
    plotly : bool
        If true, return Plotly figure instead of Matplotlib
    title : bool
        If true, return figure with title (default: "Autocorrelation")

    Returns
    -------
    fig : plot figure object
        A figure object associated with the signal autocorrelation plot.
        If plotly = True, then it returns a Plotly figure while a Matplotlib figure otherwise.
    """
    
    time = np.arange(0, max_lag+1)
    fig = tsaplots.plot_acf(sig, lags = [x for x in range(0,max_lag*fs)], title=None) # CORRECT THIS ERROR ON RNDSIGNAL
    ax = fig.gca()
    
    # plot modifications
    ax.set_xlabel("Time Lag (seconds)")
    ax.set_ylabel("Correlation Coefficient")
    ax.set_xticks(np.arange(0, fs*(max_lag+1), fs))
    ax.set_xticklabels(time)

    if title:
        ax.set_title("Autocorrelation")
    
    if plotly == True:
        # pull x and y data from mpl fig
        x = ax.lines[1].get_xdata()
        y = ax.lines[1].get_ydata()

        # layout
        if title:
            layout = go.Layout(
                title = 'Autocorrelation',
                yaxis = dict(title = 'Correlation Coefficient'), # y-axis label
                xaxis = dict(title = "Time Lag (seconds)"), # x-axis label
            )
        else:
            layout = go.Layout(
                yaxis = dict(title = 'Correlation Coefficient'), # y-axis label
                xaxis = dict(title = "Time Lag (seconds)"), # x-axis label
            )

        # plot figure
        fig = go.Figure(data=go.Scatter(x=x, y=y,
                                        mode='lines+markers',
                                        name='lines+markers'),
                        layout=layout)

        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = np.arange(0, fs*(max_lag+1), fs),
                ticktext = time
            )
        )
        
        fig.update_layout(
            autosize=False,
            width=800,
            height=600,)

        return fig
        
    else:
        return fig
