# Import libraries
from bokeh.plotting import figure, show
from bokeh.models import BoxAnnotation, LinearAxis, Range1d 
from bokeh.io import output_notebook
from bokeh.layouts import gridplot
output_notebook(hide_banner=True)
import biosignalsnotebooks as bsnb
import neurokit2 as nk
import numpy as np
import scipy
from rndSignal.signal_quality.snr import snr


def ecg_plot(sig, sr, plot_peaks = False, plot_tachogram = False, plot_ectopic_beats = False, 
             plot_time = False, plot_poincare = False, plot_frequency = False):
    """This function displays any plot specified by the user. These plots are generated from 
    the heart rate variability (HRV) analysis using ECG signals.
    
    Parameters
    ----------
    sig : array
        The filtered ECG signal data.
    sr : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    plot_tachogram: bool
        If True, then the function will display a 2x1 gridplot presenting the tachogram
        (top cell) and the respective ECG signal (bottom cell), highlighting each cardiac
        cycle.
    plot_ectopic_beats: bool
        If True, then the function will display a plot containing the tachogram before
        and after removing the influence of ectopic heart beats.
    plot_time: bool
        If True, then the function will display a plot containing various HRV parameters
        extracted from the tachogram.
    plot_poincare: bool
        If True, then the function will display the Poincaré plot together with the axis
        needed to estimate the SD1 and SD2 parameters.
    plot_frequency: bool
        If True, then the function will display a plot containing the tachogram power
        spectrum along with meaningful frequency bands.
        
    Returns
    -------
    Any interactive bokeh plot(s) specified by the user.
    """   
    
    ecg_snr = snr(sig, sr, value = True)
    if(ecg_snr < 0):
        raise Exception("The signal is not of good quality.")
    
    time = bsnb.generate_time(sig, sr)
    # Detecting R Peaks
    instant_peaks, time_r_peaks = nk.ecg_peaks(sig, sr, correct_artifacts = True)
    time_r_peaks = time_r_peaks['ECG_R_Peaks']
    time_r_peaks = time_r_peaks / sr 

    tachogram_data = np.diff(time_r_peaks)
    tachogram_time = (time_r_peaks[1:] + time_r_peaks[:-1]) / 2
    
    if(plot_peaks == True): # uses modified function
        _plot_r_peaks(sig, sr, time_units = True, plot_result = True)

    if(plot_tachogram == True):
        if(len(sig) / sr > 60): # 1 minute threshold
            print("Warning: Longer durations will lead to more condensed plots.")
            list_figures = bsnb.plot_ecg_tachogram(time, sig, tachogram_time, 
                                                   tachogram_data, time_r_peaks)
        else:
            list_figures = bsnb.plot_ecg_tachogram(time, sig, tachogram_time, 
                                                   tachogram_data, time_r_peaks)     
    
    # An ectopic beat refers to a cardiac cycle that differs in at least 20% of the duration of 
    # the previous one and needs to be removed
    tachogram_data_NN, tachogram_time_NN = bsnb.remove_ectopy(tachogram_data, tachogram_time)
    
    if(plot_ectopic_beats == True):
        # Comparison between the tachograms obtained before and after ectopic beat removal
        bsnb.plot_post_ecto_rem_tachogram(tachogram_time, tachogram_data, tachogram_time_NN, 
                                          tachogram_data_NN)
    
    # Time Parameters                    
    # Maximum, Minimum and Average RR Interval
    max_rr = np.max(tachogram_data_NN)
    min_rr = np.min(tachogram_data_NN)
    avg_rr = np.average(tachogram_data_NN)

    # Maximum, Minimum and Average Heart Rate
    max_hr = 1 / min_rr # Cycles per second
    max_bpm = max_hr * 60 # BPM

    min_hr = 1 / max_rr # Cycles per second
    min_bpm = min_hr * 60 # BPM

    avg_hr = 1 / avg_rr # Cyles per second
    avg_bpm = avg_hr * 60 # BPM

    # SDNN
    sdnn = np.std(tachogram_data_NN)

    time_param_dict = {"Maximum RR": max_rr, "Minimum RR": min_rr, "Average RR": avg_rr, 
                       "Maximum BPM": max_bpm, "Minimum BPM": min_bpm, "Average BPM": avg_bpm, 
                       "SDNN": sdnn}
    
    if(plot_time == True): # uses modified function 
        _plot_hrv_parameters(tachogram_time_NN, tachogram_data_NN, time_param_dict)
            
    # Poincaré Parameters
    if(plot_poincare == True):
        bsnb.plot_poincare(tachogram_data)

    # Frequency Parameters
    if(plot_frequency == True):
        if(len(sig) / sr < 30): # minimum duration to compute frequency parameters
            raise Exception("ECG signal duration must be at least 30 seconds.")
        else:
            # Auxiliary Structures
            freqs, power_spect = bsnb.psd(tachogram_time, tachogram_data) # Power spectrum
            bsnb.plot_hrv_power_bands(freqs, power_spect)

def _plot_r_peaks(sig, sr, time_units = False, plot_result = False):
    """This function uses Neurokit's algorithm for detecting R peaks in ECG signal and displays
    them using Biosignals PLUX's custom bokeh plots.
    
    Parameters
    ----------
    sig : array
        The filtered ECG signal data.
    sr : int
        Sampling frequency.
    time_units : boolean
        If True this function will return the R peak position in seconds.
    plot_result : boolean
        If True it will be presented a graphical representation of the R peak position in the ECG
        signal.
    """
    
    if time_units is True:
        time = np.linspace(0, len(sig) / sr, len(sig))
    else:
        time = np.linspace(0, len(sig) - 1, len(sig))

    # Detecting R Peaks
    instant_peaks, rpeaks = nk.ecg_peaks(sig, sr, correct_artifacts = True)
    r_peaks = rpeaks['ECG_R_Peaks']

    if time_units is True:
        peaks = np.array(time)[r_peaks]
    else:
        peaks = r_peaks

    # If plot is invoked by plot_result flag, then a graphical representation of the R peaks is
    # presented to the user
    if plot_result is True:
        list_figures = []
        
        list_figures.append(figure(x_axis_label = 'Time (s)', y_axis_label = 'Raw Data', 
                                   **bsnb.opensignals_kwargs("figure")))
        list_figures[-1].line(time, sig, **bsnb.opensignals_kwargs("line"))
        list_figures[-1].circle(time[r_peaks], np.array(sig)[r_peaks], size = 10, 
                                color = bsnb.opensignals_color_pallet(), legend_label = "R Peaks")

        # Apply OpenSignals style to the plot
        bsnb.opensignals_style(list_figures)
        
        # Show plot
        show(list_figures[-1])


def _plot_hrv_parameters(tachogram_time, tachogram_data, time_param_dict):
    """This function ensures the generation of an intuitive graphical representation of Heart Rate 
    Variability (HRV) parameters, extracted from the tachogram.
    
    Parameters
    ----------
    tachogram_time : list
        Time axis of the original tachogram (before ectopic beats removal).
    tachogram_data : list
        RR interval duration values linked to each entry of tachogram_time.
    time_param_dict : dict
        Dictionary containing the numerical values of the parameters that will be graphically 
        translated by this function.
        The following keys must be available:
        >> "Maximum RR"
        >> "Minimum RR"
        >> "Average RR"
        >> "Maximum BPM"
        >> "Minimum BPM"
        >> "Average BPM"
        >> "SDNN"
    """

    # Check if all keys are available
    temp_dict_keys = list(time_param_dict.keys())
    for dict_key in ["Maximum RR", "Minimum RR", "Average RR", "Maximum BPM", "Minimum BPM", 
                     "Average BPM", "SDNN"]:
        if dict_key not in temp_dict_keys:
            raise RuntimeError(dict_key + " key is not available in time_param_dict input")

    # List that store the figure handler
    list_figures = []

    # Conversion of RR interval duration to BPM
    bpm_data = (1 / np.array(tachogram_data)) * 60

    # Plotting of Tachogram
    list_figures.append(
        figure(x_axis_label = 'Time (s)', y_axis_label = 'Cardiac Cycle (s)', 
               x_range = (0, tachogram_time[-1] + 0.30 * tachogram_time[-1]),
               y_range = (0.6, 1), **bsnb.opensignals_kwargs("figure")))
    list_figures[-1].line(tachogram_time, tachogram_data, legend_label = "Original Tachogram", 
                          **bsnb.opensignals_kwargs("line"))

    # Setting the second y axis range name and range of values
    list_figures[-1].extra_y_ranges = {"BPM": Range1d(start = 60, end = 95)}

    # Addition of the second axis to the plot
    list_figures[-1].add_layout(LinearAxis(y_range_name = "BPM", axis_label = 'BPM'), 'right')

    list_figures[-1].line(tachogram_time, bpm_data, legend_label = "Heart Rate (BPM)", 
                          y_range_name = "BPM", **bsnb.opensignals_kwargs("line"))

    # Representation of Maximum, Minimum and Average Points
    dict_keys = time_param_dict.keys()
    for key in dict_keys:
        if ("Maximum" in key or "Minimum" in key) and "BPM" not in key:
            find_time = tachogram_time[np.where(tachogram_data == time_param_dict[key])[0][0]]
            list_figures[-1].circle(find_time, time_param_dict[key], size = 10, 
                                    fill_color = bsnb.opensignals_color_pallet(), 
                                    legend_label = key)

        elif ("Maximum" in key or "Minimum" in key) and "BPM" in key:
            find_time = tachogram_time[np.where(bpm_data == time_param_dict[key])[0][0]]
            list_figures[-1].circle(find_time, time_param_dict[key], size = 10, 
                                    fill_color = bsnb.opensignals_color_pallet(), 
                                    legend_label = key, y_range_name = "BPM")

        elif "Average" in key and "BPM" not in key:
            list_figures[-1].line([0, tachogram_time[-1]], 
                                  [time_param_dict[key], time_param_dict[key]], 
                                  legend_label = "Average RR", **bsnb.opensignals_kwargs("line"))

        elif "SDNN" in key:
            box_annotation = BoxAnnotation(left = 0, right = tachogram_time[-1], 
                                           top = (time_param_dict["Average RR"] + 
                                                  time_param_dict["SDNN"]),
                                           bottom = (time_param_dict["Average RR"] - 
                                                     time_param_dict["SDNN"]),
                                           fill_color = "black", fill_alpha = 0.1)
            list_figures[-1].rect(find_time, time_param_dict[key], width = 0, height = 0, 
                                  fill_color = "black", fill_alpha = 0.1, legend_label = "SDNN")
            list_figures[-1].add_layout(box_annotation)

    # Apply OpenSignals style to the plot
    bsnb.opensignals_style(list_figures, toolbar = "above")

    # Show plot
    show(list_figures[-1])

    
def ecg_plot_bpm(ecg_df,sample_rate):
    """Plot the ecg signal and the heart rate
    This is called from ecg_features.ecg_heart_rate(..., show = True)
    
    Parameters
    ----------
    ecg_df : DataFrame
        processed ecg signal from `ecg_features._nk_process_ecg` 
    sample_rate: int
        sampling frequency of the signal
        
    Returns
    -------
    Plot
        An interactive bokeh plot of the input ecg signal and the bpm values
    
    """
    
    
    time_x = bsnb.generate_time(ecg_df["ECG_Rate"].to_list(), sample_rate)
    
    
    fig_list = []
    list_figures_ecg = []
    list_figures_bpm = []
    
    list_figures_ecg.append(
        figure(
            title="ECG Signal", 
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
    
    mean_bpm = np.nanmean(ecg_df["ECG_Rate"])
    
    list_figures_ecg[-1].line(time_x,ecg_df["ECG_Raw"].to_list(),**bsnb.opensignals_kwargs("line"))
    
    
    
    list_figures_bpm[-1].line(time_x,ecg_df["ECG_Rate"],**bsnb.opensignals_kwargs("line"))
    list_figures_bpm[-1].line(time_x,[mean_bpm for i in range(len(time_x))],**bsnb.opensignals_kwargs("line"))
 


    bsnb.opensignals_style(list_figures_ecg, toolbar = "above")
    bsnb.opensignals_style(list_figures_bpm, toolbar = "above")


    grid = gridplot([list_figures_ecg,list_figures_bpm],toolbar_location='above', **bsnb.opensignals_kwargs("gridplot"))
    show(grid)
    
    