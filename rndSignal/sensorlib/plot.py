# -*- coding: iso-8859-1 -*-
"""
Continuous wavelet transform plot module for Python.

DISCLAIMER
    This module is based on routines provided by C. Torrence and G.
    Compo available at http://paos.colorado.edu/research/wavelets/, on
    routines provided by Aslak Grinsted, John Moore and Svetlana
    Jevrejeva and available at
    http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and
    on routines provided by A. Brazhe available at
    http://cell.biophys.msu.ru/static/swan/.

    This software may be used, copied, or redistributed as long as it
    is not sold and this copyright notice is reproduced on each copy
    made. This routine is provided as is without any express or implied
    warranties whatsoever.

AUTHOR
    Sebastian Krieger
    email: sebastian@nublia.com

REVISION
    1 (2013-02-15 17:51 -0300)

REFERENCES
    [1] Mallat, S. (2008). A wavelet tour of signal processing: The
        sparse way. Academic Press, 2008, 805.
    [2] Addison, P. S. (2002). The illustrated wavelet transform
        handbook: introductory theory and applications in science,
        engineering, medicine and finance. IOP Publishing.
    [3] Torrence, C. and Compo, G. P. (1998). A Practical Guide to
        Wavelet Analysis. Bulletin of the American Meteorological
        Society, American Meteorological Society, 1998, 79, 61-78.
    [4] Torrence, C. and Webster, P. J. (1999). Interdecadal changes in
        the ENSO-Monsoon system, Journal of Climate, 12(8), 2679-2690.
    [5] Grinsted, A.; Moore, J. C. & Jevrejeva, S. (2004). Application
        of the cross wavelet transform and wavelet coherence to
        geophysical time series. Nonlinear Processes in Geophysics, 11,
        561-566.
    [6] Liu, Y.; Liang, X. S. and Weisberg, R. H. (2007). Rectification
        of the bias in the wavelet power spectrum. Journal of
        Atmospheric and Oceanic Technology, 24(12), 2093-2102.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider, Button
import os

from . import stepcycle as step
from . import time as sbtime

__version__ = '$Revision: 1 $'
# $Source$


def __init__(show=False):
    fontsize = 'medium'
    params = {'font.family': 'serif',
              'font.sans-serif': ['Helvetica'],
              'font.size': 18,
              'font.stretch': 'ultra-condensed',
              'text.fontsize': fontsize,
              'xtick.labelsize': fontsize,
              'ytick.labelsize': fontsize,
              'axes.titlesize': fontsize,
              'text.usetex': True,
              'text.latex.unicode': True,
              'timezone': 'UTC'
              }
    plt.rcParams.update(params)
    plt.ion()


def figure(fp=dict(), ap=dict(left=0.15, bottom=0.12, right=0.95, top=0.95,
                              wspace=0.10, hspace=0.10), orientation='landscape'):
    """Creates a standard figure.

    PARAMETERS
        fp (dictionary, optional) :
            Figure properties.
        ap (dictionary, optional) :
            Adjustment properties.

    RETURNS
        fig : Figure object

    """

    __init__()

    if 'figsize' not in list(fp.keys()):
        if orientation == 'landscape':
            fp['figsize'] = [11, 8]
        elif orientation == 'portrait':
            fp['figsize'] = [8, 11]
        elif orientation == 'squared':
            fp['figsize'] = [8, 8]
        elif orientation == 'worldmap':
            fp['figsize'] = [9, 5.0625]  # Widescreen aspect ratio 16:9
        else:
            raise Warning('Orientation \'%s\' not allowed.' % (orientation, ))

    fig = plt.figure(**fp)
    fig.subplots_adjust(**ap)

    return fig


def cwt(t, f, cwt, sig, rectify=False, **kwargs):
    """Plots the wavelet power spectrum.

    It rectifies the bias in the wavelet power spectrum as noted by
    Liu et al. (2007) dividing the power by the wavelet scales.

    PARAMETERS
        t (array like) :
            Time array.
        f (array like) :
            Function array.
        cwt (list) :
            List containing the results from wavelet.cwt function (e.g.
            wave, scales, freqs, coi, fft, fftfreqs)
        sig (list) :
            List containig the results from wavelet.significance funciton
            (e.g. signif, fft_theor)
        rectify (boolean, optional) :
            Sets wether to rectify the wavelet power by dividing by the
            wavelet scales, according to Liu et al. (2007).

    RETURNS
        A list with the figure and axis objects for the plot.

    """
    # Sets some parameters and renames some of the input variables.
    N = len(t)
    dt = np.diff(t)[0]
    wave, scales, freqs, coi, fft, fftfreqs = cwt
    signif, fft_theor = sig
    #
    # if 'std' in list(kwargs.keys()):
    #     std = kwargs['std']
    # else:
    #     std = f.std()  # calculates standard deviation ...
    # std2 = std ** 2   # ... and variance
    #
    period = 1. / freqs
    power = (abs(wave)) ** 2  # normalized wavelet power spectrum
    # fft_power = std2 * abs(fft) ** 2  # FFT power spectrum
    sig95 = power / signif[:, None]  # power is significant where ratio > 1
    if rectify:
        scales = np.ones([1, N]) * scales[:, None]
        levels = np.arange(0, 2.1, 0.1)
        labels = levels
    else:
        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']

    result = []

    if 'fig' in list(kwargs.keys()):
        fig = kwargs['fig']
    else:
        fig = figure()
    result.append(fig)

    # Plots the normalized wavelet power spectrum and significance level
    # contour lines and cone of influece hatched area.
    if 'fig' in list(kwargs.keys()):
        ax = kwargs['ax']
    else:
        ax = fig.add_subplot(1, 1, 1)
    cmin, cmax = power.min(), power.max()
    rmin, rmax = min(levels), max(levels)
    if 'extend' in list(kwargs.keys()):
        extend = kwargs['extend']
    elif (cmin < rmin) & (cmax > rmax):
        extend = 'both'
    elif (cmin < rmin) & (cmax <= rmax):
        extend = 'min'
    elif (cmin >= rmin) & (cmax > rmax):
        extend = 'max'
    elif (cmin >= rmin) & (cmax <= rmax):
        extend = 'neither'

    if rectify:
        cf = ax.contourf(t, np.log2(period), power / scales, levels,
                         extend=extend)
    else:
        cf = ax.contourf(t, np.log2(period), np.log2(power),
                         np.log2(levels), extend=extend)
    ax.contour(t, np.log2(period), sig95, [-99, 1], colors='r',
               linewidths=2.)
    ax.fill(np.concatenate([t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt, t[:1]-dt]),
            np.log2(np.concatenate([coi, [1e-9], period[-1:], period[-1:], [1e-9]])),
            'k', alpha=0.3, hatch='x')
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels(Yticks)
    # ax.set_yticklabels(labels)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim(np.log2([period.min(), min([coi.max(), period.max()])]))
    ax.invert_yaxis()
    if rectify:
        cbar = fig.colorbar(cf)
    if not rectify:
        cbar = fig.colorbar(cf, ticks=np.log2(levels))
        cbar.ax.set_yticklabels(labels)
    result.append(ax)

    return result


def xwt(*args, **kwargs):
    """Plots the cross wavelet power spectrum and phase arrows.
    function.

    The relative phase relationship convention is the same as adopted
    by Torrence and Webster (1999), where in phase signals point
    upwards (N), anti-phase signals point downwards (S). If X leads Y,
    arrows point to the right (E) and if X lags Y, arrow points to the
    left (W).

    PARAMETERS
        xwt (array like) :
            Cross wavelet transform.
        coi (array like) :
            Cone of influence, which is a vector of N points containing
            the maximum Fourier period of useful information at that
            particular time. Periods greater than those are subject to
            edge effects.
        freqs (array like) :
            Vector of Fourier equivalent frequencies (in 1 / time units)
            that correspond to the wavelet scales.
        signif (array like) :
            Significance levels as a function of Fourier equivalent
            frequencies.
        da (list, optional) :
            Pair of integers that the define frequency of arrows in
            frequency and time, default is da = [3, 3].

    RETURNS
        A list with the figure and axis objects for the plot.

    SEE ALSO
        wavelet.xwt

    """
    # Sets some parameters and renames some of the input variables.
    xwt, t, coi, freqs, signif = args[:5]
    if 'scale' in list(kwargs.keys()):
        scale = kwargs['scale']
    else:
        scale = 'log2'

    N = len(t)
    dt = t[1] - t[0]
    period = 1. / freqs
    power = abs(xwt)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power / sig95  # power is significant where ratio > 1

    # Calculates the phase between both time series. The phase arrows in the
    # cross wavelet power spectrum rotate clockwise with 'north' origin.
    if 'angle' in list(kwargs.keys()):
        angle = 0.5 * np.pi - kwargs['angle']
    else:
        angle = 0.5 * np.pi - np.angle(xwt)
    u, v = np.cos(angle), np.sin(angle)

    result = []

    # if 'da' in list(kwargs.keys()):
    #     da = kwargs['da']
    # else:
    #     da = [3, 3]
    if 'fig' in list(kwargs.keys()):
        fig = kwargs['fig']
    else:
        fig = figure()
    result.append(fig)

    if 'fig' in list(kwargs.keys()):
        ax = kwargs['ax']
    else:
        ax = fig.add_subplot(1, 1, 1)

    # Plots the cross wavelet power spectrum and significance level
    # contour lines and cone of influece hatched area.
    if 'crange' in list(kwargs.keys()):
        levels = labels = kwargs['crange']
    else:
        levels = [0.125, 0.25, 0.5, 1, 2, 4, 8]
        labels = ['1/8', '1/4', '1/2', '1', '2', '4', '8']
    cmin, cmax = power.min(), power.max()
    rmin, rmax = min(levels), max(levels)
    if 'extend' in list(kwargs.keys()):
        extend = kwargs['extend']
    elif (cmin < rmin) & (cmax > rmax):
        extend = 'both'
    elif (cmin < rmin) & (cmax <= rmax):
        extend = 'min'
    elif (cmin >= rmin) & (cmax > rmax):
        extend = 'max'
    elif (cmin >= rmin) & (cmax <= rmax):
        extend = 'neither'

    if scale == 'log2':
        Power = np.log2(power)
        Levels = np.log2(levels)
    else:
        Power = power
        Levels = levels

    cf = ax.contourf(t, np.log2(period), Power, Levels, extend=extend)
    ax.contour(t, np.log2(period), sig95, [-99, 1], colors='k',
               linewidths=2.)
    # q = ax.quiver(t[::da[1]], np.log2(period)[::da[0]], u[::da[0], ::da[1]],
    #               v[::da[0], ::da[1]], units='width', angles='uv', pivot='mid',
    #               linewidth=1.5, edgecolor='k', headwidth=10, headlength=10,
    #               headaxislength=5, minshaft=2, minlength=5)
    ax.fill(np.concatenate([t[:1]-dt, t, t[-1:]+dt, t[-1:]+dt, t[:1]-dt, t[:1]-dt]),
            np.log2(np.concatenate([[1e-9], coi, [1e-9], period[-1:], period[-1:], [1e-9]])),
            'k', alpha=0.3, hatch='x')
    Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),
                            np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(Yticks))
    ax.set_yticklabels(Yticks)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim(np.log2([period.min(), min([coi.max(), period.max()])]))
    ax.invert_yaxis()
    cbar = fig.colorbar(cf, ticks=Levels, extend=extend)
    cbar.ax.set_yticklabels(labels)
    result.append(ax)

    return result


def plot_channels(df, title='', color="b", figsize=(20, 12),
                  curation_widget=False):
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    axes[0].plot(df.index, df['Ax'], color=color)
    axes[0].set_ylabel("Ax (m/s$^2$)")
    axes[0].set_title(title)
    axes[1].plot(df.index, df['Ay'], color=color)
    axes[1].set_ylabel("Ay (m/s$^2$)")
    axes[2].plot(df.index, df['Az'], color=color)
    axes[2].set_ylabel("Az (m/s$^2$)")
    axes[3].plot(df.index, df['Am'], color=color)
    axes[3].set_ylabel("Am (m/s$^2$)")

    if curation_widget:
        # Make room for the widgets
        fig.subplots_adjust(bottom=0.25)

        # Default locations for cursors
        len_index = len(df.index)
        L0 = int(len_index*0.15)
        R0 = int(len_index*0.85)

        # Add cursors to the time plot
        l = axes[3].axvline(df.index[L0], lw=1, color='red')
        m = axes[3].axvline(df.index[R0], lw=1, color='red')

        # Create left and right sliders
        axcolor = 'lightgoldenrodyellow'
        ax_Left = plt.axes([0.1, 0.15, 0.8, 0.03], axisbg=axcolor)
        ax_Right = plt.axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)

        Slide_Left = Slider(
            ax_Left, 'Left', 0, len_index, valinit=L0, valfmt='%0.0f')
        Slide_Right = Slider(
            ax_Right, 'Right', 0, len_index, valinit=R0, valfmt='%0.0f')

        def update(val):
            Left_Cut = Slide_Left.val
            Right_Cut = Slide_Right.val
            l.set_xdata(df.index[Left_Cut])
            m.set_xdata(df.index[Right_Cut])
            fig.canvas.draw_idle()

        Slide_Left.on_changed(update)
        Slide_Right.on_changed(update)

        # Reset and Copy buttons
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        copyax = plt.axes([0.1, 0.025, 0.1, 0.04])
        copybutton = Button(copyax, 'Copy', color=axcolor, hovercolor='0.975')

        def reset(event):
            Slide_Left.reset()
            Slide_Right.reset()

        def copy_to_clipboard(event):
            x = "('" \
                 + str(df.index[int(Slide_Left.val)])[:19] \
                 + "', '" \
                 + str(df.index[int(Slide_Right.val)])[:19] \
                 + "'),\c"
            os.system("echo \"%s\" | pbcopy" % x)

        button.on_clicked(reset)
        copybutton.on_clicked(copy_to_clipboard)

        # Hold on to these objects so they don't get garbage collected
        # when this function exits.
        ax_Left._slide_left = Slide_Left
        ax_Right._slide_right = Slide_Right
        resetax._reset_button = button
        copyax._copy_button = copybutton

    return (fig, axes)


def plot_mag_wav(df_time, wav, low_freq=0.5, high_freq=25.0, cmap='afmhot',
                 vmin=None, vmax=None, title='', color="b", figsize=(20, 12),
                 curation_widget=False):
    """
    Plot accelerometer magnitude time series and wavelet decomposition.

    For plotting the wavelet data this function uses some fancy footwork
    with imshow to gain performance improvements over pcolormesh.

    Parameters
    ----------

    df_time : pandas dataframe of accelerometer data.

    wav : dictionary of wavelet transform results.

    Returns
    -------

    fig, axes : matplotlib objects.

    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    # `imshow` really wants floating point numbers in the `extent`
    # argument. Convert to seconds since the epoch.
    times = mdates.date2num(df_time.index._mpl_repr())
    axes[0].imshow(np.abs(wav['W']), cmap=cmap, aspect='auto',
                   extent=(times[0], times[-1],
                           wav['freqs'][-1], wav['freqs'][0]),
                   vmin=vmin, vmax=vmax)
    # Now tell `matplotlib` to represent seconds since the epoch as datetimes.
    axes[0].xaxis_date()
    # Frequency spacing the wavelet results is logarithmic.
    axes[0].set_yscale('log')
    axes[0].set_ylim(bottom=low_freq, top=high_freq)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(False)
    axes[0].set_title(title)
    axes[1].plot(df_time.index._mpl_repr(), df_time['Am'], color=color)
    axes[1].set_ylabel("Am (m/s$^2$)")

    if curation_widget:
        # Make room for the widgets
        fig.subplots_adjust(bottom=0.25)

        # Default locations for cursors
        len_index = len(df_time.index)
        L0 = int(len_index*0.15)
        R0 = int(len_index*0.85)

        # Add cursors to the time plot
        l = axes[1].axvline(df_time.index[L0], lw=1, color='red')
        m = axes[1].axvline(df_time.index[R0], lw=1, color='red')

        # Create left and right sliders
        axcolor = 'lightgoldenrodyellow'
        ax_Left = plt.axes([0.1, 0.15, 0.8, 0.03], axisbg=axcolor)
        ax_Right = plt.axes([0.1, 0.1, 0.8, 0.03], axisbg=axcolor)

        Slide_Left = Slider(
            ax_Left, 'Left', 0, len_index, valinit=L0, valfmt='%0.0f')
        Slide_Right = Slider(
            ax_Right, 'Right', 0, len_index, valinit=R0, valfmt='%0.0f')

        def update(val):
            Left_Cut = Slide_Left.val
            Right_Cut = Slide_Right.val
            l.set_xdata(df_time.index[Left_Cut])
            m.set_xdata(df_time.index[Right_Cut])
            fig.canvas.draw_idle()

        Slide_Left.on_changed(update)
        Slide_Right.on_changed(update)

        # Reset and Copy buttons
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

        copyax = plt.axes([0.1, 0.025, 0.1, 0.04])
        copybutton = Button(copyax, 'Copy', color=axcolor, hovercolor='0.975')

        def reset(event):
            Slide_Left.reset()
            Slide_Right.reset()

        def copy_to_clipboard(event):
            x = "('" \
                 + str(df_time.index[int(Slide_Left.val)])[:19] \
                 + "', '" \
                 + str(df_time.index[int(Slide_Right.val)])[:19] \
                 + "'),\c"
            os.system("echo \"%s\" | pbcopy" % x)

        button.on_clicked(reset)
        copybutton.on_clicked(copy_to_clipboard)

        # Hold on to these objects so they don't get garbage collected
        # when this function exits.
        ax_Left._slide_left = Slide_Left
        ax_Right._slide_right = Slide_Right
        resetax._reset_button = button
        copyax._copy_button = copybutton

    return (fig, axes)


def plot_cwt(cwt_vals):
    """
    cwt_vals is a df that comes from a PartitionedDataset. It comes
    in with time along the first dimension and frequency along the second.
    Transpose before plotting.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.imshow(cwt_vals.T, aspect='auto', interpolation='nearest', cmap='hot')
    ax.grid(False)
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (sec)')
    cwt_copy = cwt_vals.copy()
    cwt_copy.index = cwt_vals.index.get_level_values('time')
    time = np.linspace(0, sbtime.total_seconds(cwt_copy), cwt_copy.shape[0])
    xticks = np.linspace(0, cwt_vals.shape[0] - 1, 11, endpoint=True, dtype=int)
    xticklabels = list(map(lambda x: '{:.2f}'.format(x), time[xticks]))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    yticks = np.linspace(0, cwt_vals.shape[1], 7)
    ytick_vals = np.logspace(
        np.log10(cwt_copy.columns[-1]), np.log10(cwt_copy.columns[0]), 7)[::-1]
    yticklabels = list(map(lambda x: '{:.2f}'.format(x), ytick_vals))
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title('Continuous Wavelet Transform of Accel Magnitude')
    return (fig, ax)


def plot_cycles(normed_cycles, ave_normed, cleaned_cycles, ave_cleaned,
                title=None):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True, sharey=True)
    axes[0].plot(normed_cycles.T)
    axes[0].plot(ave_normed, color='0.05', lw=5, alpha=1.0)
    axes[0].set_ylabel("Normalized Cycles")
    axes[1].plot(cleaned_cycles.T)
    axes[1].plot(ave_cleaned, color='0.05', lw=5, alpha=1.0)
    axes[1].set_ylabel("Cleaned Cycles")
    axes[1].set_xlabel("Sample Number")
    if title:
        axes[0].set_title(title)
    return (fig, axes)


def plot_step_cycles(raw_df, title="Step Cycle Segmentation"):
    details = step.step_cycle_details(
        raw_df['Am'], sbtime.sampling_frequency_hz(raw_df))
    mr = details['mr']
    smoothed = details['smoothed']
    max_sal = details['max_sal']
    high_peaks = details['high_peaks']
    cent = details['centroids']

    zero_crossings = step.zero_crossings(smoothed)
    fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharex=True)
    axes[0].plot(raw_df.index, mr, 'b-')
    axes[0].set_ylabel("Raw Accel\nMagnitude (m/s$^2$)")
    axes[0].set_title(title)
    axes[1].plot(raw_df.index, smoothed, 'g-')
    axes[1].plot(raw_df.index[zero_crossings], smoothed[zero_crossings], 'ro')
    axes[1].plot(raw_df.index[high_peaks], smoothed[high_peaks],
                 'ko', mfc='c', mew=2.0)
    axes[1].set_ylabel("Smoothed Accel\nMagnitude (m/s$^2$)")
    axes[2].plot(raw_df.index, max_sal, 'k-', label='salience')
    axes[2].plot(raw_df.index, cent, 'y-', label='centroid')
    axes[2].set_ylabel("Salience, Centroids")
    axes[2].legend()
    return (fig, axes)
