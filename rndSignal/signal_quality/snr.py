# Import libraries
import biosignalsnotebooks as bsnb
import math
import numpy as np
import scipy

def snr(sig, sr, value = False):
    """This function returns the peak-to-peak amplitudes of the signal and noise components as 
    well as the signal to noise ratio (SNR) measured in decibels (dB). The SNR metric classifies
    objectively the quality of the acquisition, and like the name suggests, the relation
    between the intensity of the signal and the undesired noise in the acquired data, which
    is defined by the two aforementioned components, respectively.
   
    Parameters
    ----------
    sig : array
        The raw or filtered ECG signal data
    sr : int
        The sampling frequency of the signal (in Hz, i.e., samples/second).
    value : bool
        If True, the function only returns the SNR value itself.
       
    Returns
    -------
    Message
        An output of the print function detailing the amplitudes of the signal and noise
        components as well as the SNR value. 
    
    snr_db: float
        If the return argument is set to True, then the function returns the SNR value itself 
        and doesn't print the message.
    """
    
    # Measuring the peak-to-peak amplitude of the signal component
    vpp_signal = np.ptp(sig)
    
    # Finding R peaks
    time_r_peaks, amplitude_r_peaks = bsnb.detect_r_peaks(sig, sr, time_units = True, 
                                                          plot_result = False)

    # Measuring the peak-to-peak amplitude of the noise component
    # Compute the average noise amplitude from several R peaks
    vpp_noise = []
    
    for t in time_r_peaks:
        start = int((t + 0.5) * sr) # 0.5 - time between a peak and a flat
        end = int((t + 0.65)* sr) # 0.65 - time between a peak and the end of the flat
        # to make sure that the start and end points are not out of bounds
        if(start and end <= len(sig)): 
            interval = sig[start:end]
            vpp = np.ptp(interval)
            vpp_noise.append(vpp)
        
    # nanmean is used instead of mean to ensure that the result will not return NAN values 
    vpp_noise = np.nanmean(vpp_noise)
    
    # Computing signal to noise ratio (SNR) in dB
    snr = vpp_signal/vpp_noise

    # The multiplication by 20 is needed since the signals are in the unit of (micro)Siemes
    snr_db = 20 * math.log10(snr)

    if(value == True):
        return(snr_db)
    else:
        print("Amplitude of ECG signal: {} mv".format(vpp_signal))
        print("Amplitude of noise signal: {} mv".format(vpp_noise))
        print("SNR for ECG signal: {} dB".format(snr_db))
