import biosignalsnotebooks as bsnb
import pandas as pd

def add_timestamp(_sig,sample_rate, start_time = None):
    """ Add a timestamp to the signal
    
    Parameters
    ----------
    _sig : list
        input signal
    sample_rate: int
        sampling frequency
    start_time: string
        start of data capture of the signal, defaults to None
        Possible formats:
        a. 'year-month-day hour-minutes-seconds' ex. '2020-07-02 11:54:02'
        b. 'hour-minutes'/'hour-minutes-seconds' ex. '11:54:02'/'11:54'
        If 'year-month-day' is not specified, the default will be the current date
    
    Returns
    -------
    _df_sig: DataFrame
        Columns: `timestamp` and `signal`
    """
    _sig_t = bsnb.generate_time(_sig, sample_rate)
    if start_time is None:
        _df_sig = pd.DataFrame({"timestamp": _sig_t, "signal": _sig})
    else:
        _df_sig = pd.DataFrame({"timestamp": _sig_t, "signal": _sig})
        _df_sig["timestamp"] = pd.Timestamp(start_time) + pd.to_timedelta(_df_sig['timestamp'], unit='s')

    return(_df_sig)

