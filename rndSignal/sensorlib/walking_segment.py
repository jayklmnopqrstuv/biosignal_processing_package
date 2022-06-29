import numpy as np
import pandas as pd
from .features import cwt_fft
from .time import resample_df


def cwt_transform(accel_df):
    """
    This function will resample the accelerometer dataframe with
    time index to 48 Hz and transform to continuous wavelet transform dataframe
    with time index and frequency bins as columns.

    Input:  accel_df    - accelerometer dataframe with time index.
    Return: cwt_df      - continuous wavelet transform  dataframe.
    """

    # Set new sampling frequency in Hz
    samp_freq = 48.0
    samp_offset = pd.tseries.offsets.Micro(n=round(1.0e6 / samp_freq))
    # Compute new sampling period
    dt = samp_offset.nanos / 1e9
    # Resample at new sampling frequency
    accel = resample_df(accel_df, samp_freq)
    # Compute CWT
    res, freq, time = cwt_fft(accel['Am'].values, dt=dt, low_freq=0.8, high_freq=None)

    # Get the CWT magnitude, frequency and time
    cwt_magnitude = np.abs(res['W'][freq][:, time])
    freqs = res['freqs'][freq]
    times = accel.index[time]

    # Create a DataFrame of CWT magnitude, frequency and time
    cwt_df = pd.DataFrame(cwt_magnitude.T, columns=freqs, index=times)

    return cwt_df


def extract_feature_predict_rolling_mean(cwt_df, rf_clf, nmf):
    """
    This function will:

        1. Extract features from CWT dataframe for
        every 2 second window with 1 second overlap.
        2. Get the NMF Transformation of the features.
        3. Get the predicted probability using the Random
        Forest Model.

    Input:  cwt_df  - Dataframe of CWT with time index and frequency bins as columns
            rf_clf  - Random Forest Classifier Model
            nfm     - NMF model

    Return: Dataframe of predicted probabilities with time index for
            every 2 seconds window and 1 second overlap.
    """
    window = 96
    h = int(window / 2)

    # Extract Feature
    cwts_mean = cwt_df.rolling(window=window, center=True).mean()[h::h].dropna()

    # NMF Dimension Reduction
    cwts_mean_nmf = nmf.transform(cwts_mean)

    # Get Predicted Probability using Random Forest Model
    clf_predict_proba_mean = rf_clf.predict_proba(cwts_mean_nmf).T[1]

    return clf_predict_proba_mean


def timestamp_probability(predicted_probabilities_mean, cwt_df):
    """
    This Function will give a Walking Probability on each row of the cwt_df.

    Input:  predicted_probabilities_mean -  Dataframe of probabilities given to CWT
                                            for every 2 seconds window with 1 second
                                            overlap.

            cwt_df                       -  DataFrame of CWT with 48 Hz resampled rate.

    Return: Dataframe of walking probability for each row of CWT Dataframe with  48 Hz
            resampled rate.
    """

    # Even indexed probabilities
    primary_index_proba = predicted_probabilities_mean[::2]

    # Odd indexed probabilities
    secondary_index_proba = predicted_probabilities_mean[1::2][0:len(primary_index_proba)-1]


    # Compute the average of the overlapped probabilities
    proba_primary = []
    for i in range(len(primary_index_proba)):
        proba_primary = proba_primary + [primary_index_proba[i]]*96

    proba_secondary = [primary_index_proba[0]]*48
    for i in range(len(secondary_index_proba)):
        proba_secondary = proba_secondary + [secondary_index_proba[i]]*96
    proba_secondary = proba_secondary + [primary_index_proba[-1]]*48

    len_primary_proba = len(proba_primary)

    cwt_truncate = cwt_df.head(len_primary_proba).copy()

    true_probability = np.mean(np.array([proba_primary,proba_secondary]),axis=0)

    cwt_truncate['probability'] = true_probability

    timestamp_probability_df = cwt_truncate.reset_index()[['time','probability']]

    return timestamp_probability_df


def segment_start_end_probability(time_probability_df):

    """
    This Function will detect the walking segments and return a list of
    start and end times of each segment.

    Input:  time_probability_df  - Dataframe of time and probability of walking
                                 at 48 Hz sampling rate.

    Return: Dataframe of walking segments with start and end times. Each segment has
            a probability of walking.
    """

    segments_time_stamp = []
    segment_probability = []

    time_stamp = []
    group_mean_probability = []
    space_counter = 0

    space_allowed_between_segment = 1 # second
    walking_probability_threshold = 0.5

    window = 48
    h = int(window)

    # Extract Feature
    time_probability_df = time_probability_df.set_index('time').rolling(window=window, center=True).mean()[h::h].dropna().reset_index()

    for i in range(int(len(time_probability_df))):
        local_mean_probability = time_probability_df['probability'][i]
        time_center = time_probability_df['time'][i]

        if local_mean_probability > walking_probability_threshold:
            time_stamp.append(time_center)
            group_mean_probability.append(local_mean_probability)
            space_counter = 0

            if i == int(len(time_probability_df)) - 1 and len(time_stamp) > 10:
                segments_time_stamp.append([time_stamp[0],time_stamp[-(space_allowed_between_segment)], \
                                                np.mean(group_mean_probability[:-(space_allowed_between_segment)])])
                time_stamp = []
                space_counter = 0
                group_mean_probability = []
        else:
            space_counter += 1
            if len(time_stamp) <= 10:
                if space_counter == space_allowed_between_segment + 1:
                    time_stamp = []
                    space_counter = 0
                    group_mean_probability = []
                if space_counter < space_allowed_between_segment + 1:
                    time_stamp.append(time_center)
                    group_mean_probability.append(local_mean_probability)

            if len(time_stamp) > 10:
                if space_counter == space_allowed_between_segment + 1:
                    segments_time_stamp.append([time_stamp[0],time_stamp[-(space_allowed_between_segment + 1)], \
                                                np.mean(group_mean_probability[:-(space_allowed_between_segment + 1)])])
                    time_stamp = []
                    space_counter = 0
                    group_mean_probability = []

                if space_counter < space_allowed_between_segment + 1:
                    time_stamp.append(time_center)
                    group_mean_probability.append(local_mean_probability)

                    if i == int(len(time_probability_df)) - 1:
                        segments_time_stamp.append([time_stamp[0],time_stamp[-(space_allowed_between_segment)], \
                                                np.mean(group_mean_probability[:-(space_allowed_between_segment)])])
                        time_stamp = []
                        space_counter = 0
                        group_mean_probability = []

    df_start_end_proba = pd.DataFrame(segments_time_stamp, columns=['segment_started_at','segment_ended_at','walking_probability'])

    return df_start_end_proba


def get_walking_segment(accel_df, rf_clf, nmf):

    """
    Main Function to get the walking segment

    Input:  accel_df    - Dataframe of accelerometer with time index
            rf_clf      - Random Forest Classifier Model
            nmf         - NMF model

    Return: Dataframe of Walking Segments with start time, end time and
            walking probability
    """

    # Resample to 48 Hz and Compute CWT
    cwt_df = cwt_transform(accel_df)

    # Extract Feature and Predict every 2 seconds with 1 second overlap
    predicted_probability_rolling_mean = extract_feature_predict_rolling_mean(cwt_df, rf_clf, nmf)

    # Compute the probability for each row of CWT DataFrame at 48 Hz
    timestamp_probability_df = timestamp_probability(predicted_probability_rolling_mean, cwt_df)

    # Detect Walking Segments, Start Time, End Time and Probability
    return segment_start_end_probability(timestamp_probability_df)
