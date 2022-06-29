import numpy as np
import pandas as pd
from sensorlib.features import cwt_fft


def cwt_features_rolling_mean(ppg,fs, nmf):
    res, freq, time = cwt_fft(ppg, dt=1.0/fs, low_freq=0.1, high_freq=6)
    cwt_magnitude = np.abs(res['W'][freq][:, time])
    freqs = res['freqs'][freq]
    df = pd.DataFrame(cwt_magnitude.T)

    window =  30
    h = int(window/3)

    df_mean = df.rolling(window=window,center=True).mean()[h::h].dropna()
    idx = df_mean.index
    features =nmf.transform(df_mean)

    features_df = pd.DataFrame(features, index = idx)

    return features_df


def get_good_ppg_segment(predict_proba_, idx, th, minimum_good_window, minimum_space_between_group):

    df_predicted_probability = pd.DataFrame(predict_proba_, index=idx, columns=['probas'])

    good_windows = []
    good_group = []
    good_windows_proba = []

    space_counter = 0
    for i in range(len(predict_proba_)):
        if predict_proba_[i] >= th:
            good_group.append(idx[i])
            space_counter = 0

            if len(good_group) > minimum_good_window and i == len(predict_proba_)-1:
                good_windows.append([good_group[0],good_group[-1]])

        else:
            space_counter+= 1

            if len(good_group) > minimum_good_window and space_counter < minimum_space_between_group:
                good_group.append(idx[i])
                if i == len(predict_proba_)-1:
                    good_windows.append([good_group[0],good_group[-space_counter-1]])

            elif len(good_group) > minimum_good_window and space_counter >= minimum_space_between_group:
                good_windows.append([good_group[0],good_group[-space_counter]])

                good_group = []
                space_counter = 0

            elif len(good_group) <= minimum_good_window and space_counter >= minimum_space_between_group:
                good_group = []
                space_counter = 0

            elif len(good_group) <= minimum_good_window and space_counter < minimum_space_between_group:
                if len(good_group) < 2:
                    good_group = []
                    space_counter = 0
                else:
                    good_group.append(idx[i])

            else:
                continue

    for m,n in good_windows:
        good_windows_proba.append(df_predicted_probability['probas'].ix[m:n].mean())

    df = pd.DataFrame(good_windows, columns=["start_index","end_index"])
    df["good_probability"] = good_windows_proba
    df["mean_good_probability"] = np.mean(good_windows_proba)
    df["total_probability"] = np.mean(predict_proba_)

    return df


def ppg_signal_quality(ppg, fs, nmf, clf):
    W = cwt_features_rolling_mean(ppg, fs, nmf)
    idx = W.index
    proba = clf.predict_proba(W).T[1]

    df_good_segment = get_good_ppg_segment(proba, idx, 0.8, 15, 2)

    df_proba = pd.DataFrame(proba, index=idx, columns=["probability"])

    return df_good_segment, df_proba
