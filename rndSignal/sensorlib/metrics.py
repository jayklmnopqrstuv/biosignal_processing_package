import funcy as fp
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.metrics import (roc_auc_score, roc_curve, auc)


# target number of days per false rejection.
FRR_MAX_DENOM0 = 90.0
FRR_MAX_DENOM1 = 30.0


def auc_score(paired_df):
    ret = roc_auc_score(paired_df['genuine'], paired_df['proba'])
    return ret


def det_curve(paired_df):
    false_accept_rate, true_positive_rate, thresholds = roc_curve(
        paired_df['genuine'], paired_df['proba'])
    false_reject_rate = 1.0 - true_positive_rate
    return false_accept_rate, false_reject_rate, thresholds


def sensitivity(conf_matrix):
    """
    Return sensitivity from standard sklearn confusion matrix.
    """
    return conf_matrix[1, 1] / conf_matrix[1, :].sum()


def specificity(conf_matrix):
    """
    Return specificity from standard sklearn confusion matrix.
    """
    return conf_matrix[0, 0] / conf_matrix[0, :].sum()


def det_curve_df(paired_df):
    false_accept_rate, false_reject_rate, thresholds = det_curve(paired_df)
    det_df = pd.DataFrame({'far': false_accept_rate, 'frr': false_reject_rate,
                           'thresh': thresholds})
    det_df['sensitivity'] = 1.0 - det_df['frr']
    det_df['specificity'] = 1.0 - det_df['far']
    # Add the Youden index because it's easy at this point.
    det_df['youden'] = det_df['sensitivity'] + det_df['specificity'] - 1.0
    return det_df


def max_youden_index(curve_df):
    """
    Find the pandas.Series in the DET curve dataframe where Youden's index
    is maximum.
    """
    max_idx = curve_df['youden'].argmax()
    return curve_df.loc[max_idx]


def eer(curve_df):
    eer_idx = np.argmin(np.abs(curve_df.far - curve_df.frr))
    return curve_df.iloc[eer_idx]


def find_target_frr(curve_df, frr):
    far = np.interp(frr, curve_df['frr'][::-1], curve_df['far'][::-1])
    thresh = np.interp(frr, curve_df['frr'][::-1], curve_df['thresh'][::-1])
    return pd.Series({'far': far, 'frr': frr, 'thresh': thresh})


def max_frr_denom(curve_df, max_denom):
    target_frr = 1. / max_denom
    info = find_target_frr(curve_df, target_frr)
    info['frr_denom'] = 1. / info['frr']
    info['far_denom'] = 1. / info['far']
    return info


def plot_det_curve(curve_df, max_frrs_denoms=[FRR_MAX_DENOM0, FRR_MAX_DENOM1],
                   fig=None, ax=None):
    false_accept_rate = curve_df['far']
    false_reject_rate = curve_df['frr']
    linestyles = ['-', '--', '-.', ':']
    assert len(max_frrs_denoms) <= len(linestyles)

    eer_info = eer(curve_df)
    _auc = auc(false_accept_rate, 1 - false_reject_rate, reorder=True)
    if fig is None and ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if fig is None and ax is not None:
        fig = ax.figure
    ax.plot(false_accept_rate, false_reject_rate, label='AUC {:.3f}'.format(_auc))

    for max_denom, ls in zip(max_frrs_denoms, linestyles):
        info = max_frr_denom(curve_df, max_denom)
        ax.axhline(info['frr'], color='red', ls=ls, lw=1,
                   label=('FRR {:.1%} {:0.1f}, FAR {:.1%} {:0.1f}, TH = {:0.3f}'
                          .format(info['frr'], info['frr_denom'], info['far'],
                                  info['far_denom'], info['thresh'])))

    ax.axis('square')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("FAR")
    ax.set_ylabel("FRR")
    ax.set_title("DET Curve")
    ax.legend(loc='upper right')
    text = "EER = {:0.1%}\nTH = {:0.3f}".format(eer_info['far'], eer_info['thresh'])
    ax.text(0.95, 0.7, text, fontsize=13, horizontalalignment='right', verticalalignment='top')
    return fig, ax


def ave_vote(df, thresh):
    res = df.iloc[0].copy()
    res['proba'] = (df['proba'] > thresh).astype(np.float64).mean()
    res['raw_score'] = (df['raw_score'] > thresh).astype(np.float64).mean()
    return res


def ave_proba(df, thresh=None):
    res = df.iloc[0].copy()
    res['proba'] = df['proba'].mean()
    res['raw_score'] = df['raw_score'].mean()
    return res


def aggregate_to_frame(df, method='ave_proba', thresh=None,
                       user_identifier='user_name'):
    """Aggregate individual observation probability scores to frame level."""
    df = df.drop(['cycle_id', 'cycle_start', 'cycle_end'], axis=1)
    if 'fp_name' in df.columns:
        gb_cols = [user_identifier + '_ref', 'fp_name',
                   user_identifier + '_att', 'loc', 'frame_id']
    else:
        gb_cols = [user_identifier + '_ref',
                   user_identifier + '_att', 'loc', 'frame_id']
    gb = df.groupby(gb_cols, as_index=False)
    if method == 'ave_proba':
        func = ave_proba
        thresh = None
    if method == 'ave_vote':
        func = ave_vote
        if thresh is None:
            raise ValueError("ave_vote method requires a threshold.")
    frame_vote = gb.apply(func, thresh).reset_index(drop=True)
    return frame_vote


# target number of days per false rejection.
FRR_MAX_DENOM0 = 90.0
FRR_MAX_DENOM1 = 30.0


def get_metrics(paired_df):
    auc = auc_score(paired_df)
    curve_df = det_curve_df(paired_df)
    target0 = find_target_frr(curve_df, 1.0 / FRR_MAX_DENOM0)
    target1 = find_target_frr(curve_df, 1.0 / FRR_MAX_DENOM1)
    return (auc, target0['far'], target0['thresh'],
            target1['far'], target1['thresh'])


def weighted_mean(x, weights):
    return x.multiply(weights, axis=0).sum(axis=0) / weights.sum()


# User-averaged performance
def user_averaged_perf(df, df_agg, user_identifier='user_name'):
    """
    Measure overall classifier performance by weighted average
    of individual user classifier performance.
    """
    # Preallocate a DataFrame for summary results
    measures = ['cycle_auc', 'cycle_far0', 'cycle_far0_thresh', 'cycle_far1',
                'cycle_far1_thresh', 'frame_auc', 'frame_far0',
                'frame_far0_thresh', 'frame_far1', 'frame_far1_thresh']
    idxs = []
    ref_user = user_identifier + '_ref'
    att_user = user_identifier + '_att'
    user_names = df[ref_user].unique()
    locs = df['loc'].unique()
    fps = df['fp_name'].unique()
    for user_name in user_names:
        for loc in locs:
            for fp_name in fps:
                idxs.append((user_name, loc, fp_name))
    results = np.empty((len(idxs), len(measures)))
    results[:] = np.NaN
    results = pd.DataFrame(
        results,
        index=pd.MultiIndex.from_tuples(idxs, names=('user_name', 'loc', 'fp_name')),
        columns=measures).sort_index()

    # Now fill in the results
    for user_name in user_names:
        for loc in locs:
            for fp_name in fps:
                this_fp = df[(df[ref_user] == user_name) &
                             (df['loc'] == loc) &
                             (df['fp_name'] == fp_name)]
                this_fp_agg = df_agg[(df_agg[ref_user] == user_name) &
                                     (df_agg['loc'] == loc) &
                                     (df_agg['fp_name'] == fp_name)]
                # Conditions that would prompt you to skip testing
                # this fingerprint.
                skip_it = (
                    # There are no paired attempts.
                    this_fp.empty or this_fp_agg.empty or
                    # All attempt results are genuine or all attempt results
                    # are impostor. In this case no DET curve can be made.
                    this_fp['genuine'].all() or
                    (this_fp['genuine'] == False).all() or
                    this_fp_agg['genuine'].all() or
                    (this_fp_agg['genuine'] == False).all()
                )
                if skip_it:
                    continue
                # cycle level
                auc, far0, thresh0, far1, thresh1 = get_metrics(this_fp)
                results.loc[(user_name, loc, fp_name), 'cycle_auc'] = auc
                results.loc[(user_name, loc, fp_name), 'cycle_far0'] = far0
                results.loc[(user_name, loc, fp_name), 'cycle_far0_thresh'] = thresh0
                results.loc[(user_name, loc, fp_name), 'cycle_far1'] = far1
                results.loc[(user_name, loc, fp_name), 'cycle_far1_thresh'] = thresh1
                # frame level
                auc, far0, thresh0, far1, thresh1 = get_metrics(this_fp_agg)
                results.loc[(user_name, loc, fp_name), 'frame_auc'] = auc
                results.loc[(user_name, loc, fp_name), 'frame_far0'] = far0
                results.loc[(user_name, loc, fp_name), 'frame_far0_thresh'] = thresh0
                results.loc[(user_name, loc, fp_name), 'frame_far1'] = far1
                results.loc[(user_name, loc, fp_name), 'frame_far1_thresh'] = thresh1

    # Aggregate
    results = results.dropna().sort_index()
    n_cycles = df.groupby(by=['loc', 'fp_name', att_user]).size()
    results_agg = results.groupby(level=['loc', 'fp_name']).agg(
        lambda x: weighted_mean(x, n_cycles.loc[
            ((x.index.get_level_values('loc')[0]),
             (x.index.get_level_values('fp_name')[0])
             )].values))
    return results, results_agg


# Micro-averaging considers all positive and negative predictions together
# as a binary decision between only two classes. This is a good model of
# the genuine vs. impostor problem where we really don't care as much which
# specific user we are dealing with. Rather, we want to know if for any user
# the collection of classifiers correctly predicts genuine or impostor.
def micro_averaged_perf(df, df_agg):
    """
    Measure overall classifier performance by micro averaging
    of individual user classifier performance.
    """
    # Preallocate a DataFrame for summary results
    measures = ['cycle_auc', 'cycle_far0', 'cycle_far0_thresh', 'cycle_far1',
                'cycle_far1_thresh', 'frame_auc', 'frame_far0',
                'frame_far0_thresh', 'frame_far1', 'frame_far1_thresh']
    idxs = []
    locs = df['loc'].unique()
    fps = df['fp_name'].unique()
    for loc in locs:
        for fp_name in fps:
            idxs.append((loc, fp_name))
    results = np.empty((len(idxs), len(measures)))
    results[:] = np.NaN
    results = pd.DataFrame(
        results,
        index=pd.MultiIndex.from_tuples(idxs, names=('loc', 'fp_name')),
        columns=measures)

    # Now fill in the results
    for loc in locs:
        for fp_name in fps:
            this_fp = df[(df['loc'] == loc) &
                         (df['fp_name'] == fp_name)]
            this_fp_agg = df_agg[(df_agg['loc'] == loc) &
                                 (df_agg['fp_name'] == fp_name)]
            # Conditions that would prompt you to skip testing
            # this fingerprint.
            skip_it = (
                # There are no paired attempts.
                this_fp.empty or this_fp_agg.empty or
                # All attempt results are genuine or all attempt results
                # are impostor. In this case no DET curve can be made.
                this_fp['genuine'].all() or
                (this_fp['genuine'] == False).all() or
                this_fp_agg['genuine'].all() or
                (this_fp_agg['genuine'] == False).all()
            )
            if skip_it:
                continue
            # cycle level
            auc, far0, thresh0, far1, thresh1 = get_metrics(this_fp)
            results.loc[(loc, fp_name), 'cycle_auc'] = auc
            results.loc[(loc, fp_name), 'cycle_far0'] = far0
            results.loc[(loc, fp_name), 'cycle_far0_thresh'] = thresh0
            results.loc[(loc, fp_name), 'cycle_far1'] = far1
            results.loc[(loc, fp_name), 'cycle_far1_thresh'] = thresh1
            # frame level
            auc, far0, thresh0, far1, thresh1 = get_metrics(this_fp_agg)
            results.loc[(loc, fp_name), 'frame_auc'] = auc
            results.loc[(loc, fp_name), 'frame_far0'] = far0
            results.loc[(loc, fp_name), 'frame_far0_thresh'] = thresh0
            results.loc[(loc, fp_name), 'frame_far1'] = far1
            results.loc[(loc, fp_name), 'frame_far1_thresh'] = thresh1

    # Cleanup
    results = results.dropna().sort_index()
    return results
