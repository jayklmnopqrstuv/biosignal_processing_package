import funcy as fp
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns


def add_verification_labels(df, user_identifier='user_name'):
    df['genuine'] = (df[user_identifier + '_ref'] ==
                     df[user_identifier + '_att'])
    df['label'] = 'Genuine'
    df.ix[~df['genuine'], 'label'] = 'Impostor'
    return df


__DEFAULT_DISTPLOT_KARGS = {'kde': True, 'norm_hist': True}


def distplot_pdist_df(pdist_df, figsize=(10, 7), **distplot_kargs):
    plt.figure(figsize=figsize)
    for label in ['Genuine', 'Impostor']:
        sns.distplot(pdist_df.query('label == "{}"'.format(label))['proba'],
                     label=label,
                     **fp.merge(__DEFAULT_DISTPLOT_KARGS, distplot_kargs))
    plt.legend()


__DEFAULT_GRID = {'col_wrap': 3, 'sharey': False, 'sharex': True}


def distplot_pdist_df_users(pdist_df, min_genuine=1, grid={}, distplot={},
                            user_identifier='user_name'):
    # there is probably a better way of doing this without creating the
    # intermediate counts df...
    ref_user = user_identifier + '_ref'
    counts = pdist_df.groupby(ref_user)['label'].value_counts()
    counts.name = 'size'
    counts = pd.DataFrame(counts)
    users_with_data = counts.query(
        "label == 'Genuine' and size > {}".format(min_genuine))

    users = users_with_data.reset_index()[ref_user].tolist()
    df = pdist_df.query('{} == {}'.format(ref_user, users))
    # plt.figure(figsize=(20,20))
    grid = fp.merge(__DEFAULT_GRID, grid)
    fg = sns.FacetGrid(df, col=ref_user, hue='label', **grid)
    distplot_kargs = fp.merge(__DEFAULT_DISTPLOT_KARGS, distplot)
    fg = (fg.map(sns.distplot, 'proba', **distplot_kargs).add_legend())
