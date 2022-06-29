from datetime import datetime
import numpy as np
import scipy.spatial.distance as dist
#import funcy as fp #removed temporarily to assist package creation
import pandas as pd


def progress_printer():
    start_time = datetime.now()

    def print_progress(message):
        now_time = datetime.now()
        print('{} {} {}'.format(now_time, now_time - start_time, message))
    return print_progress

def extract_index_as_df(df, levels=None):
    """
    Returns the index of the DataFrame as a standalone DataFrame.
    Works with single and MultiIndex DataFrames.
    """
    return df.reset_index().drop(df.columns, axis=1)

def pdist_df(df, dist_func='euclidean', suffixes=('_a', '_b')):
    """
    Wrapper around scipy.spatial.distance.pdist that operates
    on a pandas DataFrame whose columns represent a vector.

    A DataFrame is returned with the (Multi)Index of both pairs
    returned as columns with the supplied suffixes along with
    the 'dist' calculated for the pair.

    Example:

    In [9]: foo = pd.DataFrame(np.random.rand(5,3), index=pd.MultiIndex.from_tuples([('a',0), ('a',1), ('b', 2), ('c',3), ('c',4)]))

    In [10]: foo.index.names=('letter', 'num')

    In [11]: foo
    Out[11]:
    0         1         2
    letter num
    a      0    0.010078  0.426312  0.566798
    1    0.312353  0.150572  0.922964
    b      2    0.706812  0.797651  0.272342
    c      3    0.285297  0.286570  0.795366
    4    0.160878  0.565855  0.437329

    In [12]: pdist_df(foo)
    Out[12]:
    letter_a  num_a      dist letter_b  num_b
    0        a      1  0.542454        a      0
    1        b      2  0.842636        a      0
    2        c      3  0.384078        a      0
    3        c      4  0.242847        a      0
    4        b      2  0.998808        a      1
    5        c      3  0.188438        a      1
    6        c      4  0.656693        a      1
    7        c      3  0.844057        b      2
    8        c      4  0.615625        b      2
    9        c      4  0.470818        c      3
    """
    df = df.copy()
    df['row_id'] = np.arange(len(df))
    df.set_index('row_id', append=True, inplace=True)
    dist_matrix = dist.pdist(df.values, dist_func)
    index_tuples = df.index.tolist()

    def index_chunk(i):
        return fp.map(lambda t: t + (i,), fp.drop(i + 1, index_tuples))

    dist_ix_tuples = list(fp.mapcat(index_chunk, range(len(index_tuples) - 1)))
    dist_index = pd.MultiIndex.from_tuples(dist_ix_tuples, names=df.index.names + ['target_id'])
    dist_df = pd.DataFrame(dist_matrix, index=dist_index, columns=['dist']).reset_index()

    targets = extract_index_as_df(df)
    res = dist_df.merge(targets, left_on='target_id', right_on='row_id', suffixes=suffixes)
    to_drop = ['row_id' + suf for suf in suffixes] + ['target_id']
    return res.drop(to_drop, axis=1)
