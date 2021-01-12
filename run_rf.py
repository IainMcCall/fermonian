import os

import numpy as np
import pandas as pd

from quant_functions.random_forest import random_forest_fun
from quant_functions.basic import levels_to_returns


def main():
    work_dir = os.path.join('C:/', 'LocalFolder', 'temp', 'random_forest')
    data_dir = os.path.join('C:/', 'LocalFolder', 'temp', 'fx')
    golden_dates = pd.read_csv(os.path.join(data_dir, 'golden_dates.csv'))
    hmd = pd.read_csv(os.path.join(data_dir, 'fx_hmd.csv'), index_col=0, parse_dates=True)
    hmd = hmd[hmd.index.isin(golden_dates['Date'])][-100:]
    ret = levels_to_returns(hmd, 'log', 1) * 10000
    ret = ret.fillna(0)
    ret.to_csv(os.path.join(work_dir, 'ret.csv'))
    x = ret[:-1]
    y = ret['GBP'][1:]
    rf = random_forest_fun(x, y, work_dir)
    print(rf)
