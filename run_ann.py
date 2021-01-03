import os

import numpy as np
import pandas as pd

from quant_functions.neural_network import NeuralNetwork
from quant_functions.basic import levels_to_returns


def main():
    work_dir = os.path.join('C:/', 'LocalFolder', 'temp')
    golden_dates = pd.read_csv(os.path.join(work_dir, 'golden_dates.csv'))
    hmd = pd.read_csv(os.path.join(work_dir, 'fx_hmd.csv'), index_col=0)
    hmd = hmd[hmd.index.isin(golden_dates['Date'])]
    hmd = hmd[-100:]
    ret = levels_to_returns(hmd, 'log', 1)
    # x = np.array(ret[['GBP', 'EUR']][1:].values) # Daily 1-day lag returns
    # y = np.array(ret['GBP'][:-1].values) # Next 1-day return
    # z = NeuralNetwork(x, y)
    x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([0, 1, 1, 0])
    z = NeuralNetwork(x, y)
    errors = []
    for i in range(10000):
        z.feedforward()
        z.backprop()
        errors.append(z.error)
        print('sim ' + str(i) + ' error: ' + str(z.error))
    print(z.output)
    pd.DataFrame(errors).to_csv(os.path.join(work_dir, 'ann_errors.csv'))


if __name__ == "__main__":
        main()