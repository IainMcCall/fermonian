import os

import numpy as np
import pandas as pd

from quant_functions.neural_network import NeuralNetwork, sklearn_ann
from quant_functions.basic import levels_to_returns

from sklearn.neural_network import MLPClassifier


def main():
    work_dir = os.path.join('C:/', 'LocalFolder', 'temp', 'ann')
    hmd = pd.read_csv(os.path.join(work_dir, 'eur_gbp.csv'), index_col=0)
    hmd = hmd[-250:]
    ret = pd.DataFrame()
    for r in hmd:
        ret[r] = levels_to_returns(hmd[r], 'log', 1, overlapping=False, data_fill='interp_lin')
        # bools = []
        # for x in ret[r]:
        #     if x >= 0:
        #         bools.append(1)
        #     elif x < 0:
        #         bools.append(0)
        # ret[r] = bools
    ann, success_rate, parameters = sklearn_ann(ret[1:], ret['GBP'][:-1], 0.2)


    # # x = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    # # y = np.array([0, 1, 1, 0])
    # # z = NeuralNetwork(x, y)
    #
    # errors = []
    # for i in range(100):
    #     z.feedforward()
    #     z.backprop()
    #     errors.append(z.error)
    #     print('sim ' + str(i) + ' error: ' + str(z.error))
    # print(z.output)
    # pd.DataFrame(errors).to_csv(os.path.join(work_dir, 'ann_errors.csv'))


if __name__ == "__main__":
        main()