"""

"""
import os

import datetime
import pandas as pd

from quant_functions.basic import linear_interpolate_1d

BASE_TENORS = [45,136,227,318,410,501,592,683,775,866,957,1048,1140,1231,1322,1413,1505,1596,1687,1778,1870]


def interpolate_rates(tenors, rates, pillars):
    return linear_interpolate_1d(tenors, rates, pillars)


def main():
    ir_hmd = pd.read_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'short_sterling.csv'), index_col=0)
    expiries = pd.read_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'expiries.csv'), index_col='contract')
    ir_hmd['expiry'] = [datetime.datetime.strptime(expiries.at[x, 'expiry'], "%Y-%m-%d") for x in ir_hmd.index]
    output_hmd = pd.DataFrame(index=BASE_TENORS)
    for date in ir_hmd:
        if date != 'expiry':
            d = datetime.datetime.strptime(date, "%Y-%m-%d")
            futures = ir_hmd[[date, 'expiry']].dropna()
            if len(futures) > 0:
                rates = (100 - futures[date]) / 100
                days = [int((t - d).days) for t in futures['expiry']]
                output_hmd[date] = interpolate_rates(days, rates, BASE_TENORS)
            print('Calculating curves for ' + date)
    output_hmd.T.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'short_sterling_interpolated.csv'))


if __name__ == '__main__':
    main()