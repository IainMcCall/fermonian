"""
Provides functions to call data from quandl.

"""

import os

import pandas as pd
import quandl

quandl.ApiConfig.api_key = 'Zh-i4zVaLQsXGPZzeFDe'


def main():
    # # EURIBOR 3M futures data
    # YEARS = ['2020', '2021', '2022', '2023', '2024', '2025']
    # MONTHS = ['H', 'M', 'U', 'Z']
    # data = pd.DataFrame()
    # for y in YEARS:
    #     for m in MONTHS:
    #         ticker = 'EUREX/FEU3' + m + y
    #         print('Extracting data for ' + ticker)
    #         new_data = quandl.get(ticker)['Settle'].rename(ticker)
    #         data = data.append(new_data, ignore_index=False)
    # data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'euribor.csv'))
    #
    # # EONIA futures data
    # YEARS = ['2020', '2021', '2022', '2023', '2024', '2025']
    # MONTHS = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
    # data = pd.DataFrame()
    # for y in YEARS:
    #     for m in MONTHS:
    #         ticker = 'EUREX/FEO1' + m + y
    #         print('Extracting data for ' + ticker)
    #         try:
    #             new_data = quandl.get(ticker)['Settle'].rename(ticker)
    #             data = data.append(new_data, ignore_index=False)
    #         except:
    #             print('No quandle data available for ' + ticker)
    # data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'eonia.csv'))


    # STERLING futures data
    # CONTRACTS = range(16)
    # data = pd.DataFrame()
    # for c in CONTRACTS:
    #     ticker = 'CHRIS/LIFFE_L' + str(c)
    #     print('Extracting data for ' + ticker)
    #     try:
    #         new_data = quandl.get(ticker)['Settle'].rename(ticker)
    #         data = data.append(new_data, ignore_index=False)
    #     except:
    #         print('No quandle data available for ' + ticker)
    # data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'short_sterling.csv'))

    # Economic data
    # data = pd.DataFrame()
    # ticker = 'ECB/BKN_H_AT_A020_Z_ZZZZ_ZZ_S_Q'
    # print('Extracting data for ' + ticker)
    # new_data = quandl.get(ticker)
    # data = data.append(new_data, ignore_index=False)
    # data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'economics', 'german_gdp.csv'))

    # Euronext data
    # data = pd.DataFrame()
    # ticker = 'EURONEXT/DB1D'
    # print('Extracting data for ' + ticker)
    # new_data = quandl.get(ticker)
    # data = data.append(new_data, ignore_index=False)
    # data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'euronext.csv'))

    # Office of National Statistics (UK)
    # data = pd.DataFrame()
    # ticker = 'UKONS/DLHT_A'
    # print('Extracting data for ' + ticker)
    # new_data = quandl.get(ticker)
    # data = data.append(new_data, ignore_index=False)
    # data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'economics', 'uk_gbp.csv'))

    # FRED
    data = pd.DataFrame()
    ticker = 'FRED/DGS5'
    print('Extracting data for ' + ticker)
    new_data = quandl.get(ticker)
    data = data.append(new_data, ignore_index=False)
    data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'ir', 'us_5y.csv'))


if __name__ == '__main__':
    main()