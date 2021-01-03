import os

import pandas as pd
import yfinance as yf

TICKERS = ['^FTSE', '^GSPC', '^STOXX50E', '^N225']

for t in TICKERS:
    print("Extracting data for " + t)
    name = yf.Ticker(t)

    # get stock info
    # pd.DataFrame(data=name.info).to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', TICKER + '_info.csv'))
    # print(name.options)

    # get historical market data
    # hist = name.history(period="max")
    # hist.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', t + '_hmd.csv'))

    # show financials
    name.financials.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_financials.csv'))
    name.quarterly_financials.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_financials.csv'))

    # # show major holders
    # msft.major_holders.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_majority_holders.csv'))
    #
    # # show institutional holders
    # msft.institutional_holders.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_institutional_holders.csv'))
    #
    # # show balance sheet
    # msft.balance_sheet.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_balance_sheet.csv'))
    # msft.quarterly_balance_sheet.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_balance_sheet.csv'))
    #
    # # show cashflow
    # msft.cashflow.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_cashflow.csv'))
    # msft.quarterly_cashflow.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_cashflow.csv'))
    #
    # # show earnings
    # msft.earnings.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_earnings.csv'))
    # msft.quarterly_earnings.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_earnings.csv'))
    #
    # # show sustainability
    # msft.sustainability.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_sustainability.csv'))
    #
    # # show analysts recommendations
    # msft.recommendations.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_recommendations.csv'))
    #
    # # show next event (earnings, etc)
    # msft.calendar.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_calendar.csv'))
