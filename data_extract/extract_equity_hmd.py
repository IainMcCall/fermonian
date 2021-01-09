"""
Provides functions extract daily equity prices from Yahoo finance. Snap time is close of exchange.

"""
import logging

import yfinance as yf
import datetime
import pandas as pd

from data_extract.dates_extract import filter_update_dates


def equity_master_update(file_name, update_date, all_dates, tickers):
    logger = logging.getLogger('main')
    master = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    new_dates = filter_update_dates(all_dates, update_date, max(master.index.values))
    existing_tickers = ''
    new_tickers = ''
    master_tickers = [t.replace('_close', '') for t in master.columns]
    for t in tickers:
        if t in master_tickers:
            existing_tickers = existing_tickers + t + ' '
        else:
            new_tickers = new_tickers + t + ' '
    existing_tickers = existing_tickers[:-1]
    new_tickers = new_tickers[:-1]

    logger.info("Extracting equity data for existing tickers")
    if len(existing_tickers) > 0:
        new_data = yf.download(existing_tickers, start=str(max(master.index.values))[:10], end=str(update_date)[:10])
        for d in new_dates:
            d = datetime.datetime(int(str(d)[:4]), int(str(d)[5:7]), int(str(d)[8:10]))
            for t in existing_tickers.split(' '):
                try:
                    master.at[d, t + '_close'] = new_data['Close'][t][d]
                    master.at[d, t + '_open'] = new_data['Open'][t][d]
                    master.at[d, t + '_low'] = new_data['Low'][t][d]
                    master.at[d, t + '_high'] = new_data['High'][t][d]
                    master.at[d, t + '_volume'] = new_data['Volume'][t][d]
                except:
                    logger.error("No equity available for " + t + ' on date: ' + str(d))
                    master.at[d, t + '_close'] = None
                    master.at[d, t + '_open'] = None
                    master.at[d, t + '_low'] = None
                    master.at[d, t + '_high'] = None
                    master.at[d, t + '_volume'] = None

    logger.info("Extracting equity data for new tickers")
    if len(new_tickers) > 0:
        new_data = yf.download(new_tickers, start=str(min(master.index.values))[:10], end=str(update_date)[:10])
        for d in master.index:
            for t in new_tickers.split(' '):
                try:
                    master.at[d, t + '_close'] = new_data['Close'][t][d]
                    master.at[d, t + '_open'] = new_data['Open'][t][d]
                    master.at[d, t + '_low'] = new_data['Low'][t][d]
                    master.at[d, t + '_high'] = new_data['High'][t][d]
                    master.at[d, t + '_volume'] = new_data['Volume'][t][d]
                except:
                    logger.error("No equity available for " + t + ' on date: ' + str(d))
                    master.at[d, t + '_close'] = None
                    master.at[d, t + '_open'] = None
                    master.at[d, t + '_low'] = None
                    master.at[d, t + '_high'] = None
                    master.at[d, t + '_volume'] = None
    master.to_csv(file_name)