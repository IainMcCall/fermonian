"""
Provides functions extract daily ir rates from quandl. Snap time to be confirmed.

"""
import logging

import quandl
import datetime
import pandas as pd

from data_extract.dates_extract import filter_update_dates

quandl.ApiConfig.api_key = 'Zh-i4zVaLQsXGPZzeFDe'


def ir_master_update(file_name, update_date, all_dates, rates):
    logger = logging.getLogger('main')
    master = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    new_dates = filter_update_dates(all_dates, update_date, max(master.index.values))
    logger.info("Extracting ir data for existing tickers")
    if len(new_dates) > 0:
        start_date = min(new_dates)
        end_date = max(new_dates)
        start_date = datetime.datetime(int(str(start_date)[:4]), int(str(start_date)[5:7]), int(str(start_date)[8:10]))
        end_date = datetime.datetime(int(str(end_date)[:4]), int(str(end_date)[5:7]), int(str(end_date)[8:10]))
        for i in range(len(rates)):
            ticker = rates['quanl'][i]
            name = rates['risk_factor'][i]
            field = rates['field'][i]
            logger.info('Extracting rates data for ' + name)
            data = quandl.get(ticker, start_date=start_date, end_date=end_date)
            for d in new_dates:
                try:
                    master.at[d, name] = data.at[d, field]
                except:
                    logger.error("No ir rates available for " + name + ' on date: ' + str(d))
                    master.at[d, name] = None
        master.to_csv(file_name)
    else:
        logger.info("No new ir dates")
