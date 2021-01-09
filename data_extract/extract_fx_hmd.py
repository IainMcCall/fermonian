"""
Provides functions extract daily FX rates at ECT 3pm.

"""
import logging

from forex_python.converter import get_rate
import datetime
import pandas as pd

from data_extract.dates_extract import filter_update_dates


def fx_master_update(file_name, update_date, all_dates, major, minor):
    logger = logging.getLogger('main')
    master = pd.read_csv(file_name, index_col='Date', parse_dates=True)
    new_dates = filter_update_dates(all_dates, update_date, max(master.index.values))
    for d in new_dates:
        d = datetime.datetime(int(str(d)[:4]), int(str(d)[5:7]), int(str(d)[8:10]))
        logger.info("Extracting fx rates for " + str(d))
        for ccy in (minor + major):
            try:
                master.at[d, ccy] = get_rate('USD', ccy, d)
            except:
                logger.error("No rates available for " + ccy + ' on date: ' + str(d))
                master.at[d, ccy] = None
    master.to_csv(file_name)
