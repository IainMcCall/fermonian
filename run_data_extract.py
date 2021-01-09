"""
Calls the functions to update the master databases.

"""
import configparser
import datetime
import logging
import os
import time

import numpy as np
import pandas as pd

from data_extract.dates_extract import business_dates
from data_extract.extract_fx_hmd import fx_master_update
from data_extract.extract_equity_hmd import equity_master_update
from data_extract.extract_ir_hmd import ir_master_update


def main():
    # Define update date
    update_date_str = datetime.datetime.today().strftime('%Y-%m-%d')
    update_date = np.datetime64(update_date_str)

    # Import configs
    configs = configparser.ConfigParser()
    configs.read('config.ini')

    # Set up logger
    log_dir = configs['COMMON']['LOG_DIR']
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    log_file = os.path.join(log_dir, 'data_extract_' + update_date_str + '.log')
    if os.path.isfile(log_file):
        os.remove(os.path.join(log_file))
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Starting data extract')

    start_time = time.time()

    # Determine business dates
    logger.info('Importing business dates at ' + str(time.time() - start_time) + ' seconds.')
    dates_file = configs['COMMON']['DATES']
    date_regions = configs['COMMON']['DATE_REGIONS'].split(',')
    date_regions = [r.replace(' ', '') for r in date_regions]
    golden_dates = business_dates(dates_file, date_regions)
    logger.info('Finished importing business dates at ' + str(time.time() - start_time) + ' seconds.')

    # Update FX data for new dates
    logger.info('Importing fx data at ' + str(time.time() - start_time) + ' seconds.')
    master_fx_file = configs['FX']['MASTER_FILE']
    major_ccys = configs['FX']['MAJOR_CCYS'].split(',')
    major_ccys = [r.replace(' ', '') for r in major_ccys]
    minor_ccys = configs['FX']['MINOR_CCYS'].split(',')
    minor_ccys = [r.replace(' ', '') for r in minor_ccys]
    fx_master_update(master_fx_file, update_date, golden_dates, major_ccys, minor_ccys)
    logger.info('Finished importing fx data at ' + str(time.time() - start_time) + ' seconds.')

    # Update equity price data for new dates
    logger.info('Importing equity data at ' + str(time.time() - start_time) + ' seconds.')
    master_equity_file = configs['EQ']['MASTER_FILE']
    tickers = pd.read_csv(configs['EQ']['TICKER_FILE'])['ticker']
    equity_master_update(master_equity_file, update_date, golden_dates, tickers)
    logger.info('Finished importing equity data at ' + str(time.time() - start_time) + ' seconds.')

    # Update rates for new dates
    logger.info('Importing rates data at ' + str(time.time() - start_time) + ' seconds.')
    master_equity_file = configs['IR']['MASTER_FILE']
    rates = pd.read_csv(configs['IR']['RATES_FILE'])
    ir_master_update(master_equity_file, update_date, golden_dates, rates)
    logger.info('Finished importing equity data at ' + str(time.time() - start_time) + ' seconds.')


if __name__ == "__main__":
    main()