"""
Calls the functions to update the master databases.

"""
import configparser
import logging
import os
import time

import numpy as np
import pandas as pd

# import main
from data_extract.dates_extract import business_dates
from data_extract.extract_fx_hmd import fx_master_update
from data_extract.extract_equity_hmd import equity_master_update
from data_extract.extract_ir_hmd import ir_master_update

logger = logging.getLogger('main')


def run_data_extract_call(date, data_type):
    update_date = np.datetime64(date)
    configs = configparser.ConfigParser()
    configs.read('config.ini')

    log_dir = configs['COMMON']['LOG_DIR']
    log_file = os.path.join(log_dir, 'data_extract_' + date + '.log')
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

    if data_type in ['all', 'fx']:
        logger.info('Importing fx data at ' + str(time.time() - start_time) + ' seconds.')
        master_fx_file = configs['FX']['MASTER_FILE']
        major_ccys = configs['FX']['MAJOR_CCYS'].split(',')
        major_ccys = [r.replace(' ', '') for r in major_ccys]
        minor_ccys = configs['FX']['MINOR_CCYS'].split(',')
        minor_ccys = [r.replace(' ', '') for r in minor_ccys]
        fx_master_update(master_fx_file, update_date, golden_dates, major_ccys, minor_ccys)
        logger.info('Finished importing fx data at ' + str(time.time() - start_time) + ' seconds.')

    if data_type in ['all', 'equity']:
        logger.info('Importing equity data at ' + str(time.time() - start_time) + ' seconds.')
        master_equity_file = configs['EQUITY']['MASTER_FILE']
        tickers = pd.read_csv(configs['EQUITY']['TICKER_FILE'])['ticker']
        equity_master_update(master_equity_file, update_date, golden_dates, tickers)
        logger.info('Finished importing equity data at ' + str(time.time() - start_time) + ' seconds.')

    if data_type in ['all', 'ir']:
        logger.info('Importing rates data at ' + str(time.time() - start_time) + ' seconds.')
        master_equity_file = configs['IR']['MASTER_FILE']
        rates = pd.read_csv(configs['IR']['RATES_FILE'])
        ir_master_update(master_equity_file, update_date, golden_dates, rates)
        logger.info('Finished importing ir data at ' + str(time.time() - start_time) + ' seconds.')


def run_data_extract_drop_call(date, days, data_type, all_data_types):
    configs = configparser.ConfigParser()
    configs.read('config.ini')
    log_dir = configs['COMMON']['LOG_DIR']
    log_file = os.path.join(log_dir, 'data_extract_drop_' + date + '.log')
    if os.path.isfile(log_file):
        os.remove(os.path.join(log_file))
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Starting data extract drop')

    start_time = time.time()

    if data_type == 'all':
        data_types = all_data_types
    else:
        data_types = [data_type]
    for d in data_types:
        if d != 'all':
            logger.info('Starting ' + d + ' data drop for ' + str(days) + ' days at ' + str(time.time() - start_time) +
                        ' seconds.')
            master_file = configs[d.upper()]['MASTER_FILE']
            master = pd.read_csv(master_file, index_col='Date', parse_dates=True)
            master_new = master[:-days]
            master_new.to_csv(master_file)
            logger.info('Finished ' + d + ' data drop at ' + str(time.time() - start_time) + ' seconds.')
