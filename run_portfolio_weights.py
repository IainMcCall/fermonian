"""
Provides function to call when creating portfolio weights.

"""
import configparser
import logging
import os
import time

import numpy as np
import pandas as pd

from data_extract.dates_extract import business_dates
from quant_functions.basic import levels_to_returns
from quant_functions.portfolio_theory import markov_weights

logger = logging.getLogger('main')


def portfolio_weights_call(date):
    """
    Function creates markov minimum variance portfolios.

    Args:
        date (str): yyyy-mm-dd

    Outputs:

    """
    configs = configparser.ConfigParser()
    configs.read('model_config.ini')
    master_data_dir = configs['COMMON']['MASTER_DATA_DIR']
    master_model_dir = configs['COMMON']['MODEL_DIR']

    start_time = time.time()

    logger.info("Starting portfolio weights calc at " + str(time.time() - start_time) + " seconds")
    prediction_dir = os.path.join(master_model_dir, 'predictions')
    positions = pd.read_csv(os.path.join(prediction_dir, 'positions_' + date + '.csv'))
    settings = pd.read_csv(os.path.join(prediction_dir, 'settings.csv'), index_col='field')
    log_file = os.path.join(prediction_dir, 'logs', date + '_position_weights.log')
    if os.path.isfile(log_file):
        os.remove(os.path.join(log_file))
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Determine business dates from dates calendar specified in the input
    logger.info('Importing model business dates at ' + str(time.time() - start_time) + ' seconds.')
    date_regions = settings.at['calendar', 'input'].split(',')
    date_regions = [r.replace(' ', '') for r in date_regions]
    dates_file = os.path.join(configs['COMMON']['MASTER_DATA_DIR'], 'business_dates.csv')
    golden_dates = business_dates(dates_file, date_regions)
    logger.info('Finished importing business dates at ' + str(time.time() - start_time) + ' seconds.')

    # Translate data into input for calculation
    golden_dates = [str(d)[:10] for d in golden_dates]
    horizon = int(settings.at['horizon', 'input'])
    historical_days = int(settings.at['historical_days', 'input'])
    overlapping = settings.at['overlapping', 'input'].lower() == 'true'

    # Import data and ceonvrt into returns
    ret = pd.DataFrame()
    input_data_types = set(positions['data_type'])
    for dt in input_data_types:
        master_data = pd.read_csv(os.path.join(master_data_dir, 'master_' + dt + '.csv'), index_col='Date')
        master_data = master_data[master_data.index.isin(golden_dates)][-historical_days:]
        fields = positions['field'][positions['data_type'] == dt]
        for f in fields:
            hmd = master_data[f]
            position_dir = str(positions['position'][positions['field'] == f].values[0])
            functional_form = str(positions['functional_form'][positions['field'] == f].values[0])
            data_fill = str(positions['data_fill'][positions['field'] == f].values[0])
            ret_f = levels_to_returns(hmd, functional_form, horizon, overlapping, data_fill)
            if position_dir == 'short':
                ret_f =[r * -1 for r in ret_f]
            ret[dt + '|' + f] = ret_f
    ret = ret.fillna(0)

    # Set initial target as equally weighted portfolio
    portfolio_weights = pd.DataFrame()
    avg_returns = [np.average(ret[x]) for x in ret]
    for target in [min(avg_returns) + i * (max(avg_returns) - min(avg_returns)) / 10 for i in range(1, 10)]:
        calibration = markov_weights(ret, target)
        calibrated_weights = calibration.x
        portfolio_weights.at[target, 'stev'] = calibration.fun
        for i in range(len(ret.columns)):
            portfolio_weights.at[target, ret.columns[i]] = calibrated_weights[i]
    portfolio_weights.to_csv(os.path.join(prediction_dir, 'markov_weights_' + str(date)[:10] + '.csv'))
    logger.info("Done at " + str(time.time() - start_time) + " seconds")
