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
    positions_dir = os.path.join(master_model_dir, 'positions')
    settings = pd.read_csv(os.path.join(positions_dir, 'settings.csv'), index_col='field')

    log_file = os.path.join(prediction_dir, 'logs', date + '_position_weights.log')
    if os.path.isfile(log_file):
        os.remove(os.path.join(log_file))
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Determine positions
    positions = pd.DataFrame(columns=('position', 'data_type', 'field'))
    predictions = pd.read_csv(os.path.join(prediction_dir, 'model_predictions_' + date + '.csv'))
    underlyings = set(predictions['underlying'])
    vote_majority = float(settings.at['vote_majority', 'input'])
    row = 0
    for rf in underlyings:
        pred_rf = predictions[predictions['underlying'] == rf]
        data_type, field = rf.split('|')
        positions.at[row, 'data_type'] = data_type
        positions.at[row, 'field'] = field
        nr_models = len(pred_rf)
        up_votes = 0
        down_votes = 0
        for pred_rf_m in pred_rf['prediction']:
            if pred_rf_m in ['u', 'd']:
                pred = float(pred_rf_m.replace('u', '1').replace('d', '-1'))
            else:
                pred = float(pred_rf_m)
            if pred > 0:
                up_votes += 1
            elif pred < 0:
                down_votes += 1
        score = 0
        scaler = 1
        if up_votes / nr_models >= vote_majority:
            positions.at[row, 'position'] = 'long'
            for i in range(nr_models):
                pred = pred_rf['prediction'].values[i]
                error = float(pred_rf['test_error'].values[i])
                if pred == 'u':
                    scaler = (1 - error) / 0.5
                elif pred == 'd':
                    scaler = 0
                elif float(pred) > 0.0:
                    score += float(pred) / float(error)
            score = score * scaler
        elif down_votes / nr_models >= vote_majority:
            positions.at[row, 'position'] = 'short'
            for i in range(nr_models):
                pred = pred_rf['prediction'].values[i]
                error = float(pred_rf['test_error'].values[i])
                if pred == 'd':
                    scaler = (1 - error) / 0.5
                elif pred == 'u':
                    scaler = 0
                elif float(pred) < 0.0:
                    score += -float(pred) / error
            score = score * scaler
        positions.at[row, 'score'] = score
        row += 1
    max_pos = int(settings.at['max_pos', 'input'])
    positions = positions.sort_values(by=['score'], ascending=False)[:max_pos]

    # Import settings for data
    model_name = predictions['model'][0]
    model_inputs = pd.read_csv(os.path.join(master_model_dir, model_name, 'inputs.csv'), index_col=0)
    model_settings = pd.read_csv(os.path.join(master_model_dir, model_name, 'settings.csv'), index_col=0)

    # Determine business dates from dates calendar specified in the input
    logger.info('Importing model business dates at ' + str(time.time() - start_time) + ' seconds.')
    date_regions = model_settings.at['calendar', 'input'].split(',')
    date_regions = [r.replace(' ', '') for r in date_regions]
    dates_file = os.path.join(configs['COMMON']['MASTER_DATA_DIR'], 'business_dates.csv')
    golden_dates = business_dates(dates_file, date_regions)
    logger.info('Finished importing business dates at ' + str(time.time() - start_time) + ' seconds.')

    # Translate data into input for calculation
    golden_dates = [str(d)[:10] for d in golden_dates]
    horizon = int(model_settings.at['horizon', 'input'])
    historical_days = int(model_settings.at['historical_days', 'input'])
    overlapping = model_settings.at['overlapping', 'input'].lower() == 'true'

    # Import data and convert into returns
    ret = pd.DataFrame()
    input_data_types = set(positions['data_type'])
    for dt in input_data_types:
        master_data = pd.read_csv(os.path.join(master_data_dir, 'master_' + dt + '.csv'), index_col='Date')
        master_data = master_data[master_data.index.isin(golden_dates)][-historical_days:]
        fields = positions['field'][positions['data_type'] == dt]
        for f in fields:
            hmd = master_data[f]
            position_dir = str(positions['position'][positions['field'] == f].values[0])
            functional_form = str(model_inputs['functional_form'][model_inputs['field'] == f].values[0])
            data_fill = str(model_inputs['data_fill'][model_inputs['field'] == f].values[0])
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
    positions.to_csv(os.path.join(positions_dir, 'positions_' + str(date)[:10] + '.csv'))
    portfolio_weights.index.name = 'return'
    portfolio_weights.to_csv(os.path.join(positions_dir, 'markov_weights_' + str(date)[:10] + '.csv'))
    logger.info("Done at " + str(time.time() - start_time) + " seconds")
