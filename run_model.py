"""
Provides call_run_model function which is used to call a type of model.

"""
import configparser
import logging
import os
import time

import pandas as pd

from data_extract.dates_extract import business_dates
from quant_functions.random_forest import random_forest_fun
from quant_functions.basic import levels_to_returns

logger = logging.getLogger('main')


def run_model_call(date):
    configs = configparser.ConfigParser()
    configs.read('model_config.ini')
    master_data_dir = configs['COMMON']['MASTER_DATA_DIR']
    model_name = configs['COMMON']['MODEL_NAME']
    model_dir = configs['COMMON']['MODEL_DIR']
    model_inputs = pd.read_csv(os.path.join(model_dir, 'inputs.csv'))
    model_outputs = os.path.join(model_dir, 'outputs')
    model_settings = pd.read_csv(os.path.join(model_dir, 'settings.csv'), index_col='field')
    log_file = os.path.join(model_outputs, date + '.log')
    if os.path.isfile(log_file):
        os.remove(os.path.join(log_file))
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info('Starting ' + model_name + ' run')

    start_time = time.time()

    # Determine business dates from dates calendar specified in the input
    logger.info('Importing model business dates at ' + str(time.time() - start_time) + ' seconds.')
    date_regions = model_settings.at['calendar', 'input'].split(',')
    date_regions = [r.replace(' ', '') for r in date_regions]
    dates_file = os.path.join(configs['COMMON']['MASTER_DATA_DIR'], 'business_dates.csv')
    golden_dates = business_dates(dates_file, date_regions)
    logger.info('Finished importing business dates at ' + str(time.time() - start_time) + ' seconds.')

    # Import data
    hmd = pd.DataFrame()
    input_data_types = set(model_inputs['data_type'])
    for dt in input_data_types:
        master_data = pd.read_csv(os.path.join(master_data_dir, 'master_' + dt + '.csv'), index_col='Date')
        fields = model_inputs['field'][model_inputs['data_type'] == dt]
        for f in fields:
            hmd[dt + '_' + f] = master_data[f]

    # Translate data into input for calculation
    golden_dates = [str(d)[:10] for d in golden_dates]
    horizon = int(model_settings.at['horizon', 'input'])
    historical_days = int(model_settings.at['historical_days', 'input'])
    hmd = hmd[hmd.index.isin(golden_dates)][-historical_days:]
    functional_form = str(model_settings.at['functional_form', 'input'])
    ret = levels_to_returns(hmd, functional_form, horizon)
    ret = ret.fillna(0)

    # Run selected model
    model_output_file = pd.read_csv(os.path.join(model_outputs, 'model_outputs.csv'), index_col='Date')
    model_type = str(model_settings.at['model_type', 'input'])
    features = []
    labels = []
    for i in range(len(model_inputs)):
        name = model_inputs['data_type'][i] + '_' + model_inputs['field'][i]
        if model_inputs['axis'][i] == 'feature':
            features.append(name)
        elif model_inputs['axis'][i] == 'label':
            labels.append(name)
    x = ret[features][:-1]
    for l in labels:
        logger.info('Beginning model analysis for ' + l)
        y = ret[l][1:]
        if model_type == 'random_forest':
            nr_estimators = int(model_settings.at['estimators', 'input'])
            test_size = float(model_settings.at['test_size', 'input'])
            rf, feature_importances, mean_error, stdev, stdev_pred = random_forest_fun(x, y, estimators=nr_estimators,
                                                                                       test_size=test_size)
            y_pred = rf.predict(ret[features][-1:])
            model_output_file.at[date, 'pred:' + l] = y_pred
            model_output_file.at[date, 'mean_error:' + l] = mean_error
            model_output_file.at[date, 'target_stdev:' + l] = stdev
            model_output_file.at[date, 'pred_stdev:' + l] = stdev_pred
            for i in range(len(feature_importances)):
                f = feature_importances[i][0]
                model_output_file.at[date, 'importance:' + l + '_' + f] = feature_importances[i][1]
        else:
            logger.error(model_type + ' is not a supported model type')
        model_output_file.to_csv(os.path.join(model_outputs, 'model_outputs.csv'))
