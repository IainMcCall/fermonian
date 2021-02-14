"""
Provides call_run_model function which is used to call a type of model.

"""
import configparser
import logging
import os
import time

import pandas as pd

from data_extract.dates_extract import business_dates
from quant_functions.basic import levels_to_returns

logger = logging.getLogger('main')


def run_backtest_call(date):
    configs = configparser.ConfigParser()
    configs.read('model_config.ini')
    master_data_dir = configs['COMMON']['MASTER_DATA_DIR']

    start_time = time.time()

    # Determine models to run
    with open(os.path.join(configs['COMMON']['MODEL_DIR'], 'control.txt')) as f:
        all_models = [line.rstrip() for line in f]

    for model_name in all_models:
        logger.info("Starting " + model_name + " backtest at " + str(time.time() - start_time) + " seconds")
        model_dir = os.path.join(configs['COMMON']['MODEL_DIR'], model_name)
        model_inputs = pd.read_csv(os.path.join(model_dir, 'inputs.csv'))
        model_outputs = os.path.join(model_dir, 'outputs')
        model_backtest_output = os.path.join(model_dir, 'backtests')
        model_settings = pd.read_csv(os.path.join(model_dir, 'settings.csv'), index_col='field')
        log_file = os.path.join(model_backtest_output, 'logs', date + '_backtest.log')
        if os.path.isfile(log_file):
            os.remove(os.path.join(log_file))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.info('Starting ' + model_name + '  back-test')

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
        input_data_types = set(model_inputs['data_type'])
        for dt in input_data_types:
            master_data = pd.read_csv(os.path.join(master_data_dir, 'master_' + dt + '.csv'), index_col='Date')
            master_data = master_data[master_data.index.isin(golden_dates)][-historical_days:]
            fields = model_inputs['field'][model_inputs['data_type'] == dt]
            for f in fields:
                hmd = master_data[f]
                functional_form = str(model_inputs['functional_form'][model_inputs['field'] == f].values[0])
                data_fill = str(model_inputs['data_fill'][model_inputs['field'] == f].values[0])
                ret_f = levels_to_returns(hmd, functional_form, horizon, overlapping, data_fill)
                ret[dt + '|' + f] = ret_f
        ret.index = hmd.index.values[horizon:]
        ret = ret.fillna(0)

        # Run selected model
        model_output_file = pd.read_csv(os.path.join(model_outputs, 'model_outputs.csv'), index_col='Date')
        backtest_results = pd.DataFrame()
        for i in range(len(model_inputs)):
            if model_inputs['axis'][i] == 'label':
                label = model_inputs['field'][i]
                label_conc = model_inputs['data_type'][i] + '|' + label
                logger.info('Beginning backtesting for ' + label_conc)
                for d in model_output_file.columns:
                    actual = ret.at[d, label_conc]
                    pred = model_output_file.at['pred:' + label_conc, d]
                    backtest_results.at['actual:' + label_conc, d] = actual
                    backtest_results.at['model:' + label_conc, d] = pred
                    backtest_results.at['error:' + label_conc, d] = pred - actual
        backtest_results.to_csv(os.path.join(model_backtest_output, 'model_backtesting.csv'))
