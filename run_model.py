"""
Provides call_run_model function which is used to call a type of model.

"""
import configparser
import logging
import os
import time

import numpy as np
import pandas as pd
from scipy import stats

from data_extract.dates_extract import business_dates
from quant_functions.basic import levels_to_returns, string_to_none
from quant_functions.neural_network import sklearn_ann
from quant_functions.random_forest import random_forest_fun
from quant_functions.regression import regression_fun

logger = logging.getLogger('main')


def run_model_call(date):
    configs = configparser.ConfigParser()
    configs.read('model_config.ini')
    master_data_dir = configs['COMMON']['MASTER_DATA_DIR']
    master_model_dir = configs['COMMON']['MODEL_DIR']

    start_time = time.time()

    # Determine models to run
    with open(os.path.join(configs['COMMON']['MODEL_DIR'], 'control.txt')) as f:
        all_models = [line.rstrip() for line in f]

    model_predictions = pd.DataFrame()
    pos = 0
    for model_name in all_models:
        logger.info("Starting " + model_name + " run at " + str(time.time() - start_time) + " seconds")
        model_dir = os.path.join(master_model_dir, model_name)
        model_inputs = pd.read_csv(os.path.join(model_dir, 'inputs.csv'))
        model_outputs = os.path.join(model_dir, 'outputs')
        model_settings = pd.read_csv(os.path.join(model_dir, 'settings.csv'), index_col='field')
        log_file = os.path.join(model_outputs, 'logs', date + '.log')
        if os.path.isfile(log_file):
            os.remove(os.path.join(log_file))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

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
        current_levels = pd.DataFrame()
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
                current_levels.at[dt + '|' + f, 'current_value'] = hmd[-1]
                current_levels.at[dt + '|' + f, 'target_value'] = ret_f[-1]
        ret = ret.fillna(0)

        # Run selected model
        model_output_file = pd.read_csv(os.path.join(model_outputs, 'model_outputs.csv'), index_col='Date')
        model_type = str(model_settings.at['model_type', 'input'])
        features = []
        labels = []
        for i in range(len(model_inputs)):
            name = model_inputs['data_type'][i] + '|' + model_inputs['field'][i]
            if model_inputs['axis'][i] == 'feature':
                features.append(name)
            elif model_inputs['axis'][i] == 'label':
                labels.append(name)
        x = ret[features][:-1]
        for l in labels:
            logger.info("Starting " + model_name + " for " + l + " at " + str(time.time() - start_time) + " seconds")
            y = ret[l][1:]
            # Random forest
            test_size = float(model_settings.at['test_size', 'input'])
            y_pred = 0
            mean_test_error = 0
            mean_test_fit = 0
            if model_type == 'random_forest':
                rf, feature_importances, mean_test_error, mean_test_fit = random_forest_fun(x, y, test_size=test_size,
                                                                                            n_estimators=int(model_settings.at['n_estimators', 'input']),
                                                                                            criterion=model_settings.at['criterion', 'input'],
                                                                                            max_depth=string_to_none(model_settings.at['max_depth', 'input']),
                                                                                            min_samples_split=int(model_settings.at['min_samples_split', 'input']),
                                                                                            min_samples_leaf=int(model_settings.at['min_samples_leaf', 'input']),
                                                                                            min_weight_fraction_leaf=float(model_settings.at['min_weight_fraction_leaf', 'input']),
                                                                                            max_features=model_settings.at['max_features', 'input'],
                                                                                            max_leaf_nodes=string_to_none(model_settings.at['max_leaf_nodes', 'input']),
                                                                                            min_impurity_decrease=float(model_settings.at['min_impurity_decrease', 'input']),
                                                                                            min_impurity_split=string_to_none(model_settings.at['min_impurity_split', 'input']),
                                                                                            bootstrap=(model_settings.at['bootstrap', 'input'].lower() == 'true'),
                                                                                            oob_score=(model_settings.at['oob_score', 'input'].lower() == 'true'),
                                                                                            n_jobs=string_to_none(model_settings.at['n_jobs', 'input']),
                                                                                            random_state=string_to_none(model_settings.at['random_state', 'input']),
                                                                                            verbose=int(model_settings.at['verbose', 'input']),
                                                                                            warm_start=(model_settings.at['warm_start', 'input'].lower() == 'true'),
                                                                                            ccp_alpha=float(model_settings.at['ccp_alpha', 'input']),
                                                                                            max_samples=string_to_none(model_settings.at['max_samples', 'input']))
                y_pred = rf.predict(ret[features][-1:])
                model_output_file.at['pred:' + l, date] = y_pred
                model_output_file.at['mean_test_error:' + l, date] = mean_test_error
                model_output_file.at['mean_test_fit:' + l, date] = mean_test_fit
                for i in range(len(feature_importances)):
                    f = feature_importances[i][0]
                    model_output_file.at['importance:' + l + '_' + f, date] = feature_importances[i][1]

            # Regression
            elif model_type[:10] == 'regression':
                ret.to_csv(os.path.join(model_outputs, 'test.csv'))
                regression_type = model_type[11:]
                include_intercept = model_settings.at['include_intercept', 'input'].lower() == 'true'
                alpha = float(model_settings.at['alpha', 'input'])
                result, score, coef, mean_test_error, mean_test_fit, regressors = regression_fun(x, y, regression_type,
                                                                                                 include_intercept,
                                                                                                 alpha, test_size)
                y_pred = result.predict(ret[ret.columns[regressors]][-1:])
                model_output_file.at['pred:' + l, date] = y_pred
                model_output_file.at['mean_test_error:' + l, date] = mean_test_error
                model_output_file.at['mean_test_fit:' + l, date] = mean_test_fit
                model_output_file.at['r_squared:' + l, date] = score
                for i in range(len(regressors)):
                    guider = x.columns[regressors[i]]
                    model_output_file.at['beta:' + l + '_' + guider, date] = coef[i]

            # Artifical Neural Network
            elif model_type == 'ann':
                ann, success_rate, = sklearn_ann(x, y, 0.2, hidden_layer_sizes=(int(model_settings.at['hidden_layer_sizes', 'input']),),
                                                 activation=model_settings.at['activation', 'input'],
                                                 solver=model_settings.at['solver', 'input'],
                                                 alpha=float(model_settings.at['alpha', 'input']),
                                                 batch_size=model_settings.at['batch_size', 'input'],
                                                 learning_rate=model_settings.at['learning_rate', 'input'],
                                                 learning_rate_init=float(model_settings.at['learning_rate_init', 'input']),
                                                 power_t=float(model_settings.at['power_t', 'input']),
                                                 max_iter=int(model_settings.at['max_iter', 'input']),
                                                 shuffle=(model_settings.at['shuffle', 'input'].lower() == 'true'),
                                                 random_state=string_to_none(model_settings.at['random_state', 'input']),
                                                 tol=float(model_settings.at['tol', 'input']),
                                                 warm_start=(model_settings.at['warm_start', 'input'].lower() == 'true'),
                                                 momentum=float(model_settings.at['momentum', 'input']),
                                                 nesterovs_momentum=(model_settings.at['nesterovs_momentum', 'input'].lower() == 'true'),
                                                 early_stopping=(model_settings.at['early_stopping', 'input'].lower() == 'true'),
                                                 validation_fraction=float(model_settings.at['validation_fraction', 'input']),
                                                 beta_1=float(model_settings.at['beta_1', 'input']),
                                                 beta_2=float(model_settings.at['beta_2', 'input']),
                                                 epsilon=float(model_settings.at['epsilon', 'input']),
                                                 n_iter_no_change=int(model_settings.at['n_iter_no_change', 'input']),
                                                 max_fun=int(model_settings.at['max_fun', 'input']))
                y_pred = ann.predict(ret[features][-1:])
                if y_pred == 1:
                    y_pred = 'u'
                elif y_pred == 0:
                    y_pred = 'd'
                model_output_file.at['pred:' + l, date] = y_pred
                model_output_file.at['success_rate:' + l, date] = success_rate
                mean_test_fit = 'na'
                mean_test_error = 1 - success_rate

            else:
                logger.error(model_type + " is not a supported model type")
            y_mean = np.mean(y)
            y_std = np.std(y)
            test_n = int(len(ret) * test_size)
            model_predictions[pos] = [model_name, l, horizon, y_mean, y_std, y_pred[0], mean_test_fit, mean_test_error,
                                      test_n]
            pos += 1

        model_output_file.to_csv(os.path.join(model_outputs, 'model_outputs.csv'))
        model_settings_new = pd.read_csv(os.path.join(model_outputs, 'settings_archive.csv'), index_col='field')
        new_model_settings = np.append(model_settings['input'].values, [str(min(y.index)), str(max(y.index))])
        model_settings_new[date] = new_model_settings
        model_settings_new.to_csv(os.path.join(model_outputs, 'settings_archive.csv'))

    # Create model predictions and confidence distributions
    model_predictions = model_predictions.T
    model_predictions.columns = ['model', 'underlying', 'horizon_days', 'mean', 'std', 'prediction', 'test_fit',
                                 'test_error', 'test_n']
    ci = [0.01, 0.05, 0.1, 0.25, 0.35, 0.4, 0.45, 0.475, 0.5, 0.525, 0.55, 0.6, 0.65, 0.75, 0.9, 0.95, 0.99]
    for i in model_predictions.index:
        mean = model_predictions.at[i, 'prediction']
        stdev = model_predictions.at[i, 'test_error']
        n = model_predictions.at[i, 'test_n']
        current_level = current_levels.at[model_predictions.at[i, 'underlying'], 'current_value']
        model_predictions.at[i, 'current_level'] = current_level
        model_predictions.at[i, 't-1_return'] = current_levels.at[model_predictions.at[i, 'underlying'], 'target_value']
        if mean not in ['u', 'd']:
            for percentile in ci:
                percentile_pred = stats.t.ppf(percentile, n, loc=mean, scale=stdev)
                model_predictions.at[i, 'return:' + str(percentile)] = percentile_pred
            data_type, field = model_predictions.at[i, 'underlying'].split('|')
            return_form = model_inputs['functional_form'][(model_inputs['field'] == field) &
                                                          (model_inputs['data_type'] == data_type)].values[0]
            for percentile in ci:
                if return_form == 'absolute':
                    model_predictions.at[i, 'level:' + str(percentile)] = current_level + \
                                                                          model_predictions.at[
                                                                              i, 'return:' + str(percentile)]
                elif return_form == 'relative':
                    model_predictions.at[i, 'level:' + str(percentile)] = current_level * \
                                                                          (1 + model_predictions.at[
                                                                              i, 'return:' + str(percentile)])
                elif return_form == 'log':
                    model_predictions.at[i, 'level:' + str(percentile)] = current_level * np.exp(
                        model_predictions.at[i, 'return:' + str(percentile)])
    model_predictions = model_predictions.sort_values(by=['underlying'])
    model_predictions.to_csv(os.path.join(master_model_dir, 'predictions', 'model_predictions_' + str(date)[:10] + '.csv'))

    logger.info("Done at " + str(time.time() - start_time) + " seconds")
