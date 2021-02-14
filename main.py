"""
Entry point to all functions.
"""
import argparse
import logging

from run_data_extract import run_data_extract_call, run_data_extract_drop_call
from run_model import run_model_call
from run_backtest import run_backtest_call
from run_portfolio_weights import portfolio_weights_call

data_processes = 'extract', 'drop', 'model', 'backtest', 'weights'
data_types = 'all', 'equity', 'ir', 'fx', 'economics'


def main():
    # Set up logger
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Fermorian data processors.')
    parser.add_argument('date', type=str, help='Date to run analysis for')
    parser.add_argument('process', type=str, choices=data_processes, help='Type of data analysis to run')
    parser.add_argument('--data_type', type=str, required=False, default='all', choices=data_types,
                        help='Risk factor to run for')
    parser.add_argument('--points', type=int, required=False, default=1, help='Dates to drop')
    args = parser.parse_args()
    if args.process == 'extract':
        run_data_extract_call(args.date, args.data_type)
    elif args.process == 'drop':
        run_data_extract_drop_call(args.date, args.points, args.data_type, data_types)
    elif args.process == 'model':
        run_model_call(args.date)
    elif args.process == 'backtest':
        run_backtest_call(args.date)
    elif args.process == 'weights':
        portfolio_weights_call(args.date)


if __name__ == "__main__":
    main()