"""
Entry point to all functions.
"""
import argparse
import logging

from run_data_extract import run_data_extract_call, run_data_extract_drop_call
from run_model import run_model_call

data_processes = 'extract', 'extract_drop', 'model'
data_types = 'all', 'equity', 'ir', 'fx'


def main():
    # Set up logger
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Fermorian data processors.')
    parser.add_argument('date', type=str, help='Date to run analysis for')
    parser.add_argument('process', type=str, choices=data_processes, help='Type of data analysis to run')
    parser.add_argument('--data_type', type=str, required=False, choices=data_types, help='Risk factor to run for')
    parser.add_argument('--nr_dates', type=int, required=False, help='Dates to drop')
    args = parser.parse_args()
    if args.process == 'extract':
        run_data_extract_call(args.date, args.data_type)
    elif args.process == 'extract_drop':
        run_data_extract_drop_call(args.date, args.nr_dates, args.data_type, data_types)
    elif args.process == 'model':
        run_model_call(args.date)


if __name__ == "__main__":
    main()