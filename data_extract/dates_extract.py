import os

import pandas as pd


def business_dates(dates_file, regions):
    f = pd.read_csv(dates_file)
    all_dates = []
    for r in regions:
        all_dates.extend(pd.to_datetime(f[r]).dropna().values)
    all_dates = sorted(set(all_dates))
    return all_dates


def filter_update_dates(all_dates, update_date, last_update_date):
    new_dates = []
    for d in all_dates:
        if update_date > d > last_update_date:
            new_dates.append(d)
    return new_dates
