import os

import pandas as pd
import json
from urllib.request import urlopen

api_key ='6e2de46c7d2c1009e17af267e669148f'
url = "https://financialmodelingprep.com/api/v3/" + "company-key-metrics/" + 'OCDO.L' + '?apikey=' + api_key

codes = {'quarterly_income_statement': ['income-statement/', '?period=quarter&limit=400'],
         'annnual_income_statement': ['income-statement/', '?limit=120'],
         'annnual_income_statement_as_reported': ['income-statement-as-reported/', '?limit=10'],
         'quarterly_balance_sheet': ['balance-sheet-statement/', '?period=quarter&limit=400'],
         'annnual_balance_sheet': ['balance-sheet-statement/', '?limit=120'],
         'quarterly_balance_sheet_as_reported': ['balance-sheet-statement-as-reported/', '?period=quarter&limit=50'],
         'annnual_balance_sheet_as_reported': ['balance-sheet-statement-as-reported/', '?limit=10'],
         'quarterly_cash_flow': ['cash-flow-statement/', '?period=quarter&limit=400'],
         'annual_cash_flow': ['cash-flow-statement/', '?&limit=120'],
         'quarterly_cash_flow_as_reported': ['cash-flow-statement-as-reported/', '?period=quarter&limit=50'],
         'annual_cash_flow_as_reported': ['cash-flow-statement-as-reported/', '?limit=10'],
         'lse_income': ['cash-flow-statement/', '?limit=100']}
codes = {'key_metrics': ['key-metrics/', '?period=quarter&limit=130']}
tickers = ['BARC.L', 'AMZN', 'AAPL']

# Extract Equity
# for c in codes:
#     for t in tickers:
#         print('Attempting ' + c + ': ' + t)
#         url = "https://financialmodelingprep.com/api/v3/" +  codes[c][0] + t + codes[c][1] + '&apikey=' + api_key
#         response = urlopen(url)
#         data = response.read().decode("utf-8")
#         data = json.loads(data)
#         all_data = pd.DataFrame()
#         i = 0
#         for df in data:
#             print(data)
#             for f in df:
#                 all_data.at[f, i] = df[f]
#             i += 1
#         all_data.to_csv(os.path.join(r'C:/', 'LocalFolder', 'temp', 'fmp', t + '_' + c + '.csv'))


# Extract FX
print('Attempting')
url = "https://financialmodelingprep.com/api/v3/historical-price-full/USDMXN?apikey=" + api_key
response = urlopen(url)
data = response.read().decode("utf-8")
data = json.loads(data)
all_data = pd.DataFrame()
i = 0
for df in data:
    print(data)
    for f in df:
        all_data.at[f, i] = df[f]
    i += 1
all_data.to_csv(os.path.join(r'C:/', 'LocalFolder', 'temp', 'fx.csv'))


# Extract company list
# url = "https://financialmodelingprep.com/api/v3/" + "stock/list?apikey=" + api_key
# response = urlopen(url)
# data = response.read().decode("utf-8")
# data = json.loads(data)
# all_data = pd.DataFrame()
# i = 0
# for df in data:
#     for f in ['symbol', 'name', 'price', 'exchange']:
#         try:
#             all_data.at[i, f] = df[f]
#             print(df['name'] + f + ' extracted')
#         except:
#             print(':( No data available for ' + f)
#     i += 1
# all_data.to_csv(os.path.join(r'C:/', 'LocalFolder', 'temp', 'fmp', 'stock_list.csv'))

# Extract financial statement list
# url = "https://financialmodelingprep.com/api/v3/" + "financial-statement-symbol-lists?apikey=" + api_key
# response = urlopen(url)
# data = response.read().decode("utf-8")
# data = json.loads(data)
# all_data = pd.DataFrame()
# i = 0
# for ticker in data:
#     all_data.at[i, 'ticker'] = ticker
#     i += 1
# all_data.to_csv(os.path.join(r'C:/', 'LocalFolder', 'temp', 'fmp', 'stock_financials_list.csv'))