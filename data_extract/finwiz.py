# For data manipulation
import pandas as pd

# To extract fundamental data
from bs4 import BeautifulSoup as bs
import requests


# functions to get and parse data from FinViz
def fundamental_metric(soup, metric):
    return soup.find(text = metric).find_next(class_='snapshot-td2').text


def get_fundamental_data(df):
    for symbol in df.index:
        try:
            url = ("http://finviz.com/quote.ashx?t=" + symbol.lower())
            soup = bs(requests.get(url).content)
            print(soup)
            for m in df.columns:
                df.loc[symbol,m] = fundamental_metric(soup,m)
        except Exception as e:
            print (symbol, 'not found')
    return df

stock_list = ['AMZN','GOOG','PG','KO','IBM','DG','XOM','KO','PEP','MT','NL','GSB','LPL']
metric = ['P/B',
          'P/E',
          'Forward P/E',
          'PEG',
          'Debt/Eq',
          'EPS (ttm)',
          'Dividend %',
          'ROE',
          'ROI',
          'EPS Q/Q',
          'Insider Own']
df = pd.DataFrame(index=stock_list, columns=metric)
df = get_fundamental_data(df)
print(df)