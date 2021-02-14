import os

import pandas as pd
import yfinance as yf

# TICKERS = ['^FTSE', '^GSPC', '^STOXX50E', '^N225']
TICKERS = ['ULVR.L','AZN.L','HSBA.L','RIO.L','DGE.L','GSK.L','BATS.L','RDSA.L','BP.L','RDSA.L','RB.L','LSEG.L','BHP.L','VOD.L','REL.L','GLEN.L','AAL.L','PRU.L','NG.L','LLOY.L','BARC.L','CPG.L','EXPN.L','CRH.L','FLTR.L','TSCO.L','OCDO.L','FERG.L','NWG.L','SMT.L','ABF.L','AHT.L','SSE.L','BA.L','LGEN.L','ANTO.L','STAN.L','SN.L','AV.L','IMB.L','BT-A.L','JET.L','III.L','SGRO.L','AVV.L','NXT.L','WPP.L','RTO.L','HLMA.L','SKG.L','ITRK.L','IHG.L','CRDA.L','ADM.L','PSN.L','MNDI.L','MRO.L','SPX.L','JD.L','SDR.L','CCH.L','BNZL.L','INF.L','RR.L','HL.L','IAG.L','POLY.L','FRES.L','ENT.L','EVR.L','RSA.L','BDEV.L','PHNX.L','BRBY.L','SLA.L','SGE.L','STJ.L','WTB.L','UU.L','KGF.L','SMIN.L','JMAT.L','DCC.L','TW.L','BME.L','AUTO.L','PSON.L','SVT.L','BKG.L','RMV.L','HIK.L','ICP.L','SBRY.L','SMDS.L','AVST.L','MNG.L','PSH.L','LAND.L','MRW.L','BLND.L','PNN.L']

print("Extracting data")
data = yf.download(TICKERS, start="2021-01-01", end="2021-01-30")
data.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity',  'yfinance', 'data.csv'))

    # get stock info
    # pd.DataFrame(data=name.info).to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', t + '_info.csv'))
    # print(name.options)

    # get historical market data
    # hist = name.history(period="5bd")
    # hist.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', t + '_hmd.csv'))

    # show financials
    # name.financials.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', t + '_financials.csv'))
    # name.quarterly_financials.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', t + '_quarterly_financials.csv'))

    # # show major holders
    # msft.major_holders.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_majority_holders.csv'))
    #
    # # show institutional holders
    # msft.institutional_holders.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_institutional_holders.csv'))
    #
    # # show balance sheet
    # msft.balance_sheet.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_balance_sheet.csv'))
    # msft.quarterly_balance_sheet.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_balance_sheet.csv'))
    #
    # # show cashflow
    # msft.cashflow.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_cashflow.csv'))
    # msft.quarterly_cashflow.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_cashflow.csv'))
    #
    # # show earnings
    # msft.earnings.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_earnings.csv'))
    # msft.quarterly_earnings.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_quarterly_earnings.csv'))
    #
    # # show sustainability
    # msft.sustainability.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_sustainability.csv'))
    #
    # # show analysts recommendations
    # msft.recommendations.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_recommendations.csv'))
    #
    # # show next event (earnings, etc)
    # msft.calendar.to_csv(os.path.join('C:/', 'LocalFolder', 'temp', 'equity', 'msft_calendar.csv'))
