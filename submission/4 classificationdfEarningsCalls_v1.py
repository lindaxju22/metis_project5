#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:56:44 2019

@author: lindaxju
"""

#%%
import pickle

import pandas as pd

from datetime import datetime, timedelta
from pandas_datareader import data as wb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from collections import defaultdict

from datetime import date
#%%
###############################################################################
##############################Helper Functions#################################
###############################################################################
#%%
# can take out
def print_dict_keys(dict_dict):
    
    """
    Input is a dictionary of dictionaries
    Prints keys of the outer and inner dictionaries
    """
    
    for key1, val1 in dict_dict.items():
        print(key1)
        print('')
#%%
#############################Get Adj Close Prices##############################
#%%
def get_company_prices_clean_df(ticker,start_date,end_date):
            
    """
    Input is ticker, start, and end date
    Returns dataframe of dates, volumes, and adjusted close prices with all dates filled in
    """
    
    data_source = 'yahoo'
    
    ticker_data = wb.DataReader(ticker,start=start_date,end=end_date,data_source=data_source)
    company_prices_raw_df = pd.DataFrame(ticker_data)
    
    # fill in missing dates
    date = []
    volume = []
    adj_close = []
    
    current_date = company_prices_raw_df.index[0]
    prev_adj_close = company_prices_raw_df['Adj Close'][0]
    
    while current_date != max(company_prices_raw_df.index)+timedelta(1):
        date.append(current_date)
        if current_date in company_prices_raw_df.index:
            volume.append(company_prices_raw_df.loc[current_date]['Volume']/1000000)
            prev_adj_close = company_prices_raw_df.loc[current_date]['Adj Close']
            adj_close.append(prev_adj_close)
        else:
            volume.append(0)
            adj_close.append(prev_adj_close)
        current_date += timedelta(1)
    
    company_prices_clean_df = pd.DataFrame(list(zip(volume,adj_close)),index=date,columns =['volume','adj_close']) 

    return company_prices_clean_df
#%%
def plot_company_prices(transcripts_df,prices_df,timerangedelta):
    fig,ax = plt.subplots()

    ax.plot(list(prices_df.index),prices_df.adj_close)
    
    for transcript_date in transcripts_df.date:
        ax.axvspan(date2num(transcript_date),date2num(transcript_date+timerangedelta),color="gray",alpha=0.3)
    
    fig.autofmt_xdate()
    fig.set_size_inches(20,5)
    plt.show()
    fig.savefig('company_prices_adjclose.svg')
#%%
##################################Get Returns##################################
#%%
def get_returns_df(transcripts_df,prices_df,timerangedelta,return_thres=3):
                
    """
    Input is clean transcripts, clean prices, time delta, and return threshold
    Returns dataframe with label/target information
    """
    
    adj_close_dayof = []
    adj_close_delta = []

    for transcript_date in transcripts_df.date:
        adj_close_dayof.append(prices_df.adj_close.loc[transcript_date])
        adj_close_delta.append(prices_df.adj_close.loc[transcript_date+timerangedelta])
    
    returns = [(i/j-1)*100 for i,j in zip(adj_close_delta,adj_close_dayof)]
    buy_signal = []
    for ret in returns:
        if ret <= -return_thres:
            buy_signal.append(-1)
        elif ret >= return_thres:
            buy_signal.append(1)
        else:
            buy_signal.append(0)
    
    transcripts_df['adj_close_dayof'] = adj_close_dayof
    transcripts_df['adj_close_delta'] = adj_close_delta
    transcripts_df['returns'] = returns
    transcripts_df['buy_signal'] = buy_signal
    
    return transcripts_df
#%%
###############################################################################
########################Prepare Features and Target############################
###############################################################################
#%%
##########################Get Features and Target##############################
#%%
filename = 'data/' + 'company_sentiment_ordered_df_2019-12-07-16-05-18.pickle'
with open(filename,'rb') as read_file:
    company_sentiment_ordered_df = pickle.load(read_file)
#%%
filename = 'data/' + 'company_topics_ordered_df_2019-12-07-16-03-19.pickle'
with open(filename,'rb') as read_file:
    company_topics_ordered_df = pickle.load(read_file)
#%%
print(company_sentiment_ordered_df.shape)
company_sentiment_ordered_df.columns
#%%
print(company_topics_ordered_df.shape)
company_topics_ordered_df.columns
#%%
cols_sentiment = ['date','transcript_name','content_orig','content_cleaned',
                  'per_negative','per_positive','per_uncertainty','per_litigious',
                  'per_constraining','per_interesting','per_modal1','per_modal2',
                  'per_modal3']
cols_topics = ['transcript_name','c01','c02', 'c03','c04','c05','c06','c07',
               'c08','c09','c10','c11','c12','c13','c14','c15','c16','c17',
               'c18','c19','c20']
#%%
classification_raw_df = pd.merge(company_sentiment_ordered_df[cols_sentiment],company_topics_ordered_df[cols_topics],on='transcript_name',how='left')

ticker = [] # add tickers to dataframe
for transcript_name in classification_raw_df.transcript_name:
    ticker.append(transcript_name[3])

classification_raw_df.insert(loc=1,column='ticker',value=ticker)

classification_raw_df.columns
#%%
# drop last rows because may not have target label
indices_drop = []
ticker_unique = list(classification_raw_df.ticker.unique())

for row in range(len(classification_raw_df)-1,-1,-1):
    if classification_raw_df.ticker[row] in ticker_unique:
        indices_drop.append(row)
        ticker_unique.remove(classification_raw_df.ticker[row])
        if len(ticker_unique) == 0:
            break
    
indices_drop
#%%
# drop last rows because may not have target label
classification_clean_df = classification_raw_df.copy()
classification_clean_df.drop(indices_drop,inplace=True)
classification_clean_df.tail()
#%%
# create a dictionary of company prices
ticker_unique = list(classification_raw_df.ticker.unique())
timerangedelta = timedelta(30)
end_date = datetime.now()-timedelta(1) # yesterday

company_prices_clean_dict = defaultdict()

for ticker in ticker_unique:
    print(ticker)
    classification_clean_ticker_df = classification_clean_df.copy()
    classification_clean_ticker_df = classification_clean_ticker_df[classification_clean_ticker_df['ticker'] == ticker]
    start_date = min(classification_clean_ticker_df.date)-timerangedelta
    company_prices_clean_ticker_df = get_company_prices_clean_df(ticker,start_date,end_date)
    ticker_key = (ticker,start_date,end_date)
    company_prices_clean_dict[ticker_key] = company_prices_clean_ticker_df
    
company_prices_clean_dict = dict(company_prices_clean_dict)
#%%
# test the dictionary
key_test = list(company_prices_clean_dict.keys())[0]
company_prices_clean_dict[key_test]
#%%
# create a dictionary of transcript returns
ticker_unique = list(classification_raw_df.ticker.unique())

classification_dict = defaultdict()

for ticker_key in list(company_prices_clean_dict.keys()):
    ticker = ticker_key[0]
    classification_clean_ticker_df = classification_clean_df.copy()
    classification_clean_ticker_df = classification_clean_ticker_df[classification_clean_ticker_df['ticker'] == ticker]
    company_prices_clean_ticker_df = company_prices_clean_dict[ticker_key]
    classification_ticker_df = get_returns_df(classification_clean_ticker_df,company_prices_clean_ticker_df,timerangedelta,return_thres=3)
    classification_dict[ticker_key] = classification_ticker_df
    
classification_dict = dict(classification_dict)
#%%
# test the dictionary
key_test = list(classification_dict.keys())[0]
classification_dict[key_test]
#%%
classification_columns = list(classification_dict[key_test].columns)
classification_df = pd.DataFrame(columns=classification_columns)

for ticker_key in list(classification_dict.keys()):
    classification_ticker_new_df = classification_dict[ticker_key].copy()
    classification_df = pd.concat([classification_df,classification_ticker_new_df])
classification_df.reset_index(drop=True,inplace=True)
classification_df
#%%
plt.hist(classification_df.returns, bins='auto')
#%%
classification_df = classification_df[classification_df.returns <= 40]
classification_df
#%%
plt.hist(classification_df.returns, bins='auto')
#%%
#%%
#%%
#%%
#from datetime import datetime
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'classification_df'+'_'+timestamp
#classification_df.to_csv(r'data/'+filename+'.csv')
#%%
#import sys
#sys.setrecursionlimit(100000)
#
#import pickle
#from datetime import datetime
#
#timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
#filename = 'classification_df'+'_'+timestamp
#with open('data/'+filename+'.pickle', 'wb') as to_write:
#    pickle.dump(classification_df, to_write)
#%%
###############################################################################
####################################End########################################
###############################################################################
#%%
###############################################################################
################################Presentation###################################
###############################################################################
#%%
ticker_macys = 'M'
classification_clean_pres_df = classification_df.copy()
classification_clean_pres_df = classification_clean_pres_df[classification_clean_pres_df.ticker == ticker_macys]
classification_clean_pres_df.reset_index(drop=True,inplace=True)
classification_clean_pres_df

key_macys = list(company_prices_clean_dict.keys())[3]
company_prices_clean_df = company_prices_clean_dict[key_macys]
company_prices_clean_df

classification_clean_pres_df = classification_clean_pres_df.copy()[len(classification_clean_pres_df)-5*4:]
classification_clean_pres_df

zoomed_in_df = company_prices_clean_df.copy().loc[classification_clean_pres_df.date.iloc[0]:]
zoomed_in_df
#%%
fontsize = 15
#%%
fig,ax1 = plt.subplots()
plt.rcParams.update({'font.size':fontsize})

ax1.margins(0)
ax1.set_title("Macy's Quarterly Earnings Releases" + " \nQ3 2014 - Q3 2019")

ax1.set_ylim(0,max(zoomed_in_df.adj_close)*1.1)

timerangedelta = timedelta(1)

for transcript_date in classification_clean_pres_df.date:
    ax1.axvspan(date2num(transcript_date-timerangedelta),date2num(transcript_date+timerangedelta),color="gray",alpha=0.3)

line1 = ax1.plot(list(zoomed_in_df.index),[0]*len(zoomed_in_df.index),color="gray",alpha=0.8,label='Timing of Earnings Releases',linewidth=3)

charts = line1
labels = [chart.get_label() for chart in charts]
ax1.legend(charts,labels,loc='upper right',fontsize=fontsize,frameon=True).get_frame().set_edgecolor('black')

fig.autofmt_xdate()
fig.set_size_inches(20,8)
plt.show()
fig.savefig('figures/earnings_releases.svg')
#%%
fig,ax1 = plt.subplots()
plt.rcParams.update({'font.size':fontsize})

ax1.margins(0)
ax1.set_title("Macy's Stock Price" + " \nQ3 2014 - Q3 2019")

timerangedelta = timedelta(1)

for transcript_date in classification_clean_pres_df.date:
    ax1.axvspan(date2num(transcript_date-timerangedelta),date2num(transcript_date+timerangedelta),color="gray",alpha=0.3)

line1 = ax1.plot(list(zoomed_in_df.index),[0]*len(zoomed_in_df.index),color="gray",alpha=0.8,label='Timing of Earnings Releases',linewidth=3)
stock_price = ax1.plot(list(zoomed_in_df.index),zoomed_in_df.adj_close,alpha=0.8,label='Stock Price',linewidth=3)
ax1.set_ylim(0,max(zoomed_in_df.adj_close)*1.1)
ax1.set_ylabel('Stock Price')

charts = line1+stock_price
labels = [chart.get_label() for chart in charts]
ax1.legend(charts,labels,loc='upper right',fontsize=fontsize,frameon=True).get_frame().set_edgecolor('black')

fig.autofmt_xdate()
fig.set_size_inches(20,8)
plt.show()
fig.savefig('figures/company_prices_adjclose.svg')
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
ticker = 'M'
start_date = min(classification_clean_df.date)-timedelta(7)
end_date = datetime.now()-timedelta(1)
data_source = 'yahoo'
#%%
ticker_data = wb.DataReader(ticker,start=start_date,end=end_date,data_source=data_source)
company_prices_raw_df = pd.DataFrame(ticker_data)
company_prices_raw_df.head()
#%%
# fill in missing dates
date = []
volume = []
adj_close = []

current_date = company_prices_raw_df.index[0]
prev_adj_close = company_prices_raw_df['Adj Close'][0]

while current_date != max(company_prices_raw_df.index)+timedelta(1):
    date.append(current_date)
    if current_date in company_prices_raw_df.index:
        volume.append(company_prices_raw_df.loc[current_date]['Volume'])
        prev_adj_close = company_prices_raw_df.loc[current_date]['Adj Close']
        adj_close.append(prev_adj_close)
    else:
        volume.append(0)
        adj_close.append(prev_adj_close)
    current_date += timedelta(1)
#%%
company_prices_clean_df = pd.DataFrame(list(zip(volume,adj_close)),index=date,columns =['volume','adj_close']) 
company_prices_clean_df.head()
#%%
fig,ax1 = plt.subplots()
plt.rcParams.update({'font.size': 12})

ax1.margins(0)
ax1.set_title("Macy's Trading Volume and Stock Price" + " \n2003 - 2019")

line1 = ax1.plot(list(company_prices_clean_df.index),company_prices_clean_df.adj_close,alpha=0.8,label='Stock Price')
ax1.set_ylabel('Stock Price')   
ax2 = ax1.twinx()
bar2 = ax2.plot(list(company_prices_clean_df.index),company_prices_clean_df.volume,color="orange",alpha=0.8,label='Trading Volume')
ax2.set_ylim(0, max(company_prices_clean_df.volume)*1.5)
ax2.set_ylabel('Trading Volume')

timerangedelta = timedelta(0)

for transcript_date in classification_clean_df.date:
    ax1.axvspan(date2num(transcript_date),date2num(transcript_date+timerangedelta),color="gray",alpha=0.3)

bar3 = ax1.plot(list(company_prices_clean_df.index),[0]*len(company_prices_clean_df.index),color="gray",alpha=0.8,label='Earnings Release')
charts = line1+bar2+bar3
labels = [chart.get_label() for chart in charts]
ax1.legend(charts,labels,loc='upper left',fontsize=11,frameon=True).get_frame().set_edgecolor('black')

fig.autofmt_xdate()
fig.set_size_inches(20,8)
plt.show()
#fig.savefig('company_prices_adjclose.svg')
#%%
adj_close_dayof = []
adj_close_delta = []
for transcript_date in classification_clean_df.date:
    adj_close_dayof.append(company_prices_clean_df.adj_close.loc[transcript_date])
    adj_close_delta.append(company_prices_clean_df.adj_close.loc[transcript_date+timerangedelta])
#%%
returns = [(i/j-1)*100 for i,j in zip(adj_close_delta,adj_close_dayof)]
buy_signal = []
return_thres = 3
for ret in returns:
    if ret <= -return_thres:
        buy_signal.append(-1)
    elif ret >= return_thres:
        buy_signal.append(1)
    else:
        buy_signal.append(0)
#%%
classification_clean_df['adj_close_dayof'] = adj_close_dayof
classification_clean_df['adj_close_delta'] = adj_close_delta
classification_clean_df['returns'] = returns
classification_clean_df['buy_signal'] = buy_signal
classification_clean_df
#%%
ticker_data = wb.get_data_yahoo('M','01/01/2001',interval='m')
pd.DataFrame(ticker_data)
#%%
#%%
start_date = date(2001,1,1) # delta before first transcript
end_date = datetime.now()-timedelta(1) # yesterday
m_df = pd.DataFrame(wb.get_data_yahoo('M',start_date,interval='m'))
jcp_df = pd.DataFrame(wb.get_data_yahoo('JCP',start_date,interval='m'))
kss_df = pd.DataFrame(wb.get_data_yahoo('KSS',start_date,interval='m'))
jwn_df = pd.DataFrame(wb.get_data_yahoo('JWN',start_date,interval='m'))
#%%
print(m_df.shape)
print(jcp_df.shape)
print(kss_df.shape)
print(jwn_df.shape)
#%%
ticker = 'M'
timerangedelta = timedelta(30)
start_date = min(classification_clean_df.date)-timerangedelta # delta before first transcript
end_date = datetime.now()-timedelta(1) # yesterday
company_prices_clean_df = get_company_prices_clean_df(ticker,start_date,end_date)
company_prices_clean_df.tail()
#%%
plot_company_prices(classification_clean_df,company_prices_clean_df,timerangedelta)
#%%
classification_df = get_returns_df(classification_clean_df,company_prices_clean_df,timerangedelta,return_thres=3)
classification_df
#%%
#%%
# Data
df_cross_compare = pd.DataFrame({'x':m_df.index,'y1':jcp_df['Adj Close'],'y2':kss_df['Adj Close'],'y3':m_df['Adj Close'],'y4':jwn_df['Adj Close']})
 
# multiple line plot
plt.figure(figsize=(20,8))
plt.rcParams.update({'font.size': 18})
plt.margins(0)
plt.title("Stock Prices of Department Stores" + " \n2001 - 2019")

plt.plot('x','y1',data=df_cross_compare,color='#CC1102',linewidth=3,label="JCPenney")
plt.plot('x','y2',data=df_cross_compare,color='#800433',linewidth=3,label="Kohl's")
plt.plot('x','y3',data=df_cross_compare,linewidth=3,label="Macy's")
plt.plot('x','y4',data=df_cross_compare,color='gray',linewidth=3,label="Nordstrom")
plt.ylabel('Stock Price')
plt.legend(loc='upper left')
#%%
plt.plot('x','y1',data=df_cross_compare,marker='o',markerfacecolor='blue',markersize=12,color='skyblue',linewidth=4)
plt.plot('x','y2',data=df_cross_compare,marker='',color='olive',linewidth=2)
plt.plot('x','y3',data=df_cross_compare,marker='',color='olive',linewidth=2,linestyle='dashed',label="toto")
#%%
ax1 = plt.axes(frameon=False)
ax1.set_frame_on(False)
ax1.get_xaxis().tick_bottom()
ax1.axes.get_yaxis().set_visible(False)
#%%
fig,ax1 = plt.subplots()
plt.rcParams.update({'font.size':fontsize})

ax1.margins(0)
ax1.set_title("Macy's Trading Volume" + " \nQ3 2014 - Q3 2019")

volume = ax1.plot(list(zoomed_in_df.index),zoomed_in_df.volume,color="orange",alpha=0.8,label='Trading Volume',linewidth=4)
ax1.set_ylim(0,max(zoomed_in_df.volume)*1.5)
ax1.set_ylabel('Trading Volume (MM)')

timerangedelta = timedelta(0)

for transcript_date in classification_clean_pres_df.date:
    ax1.axvspan(date2num(transcript_date),date2num(transcript_date+timerangedelta),color="gray",alpha=0.3)

line1 = ax1.plot(list(zoomed_in_df.index),[0]*len(zoomed_in_df.index),color="gray",alpha=0.8,label='Earnings Release')
charts = line1+volume
labels = [chart.get_label() for chart in charts]
ax1.legend(charts,labels,loc='upper right',fontsize=fontsize,frameon=True).get_frame().set_edgecolor('black')

fig.autofmt_xdate()
fig.set_size_inches(20,8)
plt.show()
fig.savefig('figures/company_prices_adjclose1.svg')
#%%
fig,ax1 = plt.subplots()

ax1.margins(0)
ax1.set_title("Macy's Trading Volume and Stock Price" + " \nQ3 2014 - Q3 2019")

volume = ax1.plot(list(zoomed_in_df.index),zoomed_in_df.volume,color="orange",alpha=0.8,label='Trading Volume',linewidth=4)
ax1.set_ylim(0,max(zoomed_in_df.volume)*1.5)
ax1.set_ylabel('Trading Volume (MM)')   

ax2 = ax1.twinx()
stock_price = ax2.plot(list(zoomed_in_df.index),zoomed_in_df.adj_close,alpha=0.8,label='Stock Price',linewidth=3)
ax2.set_ylim(0,max(zoomed_in_df.adj_close)*1.1)
ax2.set_ylabel('Stock Price')

timerangedelta = timedelta(0)

for transcript_date in classification_clean_pres_df.date:
    ax1.axvspan(date2num(transcript_date),date2num(transcript_date+timerangedelta),color="gray",alpha=0.3)

line1 = ax1.plot(list(zoomed_in_df.index),[0]*len(zoomed_in_df.index),color="gray",alpha=0.8,label='Earnings Release')
charts = line1+volume+stock_price
labels = [chart.get_label() for chart in charts]
ax1.legend(charts,labels,loc='upper right',fontsize=fontsize,frameon=True).get_frame().set_edgecolor('black')

fig.autofmt_xdate()
fig.set_size_inches(20,8)
plt.show()
fig.savefig('figures/company_prices_adjclose2.svg')
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
import sys
sys.setrecursionlimit(100000)

import pickle
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
filename = 'company_df'+'_'+timestamp
with open('data/'+filename+'.pickle', 'wb') as to_write:
    pickle.dump(company_df, to_write)
#%%
filename = 'data/' + 'company_df_2019-12-01-12-23-05.pickle'
with open(filename,'rb') as read_file:
    company_df = pickle.load(read_file)
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%
#%%