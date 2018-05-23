#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:19:15 2018

@author: sleek_eagle
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr 
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor

  

#data preperation for average daily price
datapath = "/home/sleek_eagle/research/crypto/coinbaseUSD.csv"
data = pd.read_csv(datapath,sep=',',header = None)
data.columns = ['ts','price','volume']


def writeHrData(data):
    data ['ts'] = data['ts'].apply(getDateTimeHr)
    data = data[['ts','price']]
    
    datedata = data.groupby(['ts'])['price'].agg('mean')
    df = pd.DataFrame(datedata.values,columns = ['price'])
    df['ts'] = datedata.index
    df.columns = ['price','datetime']
    
    daterange = (pd.Series(pd.date_range(start = df['time'][0][0:10],end = df['time'][df.shape[0]-1][0:10])))
    daterange = pd.DataFrame(daterange.apply(getStrDate))
    
    daterange.columns = ['date']
    for i in range(0,24):
        daterange[str(i)] = i
    daterange = daterange.melt(id_vars = ['date']).drop(['variable'],axis = 1)
    daterange = daterange.sort_values(by = ['date','value']).reset_index().drop(['index'],axis = 1)
    daterange = daterange.astype(str)
    daterange['value'] = daterange['value'].str.zfill(2)
    daterange['datetime'] = daterange['date'] +'-'+ daterange['value']
    daterange = daterange.drop(['date','value'],axis = 1)
    daterange['price'] = 0
    mer  = pd.merge(left = df, right = daterange, how = 'outer',on = 'datetime')
    mer = mer.sort_values(by=['datetime']).reset_index()
    mer = mer.drop(['index'],axis = 1)
    mer['price_x'] = mer['price_x'].interpolate()
    mer = mer.drop(['price_y'],axis = 1)
    mer.columns = ['price','date']
    #write daily data to file
    mer.to_csv(path_or_buf = "/home/sleek_eagle/research/crypto/hourlydata.csv",sep=',',index = False)


#prepare hourly data
def writeDateData(data):
    data ['ts'] = data['ts'].apply(getDateTime)
    data = data[['ts','price']]
    #get hourly data
    datedata = data.groupby(['ts'])['price'].agg('mean')
    df = pd.DataFrame(datedata.values,columns = ['price'])
    df['ts'] = datedata.index
    df.columns = ['price','date']
    
    daterange = pd.Series(pd.date_range(start = df['ts'][0],end = df['ts'][df.shape[0]-1]))
    daterange = daterange.apply(getDate1)
    fulldata = pd.DataFrame(daterange,columns = ['date'])
    fulldata['price'] = 0
    
    mer  = pd.merge(left = df, right = fulldata, how = 'outer',on = 'date')
    mer = mer.sort_values(by=['date']).reset_index()
    mer = mer.drop(['index'],axis = 1)
    mer['price_x'] = mer['price_x'].interpolate()
    mer = mer.drop(['price_y'],axis = 1)
    mer.columns = ['price','date']
    daily = mer
    #write daily data to file
    mer.to_csv(path_or_buf = "/home/sleek_eagle/research/crypto/dailydata.csv",sep=',',)

    

def getDate(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
def getStrDate(t):
    return str(t.strftime("%Y-%m-%d"))
def getDate1(t):
    return t.strftime("%Y-%m-%d")
def getDateTimeHr(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d-%H")


'''
predicting daily price with ARIMA
'''
data = pd.read_csv("/home/sleek_eagle/research/crypto/dailydata.csv",sep=',', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
def parser(x):
	return datetime.datetime.strptime(x,"%Y-%m-%d")
#rolling forecast
X = data.values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    model = ARIMA(history[len(history) - 100 :len(history)], order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print('len history = %f' %len(history))
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
pearsonr(np.array(test),np.array(predictions).flatten())
# plot
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
er = np.array(test) - np.array(predictions).flatten()









'''
def prepData(data,lookback):
    data['modindex'] = data.index%(lookback+1)
    df = data[data['modindex']==0].reset_index()
    df['index'] = df.index
    df = df.drop(columns = ['date','modindex'])
    for i in range(1,(lookback+1)):
        tmp = data[data['modindex']==i].reset_index()
        tmp['index'] = tmp.index
        tmp = tmp.drop(columns = ['index','date','modindex'])
        df = pd.concat([df,tmp],axis = 1)
    df = df.drop(columns = ['index'])
    df = df.dropna(axis = 0,how = 'any')  
    return df    
 '''           
    
