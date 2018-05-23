#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:35:30 2018

@author: sleek_eagle
"""

import gdax
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep



def gatherData():
    public_client = gdax.PublicClient() #API reference : https://docs.gdax.com/#api
    start = '2016-02-01T12:00:00' # start date of the dataset date time in ISO 8601 format
    numdays = 800 # get data this number of days into the future from start date
    granulity = 3600 # in seconds
    
    datelist = pd.date_range(start, periods=numdays).tolist()
    dates = []
    for date in datelist:
        dates.append(getDateTime(date))
        
    #create the file to write data    
    file = open('/home/sleek_eagle/research/crypto/1hrdata.csv','w') 
    file.write('time,low,high,open,close,volume\n') 
    file.close()
    
    #retrieve data from API ad append to file
    
    startdate=dates[0]
    for i in range(1,len(dates)):
        datalen = 0
        tries = 0
        while(datalen < 10):
            tries +=1
            try:
                if((1+tries*0.5 > 5)): #check if the data retried contains enough data. request again if it only has very little data
                    public_client = gdax.PublicClient()
                data = public_client.get_product_historic_rates('BTC-USD',start = startdate,end = dates[i], granularity=granulity)
                data = pd.DataFrame(data)
                datalen = len(data)
                sleep(1 + tries*0.5)
                print("this length = " + str(len(data)))
            except Exception as exp:
                print(exp)
            print("new try")
            
        with open('/home/sleek_eagle/research/crypto/1hrdata.csv','a') as file:
            data.to_csv(file,header=False,index=False)
        
        startdate = dates[i]
        print(len(data))
        print(i)

gatherData()
#data preperation for average daily price
datapath = "/home/sleek_eagle/research/crypto/1hrdata.csv"
data = pd.read_csv(datapath,sep=',')
data['avg_price'] = (data['low'] + data['high'])/2
data['range'] = abs(data['low'] - data['high'])
prep5mindata(data)

def prep5mindata(data):
    data ['time'] = data['time'].apply(getDateTimeMin)
    #if there are data points within the same 5 min interval, get mean of them
    #datedata = data.groupby(['time'])['avg_price',''].agg('mean') 
    #df = pd.DataFrame(datedata.values,columns = ['avg_price','low','high'])
    #df['time'] = datedata.index
    #df.columns = ['price','datetime']
    df= data
    df.columns = ['datetime','low','high','open','close','volume','avg_price','range']
    
    daterange = (pd.Series(pd.date_range(start = df['datetime'][0][0:10],end = df['datetime'][df.shape[0]-1][0:10])))
    daterange = pd.DataFrame(daterange.apply(getStrDate))
    daterange.columns = ['date']
    times = getMinIntervals(60) # get 5 min intervals
    for s in times:
        daterange[s] = s
    daterange = daterange.melt(id_vars = ['date']).drop(['variable'],axis = 1)
    daterange = daterange.sort_values(by = ['date','value']).reset_index().drop(['index'],axis = 1)
    daterange = daterange.astype(str)
    daterange['datetime'] = daterange['date'] +'-'+ daterange['value']
    daterange = daterange.drop(['date','value'],axis = 1)
    daterange['price'] = 0
    
    mer  = pd.merge(left = df, right = daterange, how = 'outer',on = 'datetime')
    mer = mer.sort_values(by=['datetime']).reset_index()
    mer = mer.drop(['index'],axis = 1)
    #remove parts that are not in the initail dataset
    cond1 = mer['datetime'] >= df.iloc[0]['datetime']
    cond2 = mer['datetime'] <= df.iloc[df.shape[0]-1]['datetime']
    notindex = mer[cond1 & cond2].index
    goodmer = pd.DataFrame(mer.iloc[notindex,:])  
    goodmer['low'] = goodmer['low'].interpolate()
    goodmer['high'] = goodmer['high'].interpolate()
    goodmer['open'] = goodmer['open'].interpolate()
    goodmer['close'] = goodmer['close'].interpolate()
    goodmer['volume'] = goodmer['volume'].interpolate()
    goodmer['avg_price'] = goodmer['avg_price'].interpolate()
    goodmer['range'] = goodmer['range'].interpolate()

    goodmer = goodmer.drop(['price'],axis = 1)
    #write daily data to file
    goodmer.to_csv(path_or_buf = "/home/sleek_eagle/research/crypto/1hrdata_filled.csv",sep=',',index = False)

def getMinIntervals(mins):
    times = []
    for i in range(0,24):
        for k in range(0,60,mins):
            s = str(i).zfill(2) + '-' + str(k).zfill(2)
            times.append(s)
    return times
    
def getDateTimeMin(t):
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d-%H-%M")
def getStrDate(t):
    return str(t.strftime("%Y-%m-%d"))

def getDateTime(t):
    return datetime.datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S" ).isoformat() + '-05:00'
