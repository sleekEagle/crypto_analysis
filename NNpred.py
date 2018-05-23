#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 13:13:56 2018

@author: sleek_eagle
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr 
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import math

'''
predicting with NN
'''

def main():
    data = pd.read_csv("/home/sleek_eagle/research/crypto/hourlydata.csv",sep=',')
    data = data[data['date'] > '2017-01-01-00']
    rawdata = data
    diffdata = data['price'].diff()
    data['diffprice'] = diffdata
    data['price'] = data['diffprice']
    data = data.dropna(axis = 0,how='any')
    #data['price'] = np.log(data['price'])
    lookback = 100
    d = prepData(data,lookback)
    #scale each feature vector and variable to
    f=d.iloc[:,0:(lookback)]
    v=d.iloc[:,lookback]
    f = f.div(d.sum(axis=1), axis=0)
    #resample to randomize the data
    #d = d.sample(frac=1).reset_index(drop=True)
    #divide train and test data
    TRAIN_RATIO = 0.6
    n_data = d.shape[0]
    train = d.loc[0:n_data*TRAIN_RATIO,]
    train_v = train[str(lookback)]
    train_f = train.drop(columns = [str(lookback)])
    test = d.loc[n_data*TRAIN_RATIO:n_data,]
    test_v = test[str(lookback)]
    test_f = test.drop(columns = [str(lookback)])
    
    #train with dofferent hidden layer sizes
    errors = []
    for i in range(310,1200,50):
        er = trainNN((290,460))
        errors.append([i,er])
        print(str(i))
    errors = pd.DataFrame(errors)
    
    errors = []
    for i in range(0,10):
        for k in range(0,10):
            er = myPred(test_v,test_f,i/10,k/10)
            errors.append([i/10,k/10,er])
    errors = pd.DataFrame(errors)

    
def getF_and_V_data(data,TRAIN_RATIO):
    if 'date' in data:
        dates = pd.DataFrame(data['date'])
        data = data.drop(columns = ['date'])
        n_data = data.shape[0]
        #divide dates
        train_dates = dates.loc[0:(n_data*TRAIN_RATIO),:]
        test_dates = dates.loc[n_data*TRAIN_RATIO:n_data,]
    lookback = data.shape[1]-1
    f=data.iloc[:,0:(lookback)]
    v=data.iloc[:,lookback]
    f = f.div(data.sum(axis=1), axis=0)
    #resample to randomize the data
    #d = d.sample(frac=1).reset_index(drop=True)
    #divide train and test data
    n_data = data.shape[0]
    train = data.loc[0:n_data*TRAIN_RATIO,:]
    train_v = train[str(lookback)]
    train_f = train.drop(columns = [str(lookback)])
    test = data.loc[n_data*TRAIN_RATIO:n_data,]
    test_v = test[str(lookback)]
    test_f = test.drop(columns = [str(lookback)])
    return [train_f,train_v,test_f,test_v,train_dates,test_dates]

def scaleData(train_f,train_v):
    df = train_f
    numcols = df.shape[1]
    df[str(numcols)] = train_v
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = pd.DataFrame(scaler.transform(df))
    train_f_scaled = df_scaled.iloc[:,0:numcols]
    train_v_scaled = df_scaled.iloc[:,numcols]
    return [train_f_scaled,train_v_scaled,scaler]

def scale_from_scalers(data,scalers):
    scaled_data = []
    for i in range(0,data.shape[0]):
        scaled_data.append(scalers.iloc[i].transform(data.iloc[i]).flatten()[0])
    scaled_data=pd.Series(scaled_data)
    return scaled_data

def inverse_scale_from_scalers(data,scalers):
    scaled_data = []
    for i in range(0,data.shape[0]):
        scaled_data.append(scalers.iloc[i].inverse_transform([data.iloc[i]]).flatten()[0])
    scaled_data=pd.Series(scaled_data)
    return scaled_data
    
        
    

def get_scaler_data(data):
    data_scalers = data.apply(scale_helper1,axis = 1)
    df = pd.DataFrame(list(data_scalers))
    data_scaled = pd.DataFrame(np.matrix(list(df[0])))
    scalers = df[1]
    return[data_scaled,scalers]
    
    
def scale_helper1(row):
    n = row.shape[0]
    df=pd.DataFrame(data = row)
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df).flatten()
    return df_scaled,scaler
    
    
def scale_each_vector(train_f,train_v):
    df = train_f
    numcols = df.shape[1]
    df[str(numcols)] = train_v
    df_scaled = df.apply(scale_helper,axis =1)
    train_f_scaled = df_scaled.iloc[:,0:numcols]
    train_v_scaled = df_scaled.iloc[:,numcols]
    return [train_f_scaled,train_v_scaled]

def scale_helper(row):
    n = row.shape[0]
    df=pd.DataFrame(data = row.iloc[0:(n-1)])
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df).flatten()
    v_scaled =  scaler.transform(row.iloc[n-1]).flatten()
    ar = np.concatenate((df_scaled,v_scaled))
    return ar


def trainNN(layers,lr,train_f,train_v,test_f,test_v):
    #train the NN
    reg = MLPRegressor(hidden_layer_sizes=(layers), activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate='constant', learning_rate_init=lr, power_t=0.5, max_iter=2000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    reg.fit(train_f, train_v)
    
    pred = pd.DataFrame(reg.predict(test_f))
    pred.columns = ['pred']
    pred['actual'] = test_v.values
    #sort this
    #pred = pred.sort_values(by = ['actual']).reset_index()
    #plt.plot(pred['pred'])
    #plt.plot(pred['actual'])
    error = mean_squared_error(pred['actual'],pred['pred'])
    return [error,pred]

'''
def myPred(test_v,test_f,n1,n2):
    diff1 = (test_f[str(test_f.shape[1]-1)]-test_f[str(test_f.shape[1]-2)])
    diff2 =  ((test_f[str(test_f.shape[1]-1)]-test_f[str(test_f.shape[1]-2)]) - (test_f[str(test_f.shape[1]-2)]-test_f[str(test_f.shape[1]-3)]))
    pred = pd.DataFrame(test_f[str(test_f.shape[1]-1)] + n1*diff1 + n2 *diff2 + n3* )
    pred.columns = ['pred']
    pred['actual'] = test_v.values
    
    plt.plot(pred['actual'])
    plt.plot(pred['pred'])
    error = mean_squared_error(pred['actual'],pred['pred'])
    return error
'''    


    

def prepData(data,lookback): #there should be a columns called date
    i=lookback
    df = data.iloc[i-lookback:i+1].drop(columns = ['date']).T
    names = []
    for k in range(0,lookback+1):
        names.append(str(k))
    df.columns = names
    df['date'] = data.iloc[i]['date']
    for i in range(lookback+1,(data.shape[0]-1)):
        tmp = data.iloc[i-lookback:i+1].drop(columns = ['date']).T
        tmp.columns = names
        tmp['date'] = data.iloc[i]['date']
        df = df.append(tmp,ignore_index=True)
        s = str(i) + " of " + str((data.shape[0]-1))
        print(s)
    return df
