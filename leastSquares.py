# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:20:03 2020

@author: aelej
"""

import numpy as np
from scipy import stats

# ordinary least squares module
# beta coefficients estimation
# estimated series
# MSE
# total sum of squares, residual sum of squares, explained sum of squares
# t-test, F-test
# R2 and adjusted R2
# confidence intervals
# p-values
inv = np.linalg.inv

def regress(y,X):
    
    n = y.shape[0]
    k = X.shape[1]
    XX = X.T @ X
    
    b = inv(XX) @ X.T @ y
    y_hat = X @ b
    e = y - y_hat
    mse = e.T @ e / (n-k-1)
    se = np.sqrt(mse * np.diag(inv(XX))).reshape((k,1))
    
    return b,y_hat,e,mse,se

def R2(y,X):
    
    n = y.shape[0]
    k = X.shape[1]
    y_bar = np.sum(y) / n
    _,y_hat,e,mse,_ = regress(y,X)
    
    sst = np.sum((y-y_bar)**2) # sum square total
    sse = np.sum((y_hat-y_bar)**2) # sum square explained
    ssr = e.T @ e # sum square reiduals
    
    r2 = 1 - sse/sst
    adjr2 = 1 - (n-1) * (1-r2) / (n-k)
    
    return r2,sst,sse,ssr,adjr2

def ttest(params,data,h0=0,se=1,side=0,alpha=0.05):
    
    n = data.shape[0]
    k = params.size
    
    t = (params-h0) / se
    df = n - k - 1
    
    if side == 0: # two-sided
        ci1 = params  + alpha*se / 2
        ci2 = params - alpha*se / 2
        
        p = 1 - stats.t.pdf(t,df) - stats.t.pdf(-t,df)
        
    elif side == 1: # upper-sided
        ci1 = params  + alpha*se
        ci2 = params - alpha*se
        
        p = 1 - stats.t.pdf(t,df)
        
    elif side == 2: # low-sided
        ci1 = params  + alpha*se
        ci2 = params - alpha*se
        
        p = stats.t.pdf(t,df) - 1
    
    return t,p,ci1,ci2

