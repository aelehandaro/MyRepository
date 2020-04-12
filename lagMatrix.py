# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:25:52 2020

@author: aelej
"""

import numpy as np

def lagmatrix(series,lags,constant=0):
    '''
    takes a (T,1) vector and constructs a matrix of lagged values. the size of the matrix depends on the number of lags specified'''
    
    T = series.size
    lag_matrix = np.ones((T,lags+1))*constant
    # we add one extra column, which we will drop at the end
    # this column would store the original series, which we don't need in the lag matrix
    for l in range(lags+1):
        lag = series[:series.size-l]
        lag_matrix[l:,l:l+1] = lag
        
    lag_matrix = lag_matrix[:,1:]
    
    return lag_matrix