# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:42:37 2020

@author: aelej
"""

import numpy as np
from lagMatrix import lagmatrix

# simulation of arma processes

def arp(params,T,burn=500):
    
    P = params.size - 1
    delta = params[0]
    phi = params[1:].reshape((P,1))
    
    y = np.random.rand(T+burn,1)
    e = np.random.randn(T+burn,1)
    
    denom = 1
    for p in range(P):
        denom = denom - params[p]
    y[:burn] = delta/denom
    
    Y_ = lagmatrix(y,P,delta/denom)
    
    for t in range(P,T+burn):
        y = delta + Y_ @ phi + e
        Y_ = np.c_[np.zeros((T+burn,1)),Y_]
        Y_ = lagmatrix(y,P,delta/denom)
    
    y = y[burn:]
    
    return y

def maq(params,T,burn=500):
    
    Q = params.size - 1
    delta = params[0]
    theta = params[1:].reshape((Q,1))
    
    y = np.random.rand(T+burn,1)
    e = np.random.randn(T+burn,1)
    
    y[:burn] = 0
    
    E_ = lagmatrix(e,Q)
    
    for t in range(Q,T+burn):
        y = delta + E_ @ theta + e
        E_ = np.c_[np.zeros((T+burn,1)),E_]
        E_ = lagmatrix(e,Q)
    
    y = y[burn:]
    
    return y

def armapq(params,T,order,burn=500):
    
    P = order[0]
    Q = order[1]
    delta = params[0]
    phi = params[1:P+1].reshape((P,1))
    theta = params[P+1:].reshape((Q,1))
    
    y = np.random.rand(T+burn,1)
    e = np.random.randn(T+burn,1)
    
    denom = 1
    for p in range(P):
        denom = denom - phi[p]
    y[:burn] = delta/denom
    
    Y_ = lagmatrix(y,P,delta/denom)
    E_ = lagmatrix(e,Q)
    
    for t in range(max(P,Q),T+burn):
        y = delta + Y_ @ phi + E_ @ theta + e
        Y_ = np.c_[np.zeros((T+burn,1)),Y_]
        E_ = np.c_[np.zeros((T+burn,1)),E_]
        Y_ = lagmatrix(y,P,delta/denom)
        E_ = lagmatrix(e,Q)
    
    y = y[burn:]
    
    return y