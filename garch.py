# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:24:50 2020

@author: aelej
"""

import numpy as np
from lagMatrix import lagmatrix

def archp(params,T,burn=500,dist='normal',nu=None):
    
    P = params.size - 1
    omega = params[0]
    alpha = params[1:].reshape((P,1))
    
    if dist == 'normal':
        z = np.random.randn(T+burn,1)
        
    elif dist == 't':
        z = np.random.standard_t(nu,(T+burn,1))
    
    sigma2 = np.ones((T+burn,1))
    e = np.ones((T+burn,1))
    
    denom = 1
    for p in range(P):
        denom = denom - alpha[p]
    sigma2[:burn] = omega / denom
    e[:burn] = omega / denom
    
    E_ = lagmatrix(e,P,omega/denom)
    
    for t in range(1,T+burn):
        sigma2 = omega + (E_**2) @ alpha
        e[t] = z[t] * np.sqrt(sigma2[t])
        E_ = np.c_[np.zeros((T+burn,1)),E_]
        E_ = lagmatrix(e,P,omega/denom)
    
    sigma2 = sigma2[burn:]
    e = e[burn:]
    
    return sigma2,e

def garchpq(params,T,order,burn=500,dist='normal',nu=None):
    
    P = order[0]
    Q = order[1]
    omega = params[0]
    alpha = params[1:P+1].reshape((P,1))
    beta = params[P+1:].reshape((Q,1))
    
    if dist == 'normal':
        z = np.random.randn(T+burn,1)
        
    elif dist == 't':
        z = np.random.standard_t(nu,(T+burn,1))
    
    sigma2 = np.ones((T+burn,1))
    e = np.ones((T+burn,1))
    
    denom = 1
    for pq in range(1,P+Q):
        denom = denom - params[pq]
    sigma2[:burn] = omega / denom
    e[:burn] = omega / denom
    
    S_ = lagmatrix(sigma2,Q,omega/denom)
    E_ = lagmatrix(e,P,omega/denom)
    
    for t in range(max(P,Q),T+burn):
        sigma2 = omega + (E_**2) @ alpha + S_ @ beta
        e[t] = z[t] * np.sqrt(sigma2[t])
        E_ = np.c_[np.zeros((T+burn,1)),E_]
        S_ = np.c_[np.zeros((T+burn,1)),S_]
        E_ = lagmatrix(e,P,omega/denom)
        S_ = lagmatrix(sigma2,Q,omega/denom)
    
    sigma2 = sigma2[burn:]
    e = e[burn:]
    
    return sigma2,e