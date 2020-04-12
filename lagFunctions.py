# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 11:02:46 2020

@author: aelej
"""

import numpy as np
from scipy import special
from scipy import stats

def almon_1(phi,data):

    nx = data.shape[0] # data is hi freq ((nx,T))
    ix = np.linspace(1,nx,num=nx,endpoint=True)
    
    numerator = np.exp(phi*ix)
    denominator = np.sum(np.exp(phi*ix))
    
    weights = numerator / denominator
    
    return weights

def almon_2(phi1,phi2,data):
    
    nx = data.shape[0]
    ix = np.linspace(1,nx,num=nx,endpoint=True)
    
    numerator = np.exp(phi1*ix + phi2*ix**2)
    denominator = np.sum(np.exp(phi1*ix + phi2*ix**2))
    
    weights = numerator / denominator
    
    return weights

def almon_q(phi,data):
    
    nx = data.shape[0]
    q = phi.size
    phi = phi.reshape((q,1))
    ix = np.linspace(1,nx,num=nx,endpoint=True).reshape((nx,1))
    IX = np.ones((nx,q))
    
    for lag in range(1,q+1):
        IX[:,lag-1:lag] = ix ** lag
    
    IX_phi = IX @ phi
    exp_phi = np.exp(IX_phi)
    sum_exp = np.sum(exp_phi)
    
    weights = exp_phi / sum_exp
    
    return weights

def beta_lag(phi1,phi2,data): # phis must be positive
    
    nx = data.shape[0]
    ix = np.linspace(1,nx,num=nx,endpoint=True).reshape((nx,1))
    
    beta = stats.beta.pdf(ix/nx,phi1,phi2)
    sum_beta = np.sum(beta)
    
    weights = beta / sum_beta
    
    return weights