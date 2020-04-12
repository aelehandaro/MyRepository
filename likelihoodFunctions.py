# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:09:56 2020

@author: aelej
"""

import numpy as np
from scipy import special
from lagFunctions2 import almon_q,beta_lag

def arch_ll(params,data,dist='normal',outer=False):
    
    T = data.shape[0]
    omega = params[0]
    alpha = params[1]
    
    sigma2 = np.ones((T+1,1))
    sigma2[0] = omega/(1-alpha)
    data = np.append(0,data).reshape((T+1,1))
    
    for t in range(1,T+1):
        sigma2[t] = omega + alpha*data[t-1]**2
        
    sigma2 = sigma2[1:]
    data = data[1:]
    
    if dist == 'normal':
        constants = (1/2)*np.log(2*np.pi)
        ll = -((data**2)/(2*sigma2)) - constants - (1/2)*np.log(sigma2)
        
    elif dist == 't':
        v = params[2]
        
        delta = (1/2)*(np.log(v-2)+np.log(np.pi))
        G1 = np.log(special.gamma((v+1)/2))
        G2 = np.log(special.gamma(v/2))
        constants = G1 - G2 - delta
        ll = constants - (1/2)*np.log(sigma2) - ((v+1)/2)*np.log(1+(data**2)/(sigma2*(v-2)))
        
    if outer == True:
        
        return -ll
        
    else:
        
        return -np.sum(ll)
    

def garch_ll(params,data,dist='normal',outer=False):
    
    T = data.shape[0]
    omega = params[0]
    alpha = params[1]
    beta = params[2]
    
    sigma2 = np.ones((T+1,1))
    sigma2[0] = omega/(1-alpha-beta)
    data = np.append(0,data).reshape((T+1,1))
    
    for t in range(1,T+1):
        sigma2[t] = omega + alpha*data[t-1]**2 + beta*sigma2[t-1]
        
    sigma2 = sigma2[1:]
    data = data[1:]
    
    if dist == 'normal':
        constants = (1/2)*np.log(2*np.pi)
        ll = -((data**2)/(2*sigma2)) - constants - (1/2)*np.log(sigma2)
        
    elif dist == 't':
        v = params[3]
        
        delta = (1/2)*(np.log(v-2)+np.log(np.pi))
        G1 = np.log(special.gamma((v+1)/2))
        G2 = np.log(special.gamma(v/2))
        constants = G1 - G2 - delta
        ll = constants - (1/2)*np.log(sigma2) - ((v+1)/2)*np.log(1+(data**2)/(sigma2*(v-2)))
        
    if outer == True:
        
        return -ll
        
    else:
        
        return -np.sum(ll)

def gas_ll(params,data,dist='normal',outer=False):
    
    T = data.shape[0]
    omega = params[0]
    A = params[1]
    B = params[2]
    
    f = np.ones((T+1,1))
    s = np.ones((T+1,1))
    f[0] = omega/(1-B)
    data = np.append(0,data).reshape((T+1,1))
    
    if dist == 'normal':
        s[0] = data[0]**2 - f[0]
        
        for t in range(1,T+1):
            f[t] = omega + A*s[t-1] + B*f[t-1]
            s[t] = data[t]**2 - f[t]
        
        f = f[1:]
        data = data[1:]
        s = s[1:]
    
        constants = (1/2)*np.log(2*np.pi)
    
        ll = -((data**2)/(2*f)) - constants - (1/2)*np.log(f)
        
    elif dist == 't':
        v = params[3]
        s[0] = ((v+1)*data[0]**2)/((v-2)-(data[0]**2)/f[0]) - f[0]
    
        for t in range(1,T+1):
            f[t] = omega + A*s[t-1] + B*f[t-1]
            s[t] = ((v+1)*data[t]**2)/((v-2)+(data[t]**2)/f[t]) - f[t]
            
        f = f[1:]
        data = data[1:]
        s = s[1:]
        
        delta = (1/2)*(np.log(v-2)+np.log(np.pi))
        G1 = np.log(special.gamma((v+1)/2))
        G2 = np.log(special.gamma(v/2))
        constants = G1 - G2 - delta
    
        ll = constants - (1/2)*np.log(f) - ((v+1)/2)*np.log(1+(data**2)/(f*(v-2)))
    
    if outer == True:
        
        return -ll
    
    else:
        
        return -np.sum(ll)

def gas_midas_ll(params,low_freq,hi_freq,lag='beta',dist='normal',outer=False):
    
    T = low_freq.shape[0]
    nx = hi_freq.shape[0]
    omega = params[0]
    A1 = params[1]
    A2 = params[2]
    B = params[3]
    phi1 = params[4]
    phi2= params[5]
    
    f = np.ones((T+1,1))
    s1 = np.ones((T+1,1))
    s2 = np.ones((nx,T+1))
    f[0] = omega/(1-B)
    low_freq = np.append(0,low_freq).reshape((T+1,1))
    hi_freq = np.c_[np.zeros((nx,1)),hi_freq]
    
    s2[:,0:1] = hi_freq[:,0:1]**2 - f[0]
    
    if dist == 'normal':
        s1[0] = low_freq[0]**2 - f[0]
        
        if lag == 'beta':
            for t in range(1,T+1):
                f[t] = omega + A1*s1[t-1] + A2*np.sum(beta_lag(phi1,phi2,hi_freq)*s2[:,t-1:t]) + B*f[t-1]
                s1[t] = low_freq[t]**2 - f[t]
                s2[:,t:t+1] = hi_freq[:,t:t+1]**2 - f[t]
            
        elif lag == 'almon':
            phi = np.array((phi1,phi2)).reshape((2,1))
            for t in range(1,T+1):
                f[t] = omega + A1*s1[t-1] + A2*np.sum(almon_q(phi,hi_freq)*s2[:,t-1:t]) + B*f[t-1]
                s1[t] = low_freq[t]**2 - f[t]
                s2[:,t:t+1] = hi_freq[:,t:t+1]**2 - f[t]
            
        f = f[1:]
        s1 = s1[1:]
        s2 = s2[:,1:]
        low_freq = low_freq[1:]
        hi_freq = hi_freq[:,1:]
        
        constants = (1/2)*np.log(2*np.pi)
        
        ll = -((low_freq**2)/(2*f)) - constants - (1/2)*np.log(f)
        
    elif dist == 't':
        v = params[6]
        s1[0] = ((v+1)*low_freq[0]**2)/((v-2)+(low_freq[0]**2)/f[0]) - f[0]
        
        if lag == 'beta':
            for t in range(1,T+1):
                f[t] = omega + B*f[t-1] + A1*s1[t-1] + A2*np.sum(beta_lag(phi1,phi2,hi_freq)*s2[:,t-1:t])
                s1[t] = ((v+1)*low_freq[t]**2)/((v-2)+(low_freq[t]**2)/f[t]) - f[t]
                s2[:,t:t+1] = hi_freq[:,t:t+1]**2 - f[t]
            
        elif lag == 'almon':
            phi = np.array((phi1,phi2)).reshape((2,1))
            for t in range(1,T+1):
                f[t] = omega + B*f[t-1] + A1*s1[t-1] + A2*np.sum(almon_q(phi,hi_freq)*s2[:,t-1:t])
                s1[t] = ((v+1)*low_freq[t]**2)/((v-2)+(low_freq[t]**2)/f[t]) - f[t]
                s2[:,t:t+1] = hi_freq[:,t:t+1]**2 - f[t]
            
        f = f[1:]
        s1 = s1[1:]
        s2 = s2[:,1:]
        low_freq = low_freq[1:]
        hi_freq = hi_freq[:,1:]
        
        delta = (1/2)*(np.log(v-2)+np.log(np.pi))
        G1 = np.log(special.gamma((v+1)/2))
        G2 = np.log(special.gamma(v/2))
        constants = G1 - G2 - delta
        
        ll = constants - (1/2)*np.log(f) - ((v+1)/2)*np.log(1+(low_freq**2)/(f*(v-2)))
        
    if outer == True:
        
        return -ll
    
    else:
        
        return -np.sum(ll)