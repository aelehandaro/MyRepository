# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:40:00 2020

@author: aelej
"""

import numpy as np
# import matplotlib.pyplot as plt

def autocov(series,h):
    
    T = series.size
    y_bar = np.sum(series) / T
    
    if h == 0:
        gamma0 = np.sum((series-y_bar)**2) / (T-1)
        
        return gamma0
    
    else:
        gamma_h = np.zeros(h)
        for lag in range(h):
            y_t = series[lag:T]
            y_t_h = series[:T-lag]
            gamma_h[lag] = np.sum((y_t-y_bar)*(y_t_h-y_bar)) / (T-1)

        return gamma_h

def autocorr(series,h):
    
    gamma0 = autocov(series,0)
    rho_h = autocov(series,h) / gamma0
    
    return rho_h

# def plot(sacf):
    
#     h = sacf.size
    
#     return plt.bar(np.arange(0,h),
#                    sacf,
#                    0.5,
#                    color='black',
#                    tick_label=np.arange(h))
    
    
#     pass


# plt.bar(np.arange(0,20),
#         ibm_autocorr,
#         0.5,
#         color='black',
#         tick_label=np.arange(20))
# plt.plot(np.zeros(20),linewidth=1,color='black')
# plt.title('sacf - ibm')

# def plot(sacf,
#          h=20,
#          bottom=0.5,
#          color='black',
#          ticket_label=np.arange(20)):
        
        