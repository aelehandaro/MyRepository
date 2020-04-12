# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:55:54 2020

@author: aelej
"""

import numpy as np

def Sdx(func,a,b,c=1e+3):
    
    x = np.linspace(a,b,int(c))
    h = 1 / c
    base = 2 * h
    height = func(x)
    area = base * height
    
    return np.sum(area)