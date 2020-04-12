# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 21:39:33 2020

@author: aelej
"""

import numpy as np

# for single variable differentiation
def dydx(func,x,params=None,method=0):
    
    h = 0.1
    
    if params == None:
        f_h1 = func(x-h) - func(x)
        f_h2 = func(x+h) - func(x)
        
    else:
        f_h1 = func(x-h,params) - func(x,params)
        f_h2 = func(x+h,params) - func(x,params)
    
    if method == 0:
        f_prime = (f_h2 - f_h1) / (2*h)
        
    elif method == 1:
        f_prime = -f_h1 / h
        
    elif method == 2:
        f_prime = f_h2 / h
    
    return f_prime

def d2ydx2(func,x,params,method=0):
    
    h = 0.1
    
    if params == None:
        f_h1prime = dydx(func,x-h) - dydx(func,x)
        f_h2prime = dydx(func,x+h) - dydx(func,x)
        
    else:
        f_h1prime = dydx(func,x-h,params) - dydx(func,x,params)
        f_h2prime = dydx(func,x+h,params) - dydx(func,x,params)
    
    if method == 0:
        f_2prime = (f_h2prime - f_h1prime) / (2*h)
        
    elif method == 1:
        f_2prime = -f_h1prime / h
        
    elif method == 2:
        f_2prime = f_h2prime / h
    
    return f_2prime

# for multiple variables
def gradient(func,x,params=None,method=0):
    
    h = 0.1
    x = np.array(x)
    k = x.size
    
    grad = np.zeros((k,1),np.float64)
    
    for i in range(k):
        xx_h1 = np.array(x.copy(),np.float64)
        xx_h2 = np.array(x.copy(),np.float64)
        x_h1 = x[i] - h
        x_h2 = x[i] + h
        
        xx_h1[i:i+1] = x_h1
        xx_h2[i:i+1] = x_h2
        
        if params == None:
            f_h1 = func(xx_h1) - func(x)
            f_h2 = func(xx_h2) - func(x)
            
        else:
            f_h1 = func(xx_h1,params) - func(x,params)
            f_h2 = func(xx_h2,params) - func(x,params)
        
        if method == 0:
            grad[i] = (f_h2 - f_h1) / (2*h)
            
        elif method == 1:
            grad[i] = -f_h1 / h
            
        elif method == 2:
            grad[i] = f_h2 / h
    
    return grad

def hessian(func,x,params=None): # method?
    
    h = 0.1
    x = np.array(x)
    k = x.size
    
    Hessian = np.zeros((k,k),np.float64)
    
    for i in range(k): # row
        for j in range(k): # column (e.g. x or y)
            if i == j:
                xx_h1 = np.array(x.copy(),np.float64) # copy original point x
                xx_h2 = np.array(x.copy(),np.float64)
                x_h1 = x[j] - 2*h # create point x - h
                x_h2 = x[j] + 2*h # create point x + h
                
                xx_h1[j:j+1] = x_h1 # given column number, replace element
                xx_h2[j:j+1] = x_h2 # each row is for a variable x or y
                
                if params == None:
                    f_h1 = func(xx_h1)
                    f_h2 = func(xx_h2)
                    f = func(x)
                    
                else:
                    f_h1 = func(xx_h1,params)
                    f_h2 = func(xx_h2,params)
                    f = func(x,params)
                
                hess = f_h2 - 2*f + f_h1
                
                Hessian[i][j] = hess / (4*h**2)
                
            else:
                xx_h1 = np.array(x.copy(),np.float64)
                xx_h2 = np.array(x.copy(),np.float64)
                xx_h3 = np.array(x.copy(),np.float64)
                xx_h4 = np.array(x.copy(),np.float64)
                
                x_h1 = x[j] - h
                x_h2 = x[j] + h
                x_h3 = x[i] - h
                x_h4 = x[i] + h
                
                xx_h1[j:j+1] = x_h1
                xx_h1[i:i+1] = x_h3
                xx_h2[j:j+1] = x_h2
                xx_h2[i:i+1] = x_h4
                xx_h3[j:j+1] = x_h1
                xx_h3[i:i+1] = x_h4
                xx_h4[j:j+1] = x_h2
                xx_h4[i:i+1] = x_h3
                
                if params == None:
                    hess = func(xx_h2) - func(xx_h3) - func(xx_h4) + func(xx_h1)
                    
                else:
                    hess = func(xx_h2,params) - func(xx_h3,params) - func(xx_h4,params) + func(xx_h1,params)
                
                Hessian[i][j] = hess / (4*h**2)
    
    return Hessian