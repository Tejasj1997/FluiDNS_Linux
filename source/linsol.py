#!/usr/bin/env python
# coding: utf-8

# In[46]:

import numpy as np
from numba import jit, njit

@njit
def LUDO(a,b,c,d):
    n = len(d)
    
    L = np.zeros(n-1)
    U = np.zeros(n)
    x,y = np.zeros(n),np.zeros(n)
    
    U[0] = b[0]
    for it in range(n-1):
        L[it] = a[it]/U[it]
        U[it+1] = b[it+1] - L[it]*c[it]
        
    y[0] = d[0]
    for il in range(1,n):
        y[il] = d[il] - L[il-1]*y[il-1]
        
    x[-1] = y[-1]/U[-1]
    for ix in range(n-2,-1,-1):
        x[ix] = (y[ix] - c[ix]*x[ix+1])/U[ix]
        
    return x

@njit
def TDMA(a,b,c,d):
    n = len(d)
    w= np.zeros(n-1,float)
    g= np.zeros(n, float)
    p = np.zeros(n,float)

    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1,n-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1,n):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[n-1] = g[n-1]
    for i in range(n-1,0,-1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p
