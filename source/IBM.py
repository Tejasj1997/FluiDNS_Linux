#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba import jit, njit

def square(mot,ene,ny,nx,dy,dx,x_pos,y_pos,length,width,amp,st,fn,t):
    eps = np.zeros([ny,nx])
    eps_in = np.ones([ny,nx])
    
    if mot == 'FIV':
        y_pos = y_pos + amp*np.sin(2*np.pi*fn*st*t) 
        print("fn = " +str(fn) + "; amp = " +str(amp))
    
    for i in range(ny):
        for j in range(nx):
            ym,xm = i*dy,j*dx
            if xm > x_pos and xm < x_pos + length:
                if ym > y_pos and ym < y_pos+width:
                    eps[i,j] = 1
                    if ene == 'on':
                        eps_in[i,j] = 0
                        
#             if xm > x_pos and xm < x_pos + length:
#                 if ym > y_pos + 3.5 and ym < y_pos + 3.5 + width:
#                     eps[i,j] = 1
#                     if ene == 'on':
#                         eps_in[i,j] = 0
                        
    return eps,eps_in

@jit
def circle(mot,ene,ny,nx,dy,dx,x_pos,y_pos,ra,amp,st,fn,t):
    eps = np.zeros([ny,nx])
    eps_in = np.ones([ny,nx])
    plxlf,plylf,plxrg,plyrg = np.array([]),np.array([]),np.array([]),np.array([])
    
    if mot == 'FIV':
        y_pos = y_pos + amp*np.sin(2*np.pi*fn*st*t) 
        print("fn = " +str(fn) + "; amp = " +str(amp))
        
    for i in range(ny):
        f2 = 0   # flag 1
        for j in range(nx):
            f1 = 0 # flag2
            ym,xm = i*dy,j*dx
            r = np.sqrt((xm-x_pos)**2 + (ym-y_pos)**2)
            if (r-ra) <= 0:
                f1 = 1
                eps[i,j] = 1
                if ene == 'on':
                    eps_in[i,j] = 0
                    
            if f1 == 1 and f2 == 0:
                plxlf,plylf = np.append(plxlf,i),np.append(plylf,j)
                f2 = 1
                
            if f1 == 0 and f2 == 1:
                plxrg,plyrg = np.append(plxrg,i),np.append(plyrg,j)
                f2 = 0
                
    
    return eps,eps_in,plxlf,plylf,plxrg,plyrg
    