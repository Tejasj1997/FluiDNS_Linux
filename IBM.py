#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

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

def circle(ene,ny,nx,dy,dx,x_pos,y_pos,ra):
    eps = np.zeros([ny,nx])
    eps_in = np.ones([ny,nx])
    
    # y_pos modification for transverse oscillations
    fn,amp = 1,1
    y_pos = y_pos + amp*np.sin(2*np.pi*fn*0.141*t) 
    print("fn = " +str(fn) + "; amp = " +str(amp))
        
    for i in range(ny):
        for j in range(nx):
            ym,xm = i*dy,j*dx
            r = np.sqrt((xm-x_pos)**2 + (ym-y_pos)**2)
            if (r-ra) <= 0:
                eps[i,j] = 1
                if ene == 'on':
                    eps_in[i,j] = 0
                
    return eps,eps_in
    