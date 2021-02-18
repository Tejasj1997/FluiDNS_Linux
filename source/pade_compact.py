#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import source.auxfunx as af
# All derivative routines for now are constructed for following BCS
# Once these routines get working, other boundaries will be added
# inflow = constant u velocity inflow
# outflow = continuation same velocity boundary
# upper and lower walls = No slip boundaries

def dx(f,ny,nx,dy,dx,bcx):
    dudx = np.zeros([ny,nx])
    a,b,c,B = np.zeros(nx-1)+0.25,np.ones(nx),np.zeros(nx-1)+0.25,np.zeros(nx)
    b[0],b[-1] = 1.25,1.25 #(1+alpha)
    for l in range(1,ny-1):
        B[1:-1] = (f[l,2:]-f[l,:-2])/(2*dx)
            
        if bcx == 'inout' or bcx == 'inout-os' or bcx == 'parab':
            B[0] = (f[l,1]-f[l,0])/(2*dx) 
            B[-1] = (f[l,-1]-f[l,-2])/(2*dx)
            
        elif bcx == 'noslip':
            B[0],B[1] = f[l,1]/(2*dx),f[l,2]/(2*dx) 
            B[-1],B[-2] = -f[l,-2]/(2*dx),-f[l,-3]/(2*dx)
            
        elif bcx == 'isoth':
            B[0],B[1] = (f[l,1]-1)/(2*dx),(f[l,2]-1)/(2*dx) 
            B[-1],B[-2] = (1-f[l,-2])/(2*dx),(1-f[l,-3])/(2*dx)
            
        B = (3/2)*B  #(a = 3/2 on RHS)
        dudx[l,:] = af.LUDO(a,b,c,B)
        
    return dudx

# First Derivative according to Pade compact
def dy(f,ny,nx,dy,dx,bcy,q):
    dudy = np.zeros([ny,nx])
    a,b,c,B = np.zeros(ny-1)+0.25,np.ones(ny),np.zeros(ny-1)+0.25,np.zeros(ny)
    b[0],b[-1] = 1.25,1.25 #(1+alpha)
    for l in range(1,nx-1):
        B[1:-1] = (f[2:,l]-f[:-2,l])/(2*dy)
            
        if bcy == 'inout':
            B[0] = (f[0,l]-f[0,l])/(2*dy) 
            B[-1] = (f[-2,l]-f[-2,l])/(2*dy)
        elif bcy == 'noslip':
            B[0],B[1] = f[1,l]/(2*dy),f[2,l]/(2*dy) 
            B[-1],B[-2] = -f[-2,l]/(2*dy),-f[-3,l]/(2*dy)
        elif bcy == 'isoth':
            B[0],B[1] = (f[1,l]-1)/(2*dy),(f[2,l]-1)/(2*dy) 
            B[-1],B[-2] = (1-f[-2,l])/(2*dy),(1-f[-3,l])/(2*dy)
        elif bcy == 'adiab':
            B[0] = q
            B[-1] = -q

        B = (3/2)*B  #(a = 3/2 on RHS)
        dudy[:,l] = af.LUDO(a,b,c,B)
        
    return dudy
    
# Second derivatives according to Pade compact
def ddx(f,ny,nx,dy,dx,bcx):
    ddudx = np.zeros([ny,nx])
    a,b,c,B = np.zeros(nx-1)+0.1,np.ones(nx),np.zeros(nx-1)+0.1,np.zeros(nx)
    b[0],b[-1] = 1.1,1.1 #(1+alpha)
    for l in range(1,ny-1):
        B[1:-1] = (f[l,2:]-2*f[l,1:-1]+f[l,:-2])/(dx**2)
            
        if bcx == 'inout' or bcx == 'inout-os' or bcx == 'parab':
            B[0] = (f[l,1]-f[l,0])/(dx**2)
            B[-1] = (f[l,-2]-f[l,-1])/(dx**2)
            
        elif bcx == 'noslip':
            B[0],B[1] = f[l,1]/(dx**2),(f[l,2]-2*f[l,1])/(dx**2)
            B[-1],B[-2] = f[l,-2]/(dx**2),(-2*f[l,-2]+f[l,-3])/(dx**2)
            
        elif bcx == 'isoth':
            B[0],B[1] = (f[l,1]-1)/(dx**2),(f[l,2]-2*f[l,1]+1)/(dx**2)
            B[-1],B[-2] = (-1+f[l,-2])/(dx**2),(1-2*f[l,-2]+f[l,-3])/(dx**2)
            
        B = 1.2*B  #(a = 1.2 on RHS)
        ddudx[l,:] = af.LUDO(a,b,c,B)
    return ddudx
    
def ddy(f,ny,nx,dy,dx,bcy,q):
    ddudy = np.zeros([ny,nx])
    a,b,c,B = np.zeros(ny-1)+0.1,np.ones(ny),np.zeros(ny-1)+0.1,np.zeros(ny)
    b[0],b[-1] = 1.1,1.1 #(1+alpha)
    for l in range(1,nx-1):
        B[1:-1] = (f[2:,l]-2*f[1:-1,l]+f[:-2,l])/(dy**2)
            
        if bcy == 'inout':
            B[0] = (f[1,l]-f[0,l])/(dy**2)
            B[-1] = (f[-2,l]-f[-1,l])/(dy**2)
            
        elif bcy == 'noslip':
            B[0],B[1] = f[1,l]/(dy**2),(f[2,l]-2*f[1,l])/(dy**2)
            B[-1],B[-2] = f[-2,l]/(dy**2),(-2*f[-2,l]+f[-3,l])/(dy**2)
            
        elif bcy == 'isoth':
            B[0],B[1] = (f[1,l]-1)/(dy**2),(f[2,l]-2*f[1,l]+1)/(dy**2)
            B[-1],B[-2] = (-1+f[-2,l])/(dy**2),(1-2*f[-2,l]+f[-3,l])/(dy**2)
            
        elif bcy == 'adiab':
            B[0] = q
            B[-1] = -q
     
        B = 1.2*B  #(a = 1.2 on RHS)
        ddudy[:,l] = af.LUDO(a,b,c,B)
    return ddudy
        
