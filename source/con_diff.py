#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import source.pade_compact as pc

def con_diff_pre(bod,ischeme,buoyancy,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,dt,un,vn,pn,Tn,nu,Ri,eps=None):
    kx,ky = un.copy(),vn.copy()
    # all first derivatives        
    dudx = pc.dx(un,ny,nx,dy,dx,bcxu)
    dudy = pc.dy(un,ny,nx,dy,dx,bcyu)
    dvdx = pc.dx(vn,ny,nx,dy,dx,bcxv)
    dvdy = pc.dy(vn,ny,nx,dy,dx,bcyv)
        
    # all second derivatives
    ddudx = pc.ddx(un,ny,nx,dy,dx,bcxu)
    ddudy = pc.ddy(un,ny,nx,dy,dx,bcyu)
    ddvdx = pc.ddx(vn,ny,nx,dy,dx,bcxv)
    ddvdy = pc.ddy(vn,ny,nx,dy,dx,bcyv)
        
    # RHS construction
    kx[1:-1,1:-1] = -un[1:-1,1:-1]*dudx[1:-1,1:-1] - vn[1:-1,1:-1]*dudy[1:-1,1:-1] + nu*(ddudx[1:-1,1:-1]+ddudy[1:-1,1:-1])
    ky[1:-1,1:-1] = -un[1:-1,1:-1]*dvdx[1:-1,1:-1] - vn[1:-1,1:-1]*dvdy[1:-1,1:-1] + nu*(ddvdx[1:-1,1:-1]+ddvdy[1:-1,1:-1])
    
    if ischeme == 'RK1' or ischeme == 'RK4' or ischeme == 'AB2':
        dpdx,dpdy = np.zeros([ny,nx]),np.zeros([ny,nx])
        dpdx[1:-1,1:-1] = -(pn[1:-1,2:]- pn[1:-1,:-2])/(2*dx)
        dpdy[1:-1,1:-1] = -(pn[2:,1:-1]- pn[:-2,1:-1])/(2*dy)
        kx[1:-1,1:-1] = kx[1:-1,1:-1] + dpdx[1:-1,1:-1]
        ky[1:-1,1:-1] = ky[1:-1,1:-1] + dpdy[1:-1,1:-1]

    # IBM imposition
    if bod == 'IBM':
        kx = kx + eps*(-kx - un/dt)
        ky = ky + eps*(-ky - vn/dt)
        
    if buoyancy == 'on':
        ky = ky - Ri*Tn
        
    return kx,ky

def con_diff_ene(bod,bcx,bcy,ny,nx,dy,dx,dt,un,vn,Tn,nu,Pr,eps):
    Tx = Tn.copy()
    # convection first derivatives
    dTdx = pc.dx(Tn,ny,nx,dy,dx,bcx)
    dTdy = pc.dy(Tn,ny,nx,dy,dx,bcy)
    ddTdx = pc.ddx(Tn,ny,nx,dy,dx,bcx)
    ddTdy = pc.ddy(Tn,ny,nx,dy,dx,bcy)
    
    # RHS construction
    Tx[1:-1,1:-1] = -un[1:-1,1:-1]*dTdx[1:-1,1:-1] - vn[1:-1,1:-1]*dTdy[1:-1,1:-1] + (nu/Pr)*(ddTdx[1:-1,1:-1] + ddTdy[1:-1,1:-1])
    
    if bod == 'IBM':
        Tx = Tx + eps*(-Tx + ((1-Tn)/dt))
        
    return Tx
