#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import source.pade_compact as pc
import source.auxfunx as af
import source.con_diff as cdp
import source.bound_co as bc

def RK1(bod,ischeme,poi_b,ene,buo,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,ny,nx,dy,dx,rho,nu,Ri,Pr,dt,u,v,p,T,vel,kn,t,A0,omp,eps,eps_in,T0):
    #################### RK1 ### Time Integrator #############################################
    un,vn,pn = u.copy(),v.copy(),p.copy()
    Tn = T.copy()
    # slope calculation
    kx,ky = cdp.con_diff_pre(bod,ischeme,buo,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,dt,un,vn,pn,Tn,nu,Ri,eps)
    # RK1 formula
    u[1:-1,1:-1] = un[1:-1,1:-1] + dt*kx[1:-1,1:-1]
    v[1:-1,1:-1] = vn[1:-1,1:-1] + dt*ky[1:-1,1:-1]
    # boundary conditions
    u,v = bc.boundary_conditions(bcxu,bcxv,bcyu,bcyv,dy,dx,dt,u,un,v,vn,vel,kn,t,A0,omp)
    # pressure poisson
    p = af.pres_pois_spec(bod,poi_b,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,rho,dt,u,v,eps)
    # Energy equation integration
    if ene == 'on':
        Tx = cdp.con_diff_ene(bod,bcxt,bcyt,ny,nx,dy,dx,dt,un,vn,Tn,nu,Pr,eps)
        T[1:-1,1:-1] = Tn[1:-1,1:-1] + dt * Tx[1:-1,1:-1]
        T = bc.boundary_conditions_tem(bcxt,bcyt,ny,nx,dy,dx,dt,T,Tn,eps,eps_in,T0)
    ###################################################################################
    return u,v,p,T

def CPM(bod,ischeme,poi_b,ene,buo,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,ny,nx,dy,dx,rho,nu,Ri,Pr,dt,u,v,p,T,vel,kn,t,A0,omp,eps,eps_in,T0):
    ############ Chorin's Projection Method ##### Time Integrator ##########################
    un,vn,pn = u.copy(),v.copy(),p.copy()
    Tn = T.copy()
    u_sp,v_sp = np.zeros_like(u),np.zeros_like(v)
    # prediction
    kx,ky = cdp.con_diff_pre(bod,ischeme,buo,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,dt,un,vn,pn,Tn,nu,Ri,eps)
    u_sp[1:-1,1:-1] = un[1:-1,1:-1] + dt*kx[1:-1,1:-1]
    v_sp[1:-1,1:-1] = vn[1:-1,1:-1] + dt*ky[1:-1,1:-1]
    # boundary conditions
    u_sp,v_sp = bc.boundary_conditions(bcxu,bcxv,bcyu,bcyv,dy,dx,dt,u_sp,un,v_sp,vn,vel,kn,t,A0,omp)
    # pressure poisson
    p = af.pres_pois_spec(bod,poi_b,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,rho,dt,u_sp,v_sp,eps)
    # velocity correction
    dpdx = -(p[1:-1,2:]- p[1:-1,:-2])/(2*dx)
    dpdy = -(p[2:,1:-1]- p[:-2,1:-1])/(2*dy)
    u[1:-1,1:-1] = u_sp[1:-1,1:-1] + dt*dpdx/rho
    v[1:-1,1:-1] = v_sp[1:-1,1:-1] + dt*dpdy/rho
    # boundary conditions
    u,v = bc.boundary_conditions(bcxu,bcxv,bcyu,bcyv,dy,dx,dt,u,un,v,vn,vel,kn,t,A0,omp)
    if ene == 'on':
        Tx = cdp.con_diff_ene(bod,bcxt,bcyt,ny,nx,dy,dx,dt,un,vn,Tn,nu,Pr,eps)
        T[1:-1,1:-1] = Tn[1:-1,1:-1] + dt * Tx[1:-1,1:-1]
        T = bc.boundary_conditions_tem(bcxt,bcyt,ny,nx,dy,dx,dt,T,Tn,eps,eps_in,T0)
    #######################################################################################
    return u,v,p,T
    
def AB2(bod,ischeme,poi_b,ene,buo,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,ny,nx,dy,dx,rho,nu,Ri,Pr,dt,u,un,v,vn,p,pn,T,Tn,vel,kn,t,A0,omp,eps,eps_in,T0):
    ########## Adam-Bashforth 2nd order ############ Time Integrator ##########################
    un1,vn1,pn1 = un.copy(),vn.copy(),pn.copy()
    Tn1 = Tn.copy()
    un,vn,pn = u.copy(),v.copy(),p.copy()
    Tn = T.copy()
    # slope calculations
    kx,ky = cdp.con_diff_pre(bod,ischeme,buo,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,dt,un,vn,pn,Tn,nu,Ri,eps)
    kx1,ky1 = cdp.con_diff_pre(bod,ischeme,buo,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,dt,un1,vn1,pn1,Tn1,nu,Ri,eps)
    # Adam bashforth formula
    u[1:-1,1:-1] = un[1:-1,1:-1] + (dt/2)*(3*kx[1:-1,1:-1]  - kx1[1:-1,1:-1])
    v[1:-1,1:-1] = vn[1:-1,1:-1] + (dt/2)*(3*ky[1:-1,1:-1]  - ky1[1:-1,1:-1])
    # boundary conditions
    u,v = bc.boundary_conditions(bcxu,bcxv,bcyu,bcyv,dy,dx,dt,u,un,v,vn,vel,kn,t,A0,omp)
    # pressure poisson
    p = af.pres_pois_spec(bod,poi_b,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,rho,dt,u,v,eps)
    if ene == 'on':
        Tx = cdp.con_diff_ene(bod,bcxt,bcyt,ny,nx,dy,dx,dt,un,vn,Tn,nu,Pr,eps)
        Tx1 = cdp.con_diff_ene(bod,bcxt,bcyt,ny,nx,dy,dx,dt,un1,vn1,Tn1,nu,Pr,eps)
        T[1:-1,1:-1] = Tn[1:-1,1:-1] + (dt/2)*(3*Tx[1:-1,1:-1] - Tx1[1:-1,1:-1])
        T = bc.boundary_conditions_tem(bcxt,bcyt,ny,nx,dy,dx,dt,T,Tn,eps,eps_in,T0)
        
    ##########################################################################################
    return u,v,p,T,un,vn,pn,Tn

          
def RK4(bod,ischeme,poi_b,bcx,bcy,ny,nx,dy,dx,rho,nu,dt,u,v,p,vel,kn,t,A0,omp,eps):
    ######################### RK4 ########## Time Integrator #################################
    un,vn,pn = u.copy(),v.copy(),p.copy()
    # slope calculation
    k1x,k1y = cdp.con_diff_pre(bod,ischeme,bcx,bcy,ny,nx,dy,dx,dt,un,vn,pn,nu,eps)
    k2x,k2y = cdp.con_diff_pre(bod,ischeme,bcx,bcy,ny,nx,dy,dx,dt,un+dt*k1x*0.5,vn+dt*k1y*0.5,pn,nu,eps)
    k3x,k3y = cdp.con_diff_pre(bod,ischeme,bcx,bcy,ny,nx,dy,dx,dt,un+dt*k2x*0.5,vn+dt*k2y*0.5,pn,nu,eps)
    k4x,k4y = cdp.con_diff_pre(bod,ischeme,bcx,bcy,ny,nx,dy,dx,dt,un+dt*k3x,vn+dt*k3y*0.5,pn,nu,eps)
    con_dif_u,con_dif_v = (1/6)*(k1x+2*k2x+2*k3x+k4x),(1/6)*(k1y+2*k2y+2*k3y+k4y)
    # RK4 formula
    u[1:-1,1:-1] = un[1:-1,1:-1] + dt*con_dif_u[1:-1,1:-1]
    v[1:-1,1:-1] = vn[1:-1,1:-1] + dt*con_dif_v[1:-1,1:-1]
    # boundary conditions
    u,v = bc.boundary_conditions(bcx,bcy,dy,dx,dt,u,un,v,vn,vel,kn,t,A0,omp)
    # pressure poisson
    p = af.pres_pois_spec(bod,poi_b,bcx,bcy,ny,nx,dy,dx,rho,dt,u,v,eps)
    ###########################################################################################
    return u,v,p
        

