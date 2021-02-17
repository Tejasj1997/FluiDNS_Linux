#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def boundary_conditions(bcxu,bcxv,bcyu,bcyv,dy,dx,dt,u,un,v,vn,vel,t,A0,omp):
    ######################### x bound on u-velocity ##################################
    if bcxu == 'inout':
        u[:,-1] = un[:,-1] - (dt/(2*dx))*(un[:,-1]*un[:,-1]-un[:,-2]*un[:,-2])
        u[:,0] = vel
#         u[:,0] = u[:,1]
            
    elif bcxu == 'inout-os':
        u[:,0] = vel + A0*np.cos(omp*t)
        u[:,-1] = un[:,-1] - (dt/(2*dx))*(un[:,-1]*un[:,-1]-un[:,-2]*un[:,-2])
        u[int(ny/2),-1] = vel
        
    elif bcxu == 'noslip':
        u[:,-1],u[:,0] = 0,0
    #################################################################################
    
    ######################### x bound on v-velocity ##################################
    if bcxv == 'inout':
        v[:,-1],v[:,0] = vn[:,-1] - un[:,-1]*(dt/(2*dx))*(vn[:,-1] - vn[:,-2]),0
            
    elif bcxv == 'inout-os':
        v[:,-1],self.v[:,0] = v[:,-2],0
        
    elif bcxv == 'noslip':
        v[:,-1],v[:,0] = 0,0
    #################################################################################
    
    ############################ y bound on u-velocity #############################
    if bcyu == 'noslip':
        u[0,:],u[-1,:] = 0,0
        
    elif bcyu == 'slip':
        u[0,:],u[-1,:] =  u[1,:],u[-2,:]
    #######################################################################################
    
    ############################ y bound on v-velocity #############################
    if bcyv == 'noslip':
        v[0,:],v[-1,:] = 0,0
        
    elif bcyv == 'inout':
        v[0,:],v[-1,:]  = vel,v[-2,:]
        
    elif bcyv == 'slip':
        v[0,:],v[-1,:] = v[1,:],v[-2,:]
    #######################################################################################
        
    return u,v                               

def boundary_conditions_tem(bcx,bcy,ny,nx,dy,dx,dt,T,Tn,eps,eps_in,T0):
    if bcx == 'inout':
        T[:,-1] = T[:,-2]
        T[:,0] = 0
    elif bcx == 'isoth':
        T[:,-1] = 0
        T[:,0] = 0
        
    if bcy == 'inout':
        T[0,:] = T[1,:]
        T[-1,:] = T[-2,:]
    elif bcy == 'isoth':
        T[0,:] = 0
        T[-1,:] = 0
        
#     if bod == 'IBM':
#         T = T*eps_in + eps*T0
        
    return T
