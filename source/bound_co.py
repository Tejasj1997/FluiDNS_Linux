#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

def boundary_conditions(bcxu,bcxv,bcyu,bcyv,L,H,dy,dx,dt,u,un,v,vn,vel,t,A0,omp):
    ######################### x bound on u-velocity ##################################
    if bcxu == 'inout':
#         u[:,-1] = un[:,-2] - (vel*dt/dx)*(un[:,-2]-un[:,-3])
        u[:,0] = vel
#         u[:,0] = u[:,1]
        u[:,-1] = u[:,-2] + dx*(u[:,-2] - u[:,-3])
            
    elif bcxu == 'inout-os':
        u[:,0] = vel + A0*np.cos(omp*t)
        u[:,-1] = un[:,-1] - (dt/(2*dx))*(un[:,-1]*un[:,-1]-un[:,-2]*un[:,-2])
        u[int(ny/2),-1] = vel
        
    elif bcxu == 'noslip':
        u[:,-1],u[:,0] = 0,0
    
    elif bcxu == 'parab':
        y = np.linspace(0,H/2,int(u.shape[0]/2))
        uprof = vel * 1.5 * (1- 4 *(y/H)**2)
        uprof1 = np.append(np.flip(uprof),uprof)
        u[:,0] = uprof1
        u[:,-1] = u[:,-2] + dx*(u[:,-2] - u[:,-3])
#         u[:,-1] = un[:,-1] - (dt/(2*dx))*(un[:,-1]*un[:,-1]-un[:,-2]*un[:,-2])
    #################################################################################
    
    ######################### x bound on v-velocity ##################################
    if bcxv == 'inout':
        v[:,-1],v[:,0] = v[:,-2] + dx*(v[:,-2] - v[:,-3]),0
            
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

def boundary_conditions_tem(bcx,bcy,ny,nx,dy,dx,dt,T,Tn,eps,eps_in,T0,q):
    if bcx == 'inout':
#         T[:,-1] = T[:,-2]# + dx*(T[:,-2] - T[:,-3])
        T[:,-1] = T[:,-3] - (dt/(2*dx))*(Tn[:,-1]*Tn[:,-1]-Tn[:,-2]*Tn[:,-2])
        T[:,0] = 0
    elif bcx == 'isoth':
        T[:,-1] = 0
        T[:,0] = 0
    elif bcx == 'parab':
        y = np.linspace(0,H/2,int(u.shape[0]/2))
        uprof = 1 - (1- 4 *(y/H)**2)
        uprof1 = np.append(uprof,np.flip(uprof))
        T[:,0] = uprof1
        T[:,-1] = Tn[:,-1] - (dt/(2*dx))*(Tn[:,-1]*Tn[:,-1]-Tn[:,-2]*Tn[:,-2])
        
    if bcy == 'inout':
        T[0,:] = T[1,:]
        T[-1,:] = T[-2,:]
    elif bcy == 'isoth':
        T[-1,:] = 1
        T[0,:] = 0.5
    elif bcy == 'adiab':
        T[0,:] = T[1,:] + 0.5*dy
        T[-1,:]= T[-2,:] + 1*dy
        
    return T
