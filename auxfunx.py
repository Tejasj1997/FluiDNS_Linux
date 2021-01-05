#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
from scipy.sparse import find
import sys
from uvw import RectilinearGrid, DataArray
import pade_compact as pc
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from numba import jit, njit

######################################################
# Its a magicbox for all required auxiliary function.#
######################################################

def aero(ny,nx,p,eps,rho,u_vel):
    p_chop = p*eps
    a,b,P = find(p_chop)
    P = P.reshape((a.max()-a.min()+1,b.max()-b.min()+1))
    cp = (P-p[int(ny/2),int(nx/4)])/(0.5*rho*u_vel**2)
    cplo,cpup,cple,cpri = cp[0,:],cp[-1,:],cp[:,0],cp[:,-1]
    cl = np.mean(cpup) - np.mean(cplo)
    cd = np.mean(cple) - np.mean(cpri)
    cp_mean = np.mean(cp)
    return cl,cd,cp_mean

def particle_tracks(ptrax,ptray,u,v,dx,dy,dt):
    xl,yl = ptrax[-1],ptray[-1]
    ul,vl = u[int(xl/dx),int(yl/dy)],v[int(xl/dx),int(yl/dy)]
    ptrax,ptray = np.append(ptrax,xl + ul*dt),np.append(ptray,yl + vl *dt)
    return ptrax,ptray
    
def VIV(ypos,yvel,cl,k,dt):
    yddot = (cl/2) - k*ypos
    ydot = yvel + dt*yddot
    y = ypos + (dt) * (1.2*ydot)

    return y,ydot

def vtr_exp(sdi,i,L,H,ny,nx,u,v,p,vort,T):
    sivgt = np.zeros((ny,nx))
    dx,dy = L/(nx-1),H/(ny-1)
    sivgt[1:-1,1:-1] = ((u[1:-1,2:]-u[1:-1,:-2])/(dx**2))*((v[2:,1:-1]-v[:-2,1:-1])/(dy**2)) - ((u[2:,1:-1]-u[:-2,1:-1])/(dy**2))*((v[1:-1,2:]-v[1:-1,:-2])/(dx**2))
    # Creating coordinates
    y = np.linspace(0, L, nx)
    x = np.linspace(0, H, ny)
    
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    original_stdout = sys.stdout
    with open(sdi+'/out_it'+str(i+1)+'.vtr','w') as f:
        sys.stdout = f
        with RectilinearGrid(sys.stdout, (x, y)) as grid:
            grid.addPointData(DataArray(u, range(2), 'u velocity'))
            grid.addPointData(DataArray(v, range(2), 'v velocity'))
            grid.addPointData(DataArray(p, range(2), 'pressure'))
            grid.addPointData(DataArray(vort, range(2), 'vorticity'))
            grid.addPointData(DataArray(T, range(2), 'temperature'))
            grid.addPointData(DataArray(sivgt, range(2), 'SIVGT'))
        sys.stdout = original_stdout

def pres_pois_spec(bod,poi_b,bcxu,bcxv,bcyu,bcyv,ny,nx,dy,dx,rho,dt,u,v,eps=None):
    RHS = np.ones([ny,nx])
    if bod == 'IBM':
        u,v = (1-eps)*u,(1-eps)*v
        
    RHS = (pc.dx(u,ny,nx,dy,dx,bcxu) + pc.dy(v,ny,nx,dy,dx,bcyv))*(rho/dt)
    # FFT coefficients
    kp = 2*np.pi*np.fft.fftfreq(nx,d=dx)
    om = 2*np.pi*np.fft.fftfreq(ny,d=dy)
    kx,ky = np.meshgrid(kp,om)
    delsq = -(kx**2 + ky**2)
    delsq[0,0] = 1e-6
        
    # FFT integration
    # For periodic in axis=0 i.e. y-axis and dirichilet/pressure neumann in x-axis for inflow and outflow
    if poi_b == 'x dirichilet':
        RHS_hat = dct(np.fft.fft(RHS,axis=1),type = 1,axis = 0)
        p = RHS_hat*dt/(delsq)
        p = np.fft.ifft(dct(p,type=1,axis=0)/(2*(ny+1)),axis=1).real
        p = p - p[0,0]
            
    elif poi_b == 'double periodic':
        # For double periodic domain
        RHS_hat = np.fft.fft2(RHS)
        p = RHS_hat/delsq
        p = np.fft.ifft2(p).real
        p = p - p[0,0]                           
        
    return p

def pltt(u,v,p,vort,T,ene,i,title):
    fig = plt.figure(figsize=(20,10))
    plt.imshow(np.flipud(u),interpolation='gaussian')
    plt.colorbar()
    plt.xlabel('x-grid')
    plt.ylabel('y-grid')
    plt.title('Contours of u-velocity')
    plt.savefig(str(title)+'/u_vel_it'+str(i+1)+'.png')
    plt.close()

    fig1 = plt.figure(figsize = (20,10))
    plt.imshow(np.flipud(vort),interpolation='gaussian',vmin=-4,vmax=4)
    plt.colorbar()
    plt.xlabel('x-grid')
    plt.ylabel('y-grid')
    plt.title('Contours of vorticity')
    plt.savefig(str(title)+'/vort_it'+str(i+1)+'.png')
    plt.close()

    fig2 = plt.figure(figsize=(20,10))
    plt.imshow(np.flipud(v),interpolation='gaussian')
    plt.colorbar()
#     plt.grid()
    plt.xlabel('x-grid')
    plt.ylabel('y-grid')
    plt.title('Contours of v-vel')
    plt.savefig(str(title)+'/v_vel_it'+str(i+1)+'.png')
    plt.close()
    
    fig3 = plt.figure(figsize=(20,10))
    plt.imshow(np.flipud(p),interpolation='gaussian')
    plt.colorbar()
#     plt.grid()
    plt.xlabel('x-grid')
    plt.ylabel('y-grid')
    plt.title('Contours of pressure')
    plt.savefig(str(title)+'/pressure_it'+str(i+1)+'.png')
    plt.close()
    
    if ene == 'on':
        fig4 = plt.figure(figsize=(20,10))
        plt.imshow(np.flipud(T),interpolation='gaussian')
        plt.colorbar()
#         plt.grid()
        plt.xlabel('x-grid')
        plt.ylabel('y-grid')
        plt.title('Contours of Temperature')
        plt.savefig(str(title)+'/temperature_it'+str(i+1)+'.png')
        plt.close()




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
