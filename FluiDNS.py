#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import source.auxfunx as af
from source.simulator import FluiDNS
import time
import os
import sys

par = np.loadtxt('params.txt',dtype='str',delimiter=',')
parf = []
for i in range(len(par)-2):
    parf.append(float(par[i+2]))
    
SimTit = par[0]

# creating save directory and cleanup
work_dir = par[1]
save_dir = work_dir +str(SimTit)

# restart parameter
if parf[0] == 1:
    res = 'new'
elif parf[0] == 0:
    res = 'restart'
itr = parf[1]

if os.path.exists(save_dir) and res == 'new':
    clean = input("Do you want to clean the existing directory? [y/n]  ")
    if clean == 'y' or clean == 'yes':
        os.system("rm -r {}".format(save_dir))
        print("Directory cleaned. Making new one.")
        os.mkdir(save_dir)
#         os.system("cp {} {}".format('params.txt',save_dir)  
                  
    elif clean == 'n' or clean == 'no':
        print("")
        print("###################################################################################")
        print("")
        print("WARNING: Permission to clean exixting directory is not granted. ")
        print("Please change the save directory before running again and backup the files first.")
        print("Exiting the code now.")
        print("")
        print("###################################################################################")
        print("")
        exit()
        
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#     os.system("cp {} {}".format(str('params.txt'),save_dir)
              
# saving interval
isave = int(parf[2])
idata = int(parf[3])
if parf[4] == 1:
    aeros = 'on'
else:
    aeros = 'off'
    
# Domain setup
L,H = parf[5],parf[6]             # Domain Size
nx,ny = int(parf[7]),int(parf[8]) # grid size
dx,dy = L/(nx),H/(ny)  # spatial step size

# solver setup
if parf[15] == 0:
    ischeme = 'CPM'
elif parf[15] == 1:
    ischeme = 'RK1'
elif parf[15] == 2:
    ischeme = 'AB2'

if parf[16] == 0:
    pois_bound = 'double periodic'
elif parf[16] == 1:
    pois_bound = 'x dirichilet'
    
if parf[17] == 0:
    bcxu = 'inout'
elif parf[17] == 1:
    bcxu= 'inout-os'
elif parf[17] == 2:
    bcxu = 'noslip'
elif parf[17] == 3:
    bcxu = 'parab'
    
if parf[18] ==0:
    bcxv = 'inout'
elif parf[18] == 1:
    bcxv = 'inout-os'
elif parf[18] == 2:
    bcxv = 'noslip'

if parf[19] == 0:
    bcyu = 'slip'
elif parf[19] == 1:
    bcyu = 'noslip'
    
if parf[20] == 0:
    bcyv = 'inout'
elif parf[20] == 1:
    bcyv = 'noslip'
elif parf[20] == 2:
    bcyv = 'slip'

# simulation properties
rho = 1        # Density
Re = parf[9]      # Free-stream Reynolds number
CFL = parf[12]     # CFL condition to be obeyed

# Flow setups
u_vel = parf[21]     # Free stream velocity
A0 = parf[22]         # amplitude for in-line oscillatory flow
omp = parf[23]       # Frequency for in-line oscillatory flow

# IBM setup
if parf[24] == 1:
    bod = 'IBM'
else:
    bod = 'off'

length,width = parf[26],parf[26]    # Size of square or rectangle
radius = parf[26]/2    
if parf[25] == 0:
    shape = 'circle'
    x_pos,y_pos = parf[27],parf[28] 
elif parf[25] == 1:
    shape = 'square'
    x_pos,y_pos = parf[27],parf[28] - (width/2)
    

# Energy equation parameters and specifiers
if parf[33] == 1:
    ene = 'on'  
else:
    ene = 'off'
    
if parf[34] == 1:    
    buo = 'on'
else:
    buo = 'off'

if parf[35] == 0:
    bcxt = 'inout'  
elif parf[35] ==1:
    bcxt = 'isoth'
elif parf[35] == 2:
    bcxt = 'parab'
    
if parf[36] == 0:    
    bcyt = 'inout'
elif parf[36] == 1:
    bcyt = 'isoth'
elif parf[36] == 2:
    bcyt = 'adiab'
    
T0 = parf[37]           
Ri = parf[11]        # Richardson number for the case
Pr = parf[10]         # Prandtl number for fluid

# time setup
nt_st = int(parf[13])         # Starting Iteration
nt_en = int(parf[14])     # Ending iteration

# oscillating object setup
if parf[29] == 1:
    mot = 'FIV'
else:
    mot = 'off'
    
amp = parf[30]
fn = parf[31]
st = parf[32]

# Time step based on CFL calculations and time step recommender
if ene == 'on':
    dt = CFL*min(1/Re,6/(Pr*Re))
elif ene == 'off':
    dt = CFL/Re
    
if dt < 0.001:
    dt = round(dt,4)
elif dt < 0.01 and dt > 0.001:
    dt  = round(dt,3)
    
print("###################################################################################")
print('')
print('The system based on the present time scales and conservative CFL = '+str(CFL)+' in the simulation is recommending time step to be '+str(dt))
print('')
print("###################################################################################")
dt_acc = input("Do you want to continue with recommended time step?  [y/n]   ")
if dt_acc == 'no' or dt_acc == 'n':
    print('Please set the time step!')
    dt = float(input('dt = '))


start = time.time()
sim = FluiDNS(L,H,nx,ny,ene,res,save_dir,itr)
sim.parameters(SimTit,save_dir,isave,idata,aeros,Re,Ri,Pr,bod,ene,buo,ischeme,pois_bound,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,rho,dt,u_vel,A0,T0,omp,shape,x_pos,y_pos,length,width,radius,mot,amp,fn,st)
u,v,p,vort,T = sim.solver(nt_st,nt_en)
end  = time.time()-start


# end and final time calculations
if end < 60:
    print("total time taken = " +str(end) + " secs")
elif end > 60:
    print("total time taken = " +str(end/60) + " mins")

