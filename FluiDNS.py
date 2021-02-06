#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import source.auxfunx as af
from source.simulator import FluiDNS
import time
import os
import sys

SimTit = 'code_dev_test'
SimDes = 'Testing of vortex induced vibrations with blending factor = 1'

# creating save directory and cleanup
save_dir = '/home/tejas/project_code_dev/'+str(SimTit)

# restart parameter
res = 'new'
itr = 30000

if os.path.exists(save_dir) and res == 'new':
    clean = input("Do you want to clean the existing directory? [y/n]  ")
    if clean == 'y' or clean == 'yes':
        os.system("rm -rf {}".format(save_dir))
        print("Directory cleaned. Making new one.")
        os.mkdir(save_dir)
    if clean == 'n' or clean == 'no':
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
    
# plotting option
pltop = 'paraview'
ptr = 'off'            # calculation of particle tracks

# saving interval
isave = 50

# Domain setup
L,H = 26,12             # Domain Size
nx,ny = 600,300 # grid size
dx,dy = L/(nx-1),H/(ny-1)  # spatial step size

# solver setup
ischeme = 'CPM'                  # Time integration scheme (RK4,RK1,CPM,AB2 are available.)
pois_bound = 'double periodic'   # Spectral boundary conditions over pressure
bcxu = 'inout'
bcxv = 'inout'
bcyu = 'inout'
bcyv = 'noslip'

# simulation properties
rho = 1        # Density
Re = 100      # Free-stream Reynolds number
CFL = 0.3     # CFL condition to be obeyed

# Flow setups
u_vel = 1     # Free stream velocity
kn = 0.01      # Knudson number for slip flow at boundary
A0 = 1         # amplitude for in-line oscillatory flow
omp = 10       # Frequency for in-line oscillatory flow

# IBM setup
bod = 'IBM'        # IBM specification parameter
shape = 'square'   # Solid body shape (Square and circle are available.) 

if shape == 'square':
    # parameters for square body
    x_pos,y_pos = 6,5.5   # Left botton corner position inside domain     
elif shape == 'circle':
    # paramters for circular body
    x_pos,y_pos = 6,6    # Centre position inside domain

length,width = 1,1    # Size of square or rectangle
radius = 0.5
# Energy equation parameters and specifiers
ene = 'on'       # Energy equation on or off                   
buo = 'off'      # Buoyancy effects using Boussinesq approximation on or off
bcxt = 'inout'   # Temp boundary - adiab for adiabatic boundary
bcyt = 'inout'   # Temp boundary - Isoth for isothermal wall condition
T0 = 1           # Constant temperature at solid boundary
Ri = 1        # Richardson number for the case
Pr = 0.71         # Prandtl number for fluid

# time setup
nt_st = 0         # Starting Iteration
nt_en = 500     # Ending iteration

# oscillating object setup
mot = 'FIV'
yvel = 0
k = 0.001
amp = 1.0
fn = 1.5
st = 0.141

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


#### Displaying the setup #############
original_stdout = sys.stdout
with open(save_dir +'/setup.txt','w') as f:
    sys.stdout = f
    print('######### Simulation Title #######################')
    print('Simulation Title : ' + str(SimTit))
    print('Simulation Description : ' +str(SimDes))
    print('######### Time and Domain config #################')
    print('Time integration scheme : ' + str(ischeme))
    print('Domain Size : L = ' + str(L) + ' H = '+ str(H))
    print('Grid size : nx = ' +str(nx) + ' ny = ' + str(ny))
    print('Step size : dx = ' +str(dx) + ' dy = ' + str(dy))
    print('Flow Reynolds number : ' +str(Re))
    print('######### Velocity boundary config #################')
    print('x bound on u-vel : ' + str(bcxu))
    print('x bound on v-vel : ' + str(bcxv))
    print('y bound on u-vel : ' + str(bcyu))
    print('y bound on v-vel : ' + str(bcyv))
    print('Time step : ' + str(dt))
    print('Start time : ' + str(nt_st*dt))
    print('End time : '+str(nt_en*dt))
    if bod == 'IBM':
        print('######### Solid Body config ########################')
        print('Solid shape : ' + str(shape))
        if shape == 'square':
            print('location : x = ' + str(x_pos) + ' : y = '+str(y_pos))
            print('Size of solid : length = ' + str(length) +' : width = '+str(width))
        if shape == 'circle':
            print('location of centre : x = ' + str(x_pos) + ' : y = '+str(y_pos))
            print('Radius of circle : '+str(radius))
    if ene == 'on':
        print('######## Energy equation config #################')
        print('Energy equation : ' + str(ene))
        print('Temperature at body : ' + str(T0))
        print('Prandtl number : ' + str(Pr))
        if buo =='on':
            print('Richardson number : '+ str(Ri))
        print('x temp boundary : ' +str(bcxt))
        print('y temp boundary : ' +str(bcyt))
    print('######################################################')
    sys.stdout = original_stdout
print('Starting Iterations Now')

start = time.time()
sim = FluiDNS(L,H,nx,ny,ene,res,save_dir,itr)
sim.parameters(SimTit,save_dir,pltop,isave,Re,Ri,Pr,bod,ene,buo,ischeme,pois_bound,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,rho,dt,u_vel,kn,A0,T0,omp,shape,x_pos,y_pos,length,width,radius,ptr,mot,k,yvel,amp,fn,st)
u,v,p,vort,T = sim.solver(nt_st,nt_en)
end  = time.time()-start


# end and final time calculations
if end < 60:
    print("total time taken = " +str(end) + " secs")
elif end > 60:
    print("total time taken = " +str(end/60) + " mins")

