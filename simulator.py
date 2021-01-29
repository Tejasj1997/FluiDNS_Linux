#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import time_integrators as TIM
import time
import IBM
import auxfunx as af

class FluiDNS():
    # Initialization constructor
    def __init__(self,L,H,nx,ny,ene,restart,save_dir,itr):
        self.L,self.H,self.nx,self.ny= L,H,nx,ny
        self.dx,self.dy = self.L/(self.nx-1),self.H/(self.ny-1)
        self.declarations(ny,nx,restart,itr,save_dir,ene)
    
    ##########################################################################################
    # A function to populate all required parameters used in simulation as a class attribute
    # to be available throughout the class anywhere
    ##########################################################################################
    def parameters(self,SimTit,save_dir,pltop,isave,Re,Ri,Pr,bod,ene,buo,schema,pois_bound,bcxu,bcxv,bcyu,bcyv,bcxt,bcyt,rho,dt,vel,kn,A0,T0,omp,shape,x_pos,y_pos,length,width,radius,ptr,mot,k,yvel,amp,fn,st):
        self.title = SimTit
        self.sdi = save_dir
        self.pto = pltop
        self.isave = isave
        self.Re = Re
        self.dt,self.rho,self.nu = dt,rho,1/self.Re
        self.ischeme = schema
        self.vel = vel
        self.poi_b = pois_bound
        self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bod = bcxu,bcxv,bcyu,bcyv,bod
        self.A0,self.omp = A0,omp
        self.x_pos,self.y_pos,self.length,self.width = x_pos,y_pos,length,width
        self.kn = kn
        self.shape = shape
        self.ra = radius
        self.ene,self.T0 = ene,T0
        self.Ri,self.Pr = Ri,Pr
        self.buo = buo
        self.bcxt,self.bcyt = bcxt,bcyt
        self.part_track = ptr
        self.k,self.motion,self.y_vel = k,mot,yvel
        self.amp,self.fn,self.st = amp,fn,st
    
    ########################################################################################################
    # Declaring all required arrays for starting the simulation based on the fresh simulation or a restart
    ########################################################################################################
    def declarations(self,ny,nx,restart,itr,save_dir,ene):
        self.x = np.linspace(0, self.L, nx)
        self.y = np.linspace(0, self.H, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        if restart == 'new':
            self.u,self.v,self.p = np.zeros([ny,nx]),np.zeros([ny,nx]),np.zeros([ny,nx])
            self.vort = np.zeros([ny,nx])
            self.T = np.zeros([ny,nx])

        elif restart == 'restart':
            print('It is a restart...')
            self.u,self.v = np.fromfile(save_dir+'/out_it' + str(itr)+ '_u.dat'),np.fromfile(save_dir+'/out_it' + str(itr)+ '_v.dat')
            self.p = np.fromfile(save_dir+'/out_it' + str(itr)+ '_p.dat')
            self.vort = np.fromfile(save_dir+'/out_it' + str(itr)+ '_vort.dat')
            
            self.u,self.v,self.p = self.u.reshape((ny,nx)),self.v.reshape((ny,nx)),self.p.reshape((ny,nx)) 
            self.vort = self.vort.reshape((ny,nx))
#             self.T = np.zeros([ny,nx])
            self.T = np.fromfile(save_dir +'/out_it' + str(itr)+ '_temp.dat')
            self.T = self.T.reshape((ny,nx))
            
               
    def solver(self,nt_st,nt_en):
        # time initialization
        t,itr = nt_st*self.dt,0
        # time step calculation
        nt = nt_en-nt_st
        ntn = nt_st
        print("total iterations = " +str(nt))
        print("Start time = "+str(nt_st*self.dt))
        print("End time = "+str(nt_en*self.dt))
        est_time = int(0.2*nt)
        t_rec = np.zeros([est_time,1])
        pro_u,pro_v,pro_p = np.array([]),np.array([]),np.array([])
        pro_vo,Nur = np.array([]),np.array([])
        cpr,clr,cdr = np.array([]),np.array([]),np.array([])
#         ptrax,ptray = np.array([0]),np.array([6])
        ypos = np.array([self.y_pos])
        yvel = np.array([self.y_vel])
        
        for i in range(nt):
            st_t = time.time()
            ntn += 1
            t = ntn*self.dt
            
            # IBM Mask formulation
            if self.motion == "VIV" or self.motion == 'FIV' or self.bod == 'IBM':
                self.y_pos = ypos[-1]
                if self.shape == 'square':
                    eps,eps_in = IBM.square(self.motion,self.ene,self.ny,self.nx,self.dy,self.dx,self.x_pos,self.y_pos,self.length,self.width,self.amp,self.st,self.fn,t)
                elif self.shape == 'circle':
                    eps,eps_in = IBM.circle(self.ene,self.ny,self.nx,self.dy,self.dx,self.x_pos,self.y_pos,self.ra,t)

            ######### Time Integrators come here  #############################
            if self.ischeme == 'RK1':
                self.u,self.v,self.p,self.T = TIM.RK1(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                 self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt,self.u,self.v,self.p,self.T,self.vel,self.kn,t,self.A0,self.omp,eps,eps_in,self.T0)
                
            elif self.ischeme == 'RK4':
                self.u,self.v,self.p = TIM.RK4(self.bod,self.ischeme,self.poi_b,self.bcx,self.bcy,                                                self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.dt                                      ,self.u,self.v,self.p,self.vel,self.kn,t,self.A0,self.omp,eps)
                
            elif self.ischeme == 'CPM':
                self.u,self.v,self.p,self.T = TIM.CPM(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                                                self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt                                                ,self.u,self.v,self.p,self.T,self.vel,self.kn,t,self.A0,self.omp                 ,eps,eps_in,self.T0)
                
            elif self.ischeme == 'AB2':
                if i == 0:
                    un,vn,pn,Tn = TIM.RK1(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                                                self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt                                                ,self.u,self.v,self.p,self.T,self.vel,self.kn,t,self.A0,self.omp                                               ,eps,eps_in,self.T0)
                    
                else:
                    self.u,self.v,self.p,self.T,un,vn,pn,Tn = TIM.AB2(self.bod,self.ischeme,self.poi_b,self.ene,self.buo,self.bcxu,self.bcxv,self.bcyu,self.bcyv,self.bcxt,self.bcyt,                                                self.ny,self.nx,self.dy,self.dx,self.rho,self.nu,self.Ri,self.Pr,self.dt                                                ,self.u,un,self.v,vn,self.p,pn,self.T,Tn,self.vel,self.kn,t,self.A0,self.omp                                               ,eps,eps_in,self.T0)
            
            ###################################################################
            # vorticity field calculation
            self.vort[1:-1,1:-1] = (self.v[1:-1,2:]-self.v[1:-1,:-2])/(2*self.dx) - (self.u[2:,1:-1]-self.u[:-2,1:-1])/(2*self.dy)
           
            # probing velocities to calculate shedding frequncies
            pro_u,pro_v,pro_p = np.append(pro_u,self.u[int(self.ny/2),int(self.nx/2)]),np.append(pro_v,self.v[int(self.ny/2),int(self.nx/2)]),np.append(pro_p,self.p[int(self.ny/2),int(self.nx/2)])
            pro_vo = np.append(pro_vo,self.vort[int(self.ny/2),int(self.nx/2)])
            
            # calculation of aero coeffs and recording
            
            cl,cd,cp = af.aero(self.ny,self.nx,self.p,eps,self.rho,self.vel)
            clr,cdr,cpr = np.append(clr,cl),np.append(cdr,cd),np.append(cpr,cp)
            
            if self.ene == 'on':
                Nu = af.nusnum(self.T,self.T0,eps,self.dx,self.dy)
                Nur = np.append(Nur,Nu)
            
            if self.motion == 'VIV':
                y,ydot = af.VIV(ypos[-1],yvel[-1],cl,self.k,self.dt)
                ypos,yvel = np.append(ypos,y),np.append(yvel,ydot)
                
            # time estimation lines
            if i < est_time:
                en_t = time.time() - st_t
                t_rec[i,0] = en_t
            if i == est_time:
                ti = np.average(t_rec[:,0])*nt
                if ti > 60:
                    ti = ti/60
                    print("ETC = "+str(ti)+ ' mins')
                elif ti < 60:
                    print("ETC = "+str(ti)+ ' secs')  
                    
            if (i+1)%self.isave == 0:
                if self.pto == 'paraview':
                    af.vtr_exp(self.sdi,nt_st+i,self.L,self.H,self.ny,self.nx,self.u,self.v,self.p,self.vort,self.T)
                elif self.pto == 'builtin':
                    af.pltt(self.u,self.v,self.p,self.vort,self.T,self.ene,nt_st+i,self.sdi)
                    
            ### CFL and inertia time scale check #############################
            C_til = self.dt *(self.u.max()/self.dx + self.v.max()/self.dy)
            ##################################################################
                    
                    
            # print displays and progression show
            print('##################################################################')
            print('')
            print('Max u vel = ' + str(round(self.u.max(),4)))
            print('Max v vel = ' + str(round(self.v.max(),4)))
            print('Max vorticity = ' + str(round(self.vort.max(),4)))
            if self.ene == 'on':
                print('Max temperature = ' + str(round(self.T.max(),4)))
            print('')
            print('Courant number = '+str(round(C_til,4)))
            if C_til > 0.2:
                print('Warning: Inertial timescale being violated. Solution may diverge.')
            if self.motion == 'VIV':
                print('y position of object = ' + str(ypos[-1]))
            print('')
            print('Right now ' + str(nt_st + i) + ' iterations done, ' + str(nt-i) + ' more to go.')
            print('Time Elapsed is '+str(t) + ' secs.')
            print('')
            print('##################################################################')
            
#             if self.part_track == "on" and ptrax[-1] < self.L and ptray[-1] < self.H:
#                 ptrax,ptray = af.particle_tracks(ptrax,ptray,self.u,self.v,self.dx,self.dy,self.dt)
#                 if ptrax[-1] > self.L or ptray[-1] > self.H:
#                     np.savetxt(self.sdi+'/p1_x.dat',ptrax,delimiter=',')
#                     np.savetxt(self.sdi+'/p1_y.dat',ptray,delimiter=',')
#                     exit()
            
            if (i+1)%5000 == 0:    
                # Saving fields at the end of time loop end for restarting anytime after that
                self.u.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+'_u.dat')
                self.v.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_v.dat')
                self.p.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_p.dat')
                self.vort.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_vort.dat')
                self.T.tofile(self.sdi+'/out_it' + str(nt_st+i+1)+ '_temp.dat')
                    
            if self.vort.max() > 100 or self.T.max() > 100:
                print('##########################################################')
                print('Divergence Detected. Exiting Simulation.')
                print('##########################################################')
                exit()
        
        np.savetxt(self.sdi+'/pro_u.dat',pro_u,delimiter=',')
        np.savetxt(self.sdi+ '/pro_v.dat',pro_v,delimiter=',')
        np.savetxt(self.sdi+'/pro_p.dat',pro_p,delimiter=',')
        np.savetxt(self.sdi+'/clr.dat',clr,delimiter=',')
        np.savetxt(self.sdi+'/cdr.dat',cdr,delimiter=',')
        np.savetxt(self.sdi+'/cpr.dat',cpr,delimiter=',')
        np.savetxt(self.sdi+'/VIV_displacement2.dat',ypos,delimiter=',')
        np.savetxt(self.sdi+'/VIV_velocity2.dat',yvel,delimiter=',')
        return self.u,self.v,self.p,self.vort,self.T
            

