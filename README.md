# FluiDNS
2D DNS code in Python 

A fast DNS code in Python for the simulation of flows over the solid objects with and without motion of objects. The code also includes the solution of energy equation with bousseinesq approximation for natural and mixed convection studies. The code has 4th order spatial accuracy and 1st and 2nd order time accuracy using RK1 and Chorin's projection method and Adam-Bashforth second order method. 4th order accurate semi-implicit compact finite difference scheme is used for fast and accurate spectral like resolution. It uses the pressure possion formulation with spectral solver for poisson equation based on numpy FFTs. 

# Instructions
1. Download all the modules in a working directory.
2. FluiDNS.py is the driver script. Set all required parameters and specifications with save directory for outputs.
3. Check for all dependencies before running simulation.
4. pyrun.sh is the shell file for running simulations on individual PC as well as clusters.

# Dependencies
1. Numpy
2. Scipy
3. Numba
4. Matplotlib (for plotting)
5. uvw  (for writing .vtr files)

# Default/test case
The default parameter setup is for the simulation of mixed convection over two square cylinders arranged transversely in the presence of constant velocity inlet and free slip walls.

