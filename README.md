# FluiDNS
2D DNS code in Python 
A highly accurate pseudo-spectral DNS solver for 2 dimensional incompressible flows with heat transfer is presented. Semi-inplicit compact finite difference scheme is implemented for 4th order spatial accuracy with pressure-poisson formulation for 2-dimensional incompressible Navier-Stokes equations. Multiple time integration methods are included to capture the unsteady dynamics for various flow problems. FFT based full spectral solver is used for the highly accurate solutions of pressure-poisson equation. Solver is highly optimized for the serial implementation in Python on Linux systems with Thomas' algorithm and LU decomposition for solutions of matrix equations. It solves the governing equations on a collacated uniform cartesian mesh with immersed boundary methods for the description of solid bodies with or without oscillations inside the flow fields. The code is well tested and validated for the various 2-dimensional convective heat transfer problems.   

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
