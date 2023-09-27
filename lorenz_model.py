#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
GOAL
    Computation of correlation coefficient and rate of information transfer (Liang, 2021) with or without a lag
    Lorenz (1963) model: system of 3 ordinary differential equations having chaotic solutions for certain parameters and initial conditions (such as the ones used below)
    Simplified mathematical model representing atmospheric convection, coming from Saltzman (1962)
    Equations related the properties of a 2D fluid layer uniformly warmed from below and cooled from above
    Lorenz system is nonlinear, aperiodic, 3D and deterministic
    ODE solutions obtained with 4th order Runge-Kutta (adapted from https://scipython.com/blog/the-lorenz-attractor/)
PROGRAMMER
    D. Docquier
LAST UPDATE
    21/06/2023
'''

# Standard libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys

# My libraries
sys.path.append('/home/dadocq/Documents/Codes/Liang/')
from function_liang_nvar import compute_liang_nvar # function to compute rate of information transfer
#from function_liang_nvar_fisher import compute_liang_nvar # function to compute rate of information transfer with Fisher Information Matrix for significance

# Function of Lorenz (1963) model (convection equations)
def lorenz(t,X,sigma,rho,beta):
    
    """
    Given:
        X: point of interest in 3D space (x,y,z)
        sigma,rho,beta: parameters defining the Lorenz attractor
    Returns:
        x_dot,y_dot,z_dot: values of the Lorenz attractor's partial derivatives at the point X (x,y,z)
    """
    
    x,y,z = X
    x_dot = sigma * (y - x) # x is proportional to the intensity of the convective motion
    y_dot = rho * x - y - x * z # y is proportional to the temperature difference between the ascending and descending currents
                                # similar signs of x and y denote that the warm fluid rises and the cold fluid descends
                                # opposite signs of x and y denote that the warm fluid descends and the cold fluid rises
    z_dot = x * y - beta * z # z is proportional to the distorsion of the vertical temperature profile from linearity
                             # z>0: the strongest gradients occur near the boundaries
    
    return x_dot,y_dot,z_dot

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
        sig = 1
    else:
        sig = 0
    return sig

# Options
save_var = False # True: save Liang index and correlation coefficient (for plotting afterwards); False: don't save variables
load_series = True # True: load already saved time series (x,y,z); False: compute and save time series
lag = 0 # shift x/y/z by a number of time steps before (to take lag into account)
if lag != 0:
    shift_var = 3 # 1: shift x; 2: shift y; 3: shift z
n_iter = 100 # number of bootstrap realizations (for computing the error in Liang index)
nvar = 3 # number of variables (3 by default for the Lorenz model)
conf = 1.96 # 1.96 if 95% confidence interval; 2.58 if 99% confidence interval; 1.65 if 90% confidence interval
sampling = 1 # 1 by default

# Time parameters
dt = 0.01 # time step (Lorenz [1963]: dt=0.01)
nt = 100000 # number of time steps (Jiang & Adeli [2003]: nt=26667; Krakovska et al. [2018]: nt=21000)
tmax = int(nt * dt) # total duration in unit times (nt*dt; Jiang & Adeli [2003]: tmax=300)
t = np.linspace(0,tmax,nt) # build time vector from 0 to tmax with nt time steps
start_computation = int(100 / dt) # exclude the first transient times for computing correlation coefficient and rate of information transfer (Krakovska et al. [2018]: start_computation=1000)

# Time series
filename = 'Lorenz_series.npy'
if load_series == True: # load time series
    t,x,y,z = np.load(filename,allow_pickle=True)

else: # compute time series and save them
    
    # Lorenz parameters
    # Note: stable equilibrium if rho < sigma*(sigma+beta+3)/(sigma-beta-1) (Lorenz, 1963; https://en.wikipedia.org/wiki/Lorenz_system)
    sigma = 10. # Prandtl number (Lorenz [1963]: sigma=10; Jiang & Adeli [2003]: sigma=16)
    rho = 28. # Rayleigh ratio (Lorenz [1963]: rho=28; Jiang & Adeli [2003]: rho=45.92)
    beta = 8./3. # geometric factor (Lorenz [1963]: beta=8/3; Jiang & Adeli [2003]: beta=4)
    
    # Initial values
    x0 = 0. # (Lorenz [1963]: x0=0; Jiang & Adeli [2003]: x0=0)
    y0 = 1. # (Lorenz [1963]: y0=1; Jiang & Adeli [2003]: y0=1)
    z0 = 0. # (Lorenz [1963]: z0=0; Jiang & Adeli [2003]: z0=1)
    
    # Solve ODEs with scipy.integrate.solve_ivp (https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
    # RK45: Runge-Kutta of order 4 for the error and order 5 for steps (local interpolation)
    # dense_output=True: compute continuous solution (needed to get ODE solution)
    soln = solve_ivp(lorenz,(0,tmax),(x0,y0,z0),method='RK45',dense_output=True,args=(sigma,rho,beta)) 
    x,y,z = soln.sol(t)
    
    # Save time series
    np.save(filename,[t,x,y,z])

# Shift x/y/z (to take lag into account)
if lag != 0:
    if shift_var == 1:
        x = np.roll(x,lag)
    elif shift_var == 2:
        y = np.roll(y,lag)
    elif shift_var == 3:
        z = np.roll(z,lag)

## 3D plot of the Lorenz system
#ax = plt.figure(figsize=(12,6)).add_subplot(projection='3d')
#ax.plot(x,y,z,lw=0.5)
#ax.set_xlabel('X',fontsize=16)
#ax.set_ylabel('Y',fontsize=16)
#ax.set_zlabel('Z',fontsize=16)
#ax.tick_params(axis='both',labelsize=12)
#plt.show()
#
## Time series of the first time steps
#end_series = 3000
#ax = plt.figure(figsize=(12,6)).add_subplot()
#ax.plot(t[0:end_series],x[0:end_series],'k',label='X')
#ax.plot(t[0:end_series],y[0:end_series],'r--',label='Y')
#ax.plot(t[0:end_series],z[0:end_series],'b',label='Z')
#ax.set_xlabel('Time',fontsize=16)
#ax.set_ylabel('X, Y, Z',fontsize=16)
#plt.axhline(y=0,color='gray',linestyle='--')
#ax.tick_params(axis='both',labelsize=12)
#ax.axis([-2,int(end_series*dt+2),-23,60])
#ax.legend(fontsize=12,ncol=3)
#plt.show()
#
## Whole time series
#ax = plt.figure(figsize=(12,6)).add_subplot()
#ax.plot(t,x,'k',label='X')
#ax.plot(t,y,'r--',label='Y')
#ax.plot(t,z,'b',label='Z')
#ax.set_xlabel('Time',fontsize=16)
#ax.set_ylabel('X, Y, Z',fontsize=16)
#plt.axhline(y=0,color='gray',linestyle='--')
#ax.tick_params(axis='both',labelsize=12)
#ax.axis([-20,int(np.size(x)*dt+20),-23,60])
#ax.legend(fontsize=12,ncol=3)
#plt.show()

# Compute rate of information transfer and Pearson correlation coefficient using function_liang (excluding first transient time steps)
xx = np.array((x[start_computation::sampling],y[start_computation::sampling],z[start_computation::sampling]))
T,tau,R,error_T,error_tau,error_R = compute_liang_nvar(xx,dt,n_iter)
#T,tau,error_T = compute_liang_nvar(xx,dt)

# Compute Pearson correlation coefficient with scipy.stats.pearsonr (to check previous computation)
R_xy = pearsonr(x[start_computation::sampling],y[start_computation::sampling])
R_xz = pearsonr(x[start_computation::sampling],z[start_computation::sampling])
R_yz = pearsonr(y[start_computation::sampling],z[start_computation::sampling])

# Compute significance of rate of information transfer and correlation coefficient (by combining bootstrap samples)
sig_T = np.zeros((nvar,nvar))
sig_tau = np.zeros((nvar,nvar))
sig_R = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for k in np.arange(nvar):
        sig_T[j,k] = compute_sig(T[j,k],error_T[j,k],conf)
        sig_tau[j,k] = compute_sig(tau[j,k],error_tau[j,k],conf)
        sig_R[j,k] = compute_sig(R[j,k],error_R[j,k],conf)
        
# Save results
if save_var == True:
    if lag == 0:
        if sampling == 1:
            filename = 'Lorenz_nolag.npy'
        else:
            filename = 'Lorenz_nolag_samp' + str(sampling) + '.npy'
    else:
        if shift_var == 1:
            if sampling == 1:
                filename = 'Lorenz_xlag' + str(lag) + '.npy'
            else:
                filename = 'Lorenz_xlag' + str(lag) + '_samp' + str(sampling) + '.npy'
        elif shift_var == 2:
            if sampling == 1:
                filename = 'Lorenz_ylag' + str(lag) + '.npy'
            else:
                filename = 'Lorenz_ylag' + str(lag) + '_samp' + str(sampling) + '.npy'
        elif shift_var == 3:
            if sampling == 1:
                filename = 'Lorenz_zlag' + str(lag) + '.npy'
            else:
                filename = 'Lorenz_zlag' + str(lag) + '_samp' + str(sampling) + '.npy'
    np.save(filename,[tau,R,error_tau,error_R,sig_tau,sig_R])