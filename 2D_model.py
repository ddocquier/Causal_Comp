#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computation of correlation coefficient and rate of information transfer (Liang, 2021)
2D stochastic differential equation (Liang, 2014; eq. (12)):
dX1 = (a11 * X1 + a12 * X2) * dt + sigma1 * dW1
dX2 = (a22 * X2 + a21 * X1) * dt + sigma2 * dW2
See also eq. (2) of Vannitsem et al. (2019)

Last updated: 15/09/2023

@author: David Docquier
"""

# Import libraries
import numpy as np
import sys

# My libraries
sys.path.append('/home/dadocq/Documents/Codes/Liang/')
from function_liang_nvar import compute_liang_nvar # function to compute rate of information transfer

# Function to test significance (based on the confidence interval)
def compute_sig(var,error,conf):
    if (var-conf*error < 0. and var+conf*error < 0.) or (var-conf*error > 0. and var+conf*error > 0.):
        sig = 1
    else:
        sig = 0
    return sig

# Options
load_series = True # True: load time series; False: compute and save time series
save_var = True # True: save Liang index and correlation coefficient (for plotting afterwards); False: don't save variables
nvar = 2 # number of variables
n_iter = 200 # number of bootstrap realizations (for computing the error in Liang index)
conf = 1.96 # 1.96 if 95% confidence interval; 2.58 if 99% confidence interval; 1.65 if 90% confidence interval

# Time parameters
dt = 0.001 # time step (Liang [2014]: dt=0.001)
tmax = 1000 # total duration in unit times (Liang [2014]: tmax=100)
nt = int(tmax / dt) # number of time steps
t = np.linspace(0,tmax,nt) # time vector (varying between 0 and tmax with nt time steps)
start_computation = int(10 / dt) # exclude the first transient times for computing correlation coefficient and rate of information transfer

# Time series
filename = '2D_series.npy'
if load_series == True: # load time series
    t,X1,X2 = np.load(filename,allow_pickle=True)

else: # compute time series and save them

    # Equation parameters
    a11 = -1
    a12 = 0.5
    a21 = 0
    a22 = -1
    
    # Random noise for stochastic process
    mean_noise = 0
    std_noise = 1
    sigma1 = 0.1
    sigma2 = 0.1
    dW1 = np.sqrt(dt) * np.random.normal(mean_noise,std_noise,nt) # normal random noise in X1
    dW2 = np.sqrt(dt) * np.random.normal(mean_noise,std_noise,nt) # normal random noise in X2
    
    # Initialization of variables
    X1 = np.zeros(nt)
    X2 = np.zeros(nt)
    X1[0] = 1
    X2[0] = 2
    T = np.zeros((nvar,nvar))
    tau = np.zeros((nvar,nvar))
    R = np.zeros((nvar,nvar))
    error_T = np.zeros((nvar,nvar))
    error_tau = np.zeros((nvar,nvar))
    error_R = np.zeros((nvar,nvar))
    
    # Solve equations with Euler-Maruyama method
    for i in np.arange(nt-1):
        X1[i+1] = X1[i] + (a11 * X1[i] + a12 * X2[i]) * dt + sigma1 * dW1[i]
        X2[i+1] = X2[i] + (a22 * X2[i] + a21 * X1[i]) * dt + sigma2 * dW2[i]
    
    # Save time series
    np.save(filename,[t,X1,X2])

# Compute T21 and T12 using function_liang
xx = np.array((X1[start_computation::],X2[start_computation::]))
T,tau,R,error_T,error_tau,error_R = compute_liang_nvar(xx,dt,n_iter)

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
    filename = '2D_liang.npy'
    np.save(filename,[T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R])