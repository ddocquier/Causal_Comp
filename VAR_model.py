#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computation of correlation coefficient and rate of information transfer (Liang, 2021)
6D vector autoregressive process (VAR; Liang, 2021; Section 5.1; eq. (21))
X(t+1) = alpha + A * X(t) + B * e(t+1)
Each variable is modeled as a linear combination of past values of itself and past values of other variables in the system

VAR(1): VAR of order 1
x1(t+1) = alpha1 + a11 * x1(t) + a12 * x2(t) + a13 * x3(t) + a14 * x4(t) + a15 * x5(t) + a16 * x6(t) + b * e1(t+1)
x2(t+1) = alpha2 + a21 * x1(t) + a22 * x2(t) + a23 * x3(t) + a24 * x4(t) + a25 * x5(t) + a26 * x6(t) + b * e2(t+1)
x3(t+1) = alpha3 + a31 * x1(t) + a32 * x2(t) + a33 * x3(t) + a34 * x4(t) + a35 * x5(t) + a36 * x6(t) + b * e3(t+1)
x4(t+1) = alpha4 + a41 * x1(t) + a42 * x2(t) + a43 * x3(t) + a44 * x4(t) + a45 * x5(t) + a46 * x6(t) + b * e4(t+1)
x5(t+1) = alpha5 + a51 * x1(t) + a52 * x2(t) + a53 * x3(t) + a54 * x4(t) + a55 * x5(t) + a56 * x6(t) + b * e5(t+1)
x6(t+1) = alpha6 + a61 * x1(t) + a62 * x2(t) + a63 * x3(t) + a64 * x4(t) + a65 * x5(t) + a66 * x6(t) + b * e6(t+1)

Last update: 26/06/2023

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
save_var = False # True: save Liang index and correlation coefficient (for plotting afterwards); False: don't save variables
load_series = True # True: load already saved time series (x,y,z); False: compute and save time series
nvar = 6 # number of variables (Liang [2021]: nvar=6)
n_iter = 400 # nuof bootstrap realizations (for computing the error in Liang index)
b_factor = 1 # amplitude of stochastic perturbation
conf = 1.96 # 1.96 if 95% confidence interval; 2.58 if 99% confidence interval; 1.65 if 90% confidence interval

# Time parameters
dt = 1 # time step
tmax = 1.e6 # total duration in unit times
nt = int(tmax / dt) # number of time steps (Liang [2021]: nt=10000)
t = np.linspace(0,tmax,nt) # time vector (varying between 0 and tmax with nt time steps)
start_computation = int(10000 / dt) # exclude the first transient times for computing correlation coefficient and rate of information transfer

# VAR parameters
alpha = np.array([0.1,0.7,0.5,0.2,0.8,0.3])
A = np.array([(0,0,-0.6,0,0,0),(-0.5,0,0,0,0,0.8),(0,0.7,0,0,0,0),(0,0,0,0.7,0.4,0),(0,0,0,0.2,0,0.7),(0,0,0,0,0,-0.5)])
B = np.ones(nvar) * b_factor

# Random errors
mean_e = 0
std_e = 1
e = np.zeros((nvar,nt))
for var in np.arange(nvar):
    e[var,:] = np.random.normal(mean_e,std_e,nt)

# Initialization of variables
X = np.zeros((nvar,nt))
T = np.zeros((nvar,nvar))
tau = np.zeros((nvar,nvar))
R = np.zeros((nvar,nvar))
error_T = np.zeros((nvar,nvar))
error_tau = np.zeros((nvar,nvar))
error_R = np.zeros((nvar,nvar))

# Time series
filename = 'VAR_series.npy'
if load_series == True: # load time series
    t,X[0,:],X[1,:],X[2,:],X[3,:],X[4,:],X[5,:] = np.load(filename,allow_pickle=True)

else: # compute time series and save them

    # VAR model
    for i in np.arange(nt-1):
        for var in np.arange(nvar):
            X[var,i+1] = alpha[var] + np.sum(A[var,:] * X[:,i]) + B[var] * e[var,i+1]

    # Save time series
    np.save(filename,[t,X[0,:],X[1,:],X[2,:],X[3,:],X[4,:],X[5,:]])

# Compute rate of information transfer and correlation coefficient using function_liang
xx = np.array((X[0,start_computation::],X[1,start_computation::],X[2,start_computation::],X[3,start_computation::],X[4,start_computation::],X[5,start_computation::]))
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
    filename = 'VAR_liang.npy'
    np.save(filename,[T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R])