#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computation of correlation coefficient and rate of information transfer (Liang, 2021)
9D nonlinear stochastic model (Subramaniyam et al., 2021; eq. (17))
Online Matlab code from N. Subramaniyam: https://github.com/narayanps/causal_inference_with_OPTNs/blob/master/func/models_for_sim/test_model.m
ATTENTION: mistake in eq. (17) of Subramaniyam et al. (2021) as the exponentials should be negative and outside of the parentheses (as in the Matlab code)

Last update: 29/07/2022

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
save_series = False # True: save time series (x,y,z); False: don't save time series
nvar = 9 # number of variables
n_iter = 400 # number of bootstrap realizations (for computing the error in Liang index)
conf = 1.96 # 1.96 if 95% confidence interval; 2.58 if 99% confidence interval; 1.65 if 90% confidence interval

# Time parameters
dt = 1 # time step
tmax = 1.e6 # total duration in unit times (1.e6; Peña & Kalnay [2004]: tmax=20000)
nt = int(tmax / dt) # number of time steps
t = np.linspace(0,tmax,nt) # time vector (varying between 0 and tmax with nt time steps)
start_computation = int(10000 / dt) # exclude the first transient times for computing correlation coefficient and rate of information transfer

# Model parameters
c13 = 0.25
c21 = 2.5
c31 = 1.8
c41 = 1.5
c54 = 1.5
c64 = 1.2
c76 = 1.5
c78 = 0.8
c79 = 1.8
alpha = 3.4
omega = 0.4
    
# Zero mean Gaussian noise
mean_u = 0.
std_u = 1.
u = np.zeros((nvar,nt))
for var in np.arange(nvar):
    u[var,:] = np.random.normal(mean_u,std_u,nt)

# Initialization of variables
x1 = np.zeros(nt)
x2 = np.zeros(nt)
x3 = np.zeros(nt)
x4 = np.zeros(nt)
x5 = np.zeros(nt)
x6 = np.zeros(nt)
x7 = np.zeros(nt)
x8 = np.zeros(nt)
x9 = np.zeros(nt)
T = np.zeros((nvar,nvar))
tau = np.zeros((nvar,nvar))
R = np.zeros((nvar,nvar))
error_T = np.zeros((nvar,nvar))
error_tau = np.zeros((nvar,nvar))
error_R = np.zeros((nvar,nvar))

# 9D model
for i in np.arange(4,nt):    
    x1[i] = alpha * x1[i-1] * (1. - x1[i-1]**2.) * np.exp(-x1[i-1]**2.) + c21 * x2[i-4]  + \
            c31 * x3[i-2] + c41 * x4[i-2] + omega * u[0,i]
    x2[i] = alpha * x2[i-1] * (1. - x2[i-1]**2.) * np.exp(-x2[i-1]**2.) + omega * u[1,i]
    x3[i] = alpha * x3[i-1] * (1. - x3[i-1]**2.) * np.exp(-x3[i-1]**2.) + c13 * x1[i-1] + omega * u[2,i]
    x4[i] = alpha * x4[i-1] * (1. - x4[i-1]**2.) * np.exp(-x4[i-1]**2.) + c54 * x5[i-3] + \
            c64 * x6[i-1] + omega * u[3,i]
    x5[i] = alpha * x5[i-1] * (1. - x5[i-1]**2.) * np.exp(-x5[i-1]**2.) + omega * u[4,i]
    x6[i] = alpha * x6[i-1] * (1. - x6[i-1]**2.) * np.exp(-x6[i-1]**2.) + c76 * x7[i-3] + omega * u[5,i]
    x7[i] = alpha * x7[i-1] * (1. - x7[i-1]**2.) * np.exp(-x7[i-1]**2.) + omega * u[6,i]
    x8[i] = alpha * x8[i-1] * (1. - x8[i-1]**2.) * np.exp(-x8[i-1]**2.) + c78 * x7[i-1] + omega * u[7,i]
    x9[i] = alpha * x9[i-1] * (1. - x9[i-1]**2.) * np.exp(-x9[i-1]**2.) + c79 * x7[i-1] + omega * u[8,i]

# Save time series
if save_series == True:
    filename = '9D_stochastic_series.npy'
    np.save(filename,[t,x1,x2,x3,x4,x5,x6,x7,x8,x9])

# Compute rate of information transfer and correlation coefficient using function_liang
xx = np.array((x1[start_computation::],x2[start_computation::],x3[start_computation::], \
               x4[start_computation::],x5[start_computation::],x6[start_computation::], \
               x7[start_computation::],x8[start_computation::],x9[start_computation::]))
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
    filename = '9D_stochastic_liang.npy'
    np.save(filename,[T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R])