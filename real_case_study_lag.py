#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computation of correlation coefficient and rate of information transfer (Liang, 2021)
Extension of study from Vannitsem & Liang (2022) on the links between North Pacific and Atlantic indices
With variables downloaded by DD on 20/01/2023
With lag

Last updated: 31/01/2023

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
save_var = True # True: save Liang index and correlation coefficient (for plotting afterwards); False: don't save variables
analysis = 1 # 1: all 8 indices; 2: only atmospheric indices; 3: only oceanic indices; 4: all indices except AO; 5: all indices except AMO
if analysis == 1:
    nvar = 8 # number of variables
elif analysis == 2 or analysis == 3:
    nvar = 4
elif analysis == 4 or analysis == 5:
    nvar = 7
lag = 4 # shift variable by a number of time steps before (to take lag into account)
n_iter = 1000 # number of bootstrap realizations (for computing the error in Liang index)
conf = 1.96 # 1.96 if 95% confidence interval; 2.58 if 99% confidence interval; 1.65 if 90% confidence interval

# Time parameters
dt = 1 # time step

# Load time series
filename = 'Indices_wotrend.npy'
PNA,NAO,AO,QBO,AMO,PDO,TNA,nino34 = np.load(filename,allow_pickle=True)

# Lagged variables
if lag != 0:
    PNA_lag = np.roll(PNA,lag)
    NAO_lag = np.roll(NAO,lag)
    AO_lag = np.roll(AO,lag)
    QBO_lag = np.roll(QBO,lag)
    AMO_lag = np.roll(AMO,lag)
    PDO_lag = np.roll(PDO,lag)
    TNA_lag = np.roll(TNA,lag)
    nino34_lag = np.roll(nino34,lag)

# Initialization of variables
T = np.zeros((nvar,nvar))
tau = np.zeros((nvar,nvar))
R = np.zeros((nvar,nvar))
error_T = np.zeros((nvar,nvar))
error_tau = np.zeros((nvar,nvar))
error_R = np.zeros((nvar,nvar))

# Compute rate of information transfer and correlation coefficient using function_liang
if analysis == 1:
    if lag == 0:
        xx = np.array((PNA,NAO,AO,QBO,AMO,PDO,TNA,nino34))
    else:
        xx = np.array((PNA,NAO,AO,QBO,AMO,PDO,TNA,nino34,PNA_lag,NAO_lag,AO_lag,QBO_lag,AMO_lag,PDO_lag,TNA_lag,nino34_lag))
elif analysis == 2:
    if lag == 0:
        xx = np.array((PNA,NAO,AO,QBO))
    else:
        xx = np.array((PNA,NAO,AO,QBO,PNA_lag,NAO_lag,AO_lag,QBO_lag))
elif analysis == 3:
    if lag == 0:
        xx = np.array((AMO,PDO,TNA,nino34))
    else:
        xx = np.array((AMO,PDO,TNA,nino34,AMO_lag,PDO_lag,TNA_lag,nino34_lag))
elif analysis == 4:
    if lag == 0:
        xx = np.array((PNA,NAO,QBO,AMO,PDO,TNA,nino34))
    else:
        xx = np.array((PNA,NAO,QBO,AMO,PDO,TNA,nino34,PNA_lag,NAO_lag,QBO_lag,AMO_lag,PDO_lag,TNA_lag,nino34_lag))
elif analysis == 5:
    if lag == 0:
        xx = np.array((PNA,NAO,AO,QBO,PDO,TNA,nino34))
    else:
        xx = np.array((PNA,NAO,AO,QBO,PDO,TNA,nino34,PNA_lag,NAO_lag,AO_lag,QBO_lag,PDO_lag,TNA_lag,nino34_lag))
if lag != 0:
    nvar = nvar * 2
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
    if lag == 0:
        if analysis == 1:
            filename = 'real_case_study_liang.npy'
        elif analysis == 2:
            filename = 'real_case_study_liang_atm.npy'
        elif analysis == 3:
            filename = 'real_case_study_liang_oce.npy'
        elif analysis == 4:
            filename = 'real_case_study_liang_woao.npy'
        elif analysis == 5:
            filename = 'real_case_study_liang_woamo.npy'
    else:
        if analysis == 1:
            filename = 'real_case_study_liang_lag-' + str(lag) + '.npy'
        elif analysis == 2:
            filename = 'real_case_study_liang_atm_lag-' + str(lag) + '.npy'
        elif analysis == 3:
            filename = 'real_case_study_liang_oce_lag-' + str(lag) + '.npy'
        elif analysis == 4:
            filename = 'real_case_study_liang_woao_lag-' + str(lag) + '.npy'
        elif analysis == 5:
            filename = 'real_case_study_liang_woamo_lag-' + str(lag) + '.npy'
    np.save(filename,[T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R])