#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computation of correlation coefficient and rate of information transfer (Liang, 2021)
Extension of study from Vannitsem & Liang (2022) on the links between North Pacific and Atlantic indices
With variables downloaded by DD on 20/01/2023
With 12 lags (11 months)

Last updated: 28/06/2023

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
analysis = 1 # 1: all 8 indices; 2: only atmospheric indices; 3: only oceanic indices
if analysis == 1:
    nvar = int(8*12) # number of variables
elif analysis == 2 or analysis == 3:
    nvar = int(4*12)
n_iter = 1000 # number of bootstrap realizations (for computing the error in Liang index)
conf = 1.96 # 1.96 if 95% confidence interval; 2.58 if 99% confidence interval; 1.65 if 90% confidence interval

# Time parameters
dt = 1 # time step

# Load time series
filename = 'Indices_wotrend.npy'
PNA,NAO,AO,QBO,AMO,PDO,TNA,nino34 = np.load(filename,allow_pickle=True)

# Lagged variables
PNA_lag1 = np.roll(PNA,1)
NAO_lag1 = np.roll(NAO,1)
AO_lag1 = np.roll(AO,1)
QBO_lag1 = np.roll(QBO,1)
AMO_lag1 = np.roll(AMO,1)
PDO_lag1 = np.roll(PDO,1)
TNA_lag1 = np.roll(TNA,1)
nino34_lag1 = np.roll(nino34,1)

PNA_lag2 = np.roll(PNA,2)
NAO_lag2 = np.roll(NAO,2)
AO_lag2 = np.roll(AO,2)
QBO_lag2 = np.roll(QBO,2)
AMO_lag2 = np.roll(AMO,2)
PDO_lag2 = np.roll(PDO,2)
TNA_lag2 = np.roll(TNA,2)
nino34_lag2 = np.roll(nino34,2)

PNA_lag3 = np.roll(PNA,3)
NAO_lag3 = np.roll(NAO,3)
AO_lag3 = np.roll(AO,3)
QBO_lag3 = np.roll(QBO,3)
AMO_lag3 = np.roll(AMO,3)
PDO_lag3 = np.roll(PDO,3)
TNA_lag3 = np.roll(TNA,3)
nino34_lag3 = np.roll(nino34,3)

PNA_lag4 = np.roll(PNA,4)
NAO_lag4 = np.roll(NAO,4)
AO_lag4 = np.roll(AO,4)
QBO_lag4 = np.roll(QBO,4)
AMO_lag4 = np.roll(AMO,4)
PDO_lag4 = np.roll(PDO,4)
TNA_lag4 = np.roll(TNA,4)
nino34_lag4 = np.roll(nino34,4)

PNA_lag5 = np.roll(PNA,5)
NAO_lag5 = np.roll(NAO,5)
AO_lag5 = np.roll(AO,5)
QBO_lag5 = np.roll(QBO,5)
AMO_lag5 = np.roll(AMO,5)
PDO_lag5 = np.roll(PDO,5)
TNA_lag5 = np.roll(TNA,5)
nino34_lag5 = np.roll(nino34,5)

PNA_lag6 = np.roll(PNA,6)
NAO_lag6 = np.roll(NAO,6)
AO_lag6 = np.roll(AO,6)
QBO_lag6 = np.roll(QBO,6)
AMO_lag6 = np.roll(AMO,6)
PDO_lag6 = np.roll(PDO,6)
TNA_lag6 = np.roll(TNA,6)
nino34_lag6 = np.roll(nino34,6)

PNA_lag7 = np.roll(PNA,7)
NAO_lag7 = np.roll(NAO,7)
AO_lag7 = np.roll(AO,7)
QBO_lag7 = np.roll(QBO,7)
AMO_lag7 = np.roll(AMO,7)
PDO_lag7 = np.roll(PDO,7)
TNA_lag7 = np.roll(TNA,7)
nino34_lag7 = np.roll(nino34,7)

PNA_lag8 = np.roll(PNA,8)
NAO_lag8 = np.roll(NAO,8)
AO_lag8 = np.roll(AO,8)
QBO_lag8 = np.roll(QBO,8)
AMO_lag8 = np.roll(AMO,8)
PDO_lag8 = np.roll(PDO,8)
TNA_lag8 = np.roll(TNA,8)
nino34_lag8 = np.roll(nino34,8)

PNA_lag9 = np.roll(PNA,9)
NAO_lag9 = np.roll(NAO,9)
AO_lag9 = np.roll(AO,9)
QBO_lag9 = np.roll(QBO,9)
AMO_lag9 = np.roll(AMO,9)
PDO_lag9 = np.roll(PDO,9)
TNA_lag9 = np.roll(TNA,9)
nino34_lag9 = np.roll(nino34,9)

PNA_lag10 = np.roll(PNA,10)
NAO_lag10 = np.roll(NAO,10)
AO_lag10 = np.roll(AO,10)
QBO_lag10 = np.roll(QBO,10)
AMO_lag10 = np.roll(AMO,10)
PDO_lag10 = np.roll(PDO,10)
TNA_lag10 = np.roll(TNA,10)
nino34_lag10 = np.roll(nino34,10)

PNA_lag11 = np.roll(PNA,11)
NAO_lag11 = np.roll(NAO,11)
AO_lag11 = np.roll(AO,11)
QBO_lag11 = np.roll(QBO,11)
AMO_lag11 = np.roll(AMO,11)
PDO_lag11 = np.roll(PDO,11)
TNA_lag11 = np.roll(TNA,11)
nino34_lag11 = np.roll(nino34,11)

# Compute rate of information transfer and correlation coefficient using function_liang
if analysis == 1:
   xx = np.array((PNA,NAO,AO,QBO,AMO,PDO,TNA,nino34,
                  PNA_lag1,NAO_lag1,AO_lag1,QBO_lag1,AMO_lag1,PDO_lag1,TNA_lag1,nino34_lag1,
                  PNA_lag2,NAO_lag2,AO_lag2,QBO_lag2,AMO_lag2,PDO_lag2,TNA_lag2,nino34_lag2,
                  PNA_lag3,NAO_lag3,AO_lag3,QBO_lag3,AMO_lag3,PDO_lag3,TNA_lag3,nino34_lag3,
                  PNA_lag4,NAO_lag4,AO_lag4,QBO_lag4,AMO_lag4,PDO_lag4,TNA_lag4,nino34_lag4,
                  PNA_lag5,NAO_lag5,AO_lag5,QBO_lag5,AMO_lag5,PDO_lag5,TNA_lag5,nino34_lag5,
                  PNA_lag6,NAO_lag6,AO_lag6,QBO_lag6,AMO_lag6,PDO_lag6,TNA_lag6,nino34_lag6,
                  PNA_lag7,NAO_lag7,AO_lag7,QBO_lag7,AMO_lag7,PDO_lag7,TNA_lag7,nino34_lag7,
                  PNA_lag8,NAO_lag8,AO_lag8,QBO_lag8,AMO_lag8,PDO_lag8,TNA_lag8,nino34_lag8,
                  PNA_lag9,NAO_lag9,AO_lag9,QBO_lag9,AMO_lag9,PDO_lag9,TNA_lag9,nino34_lag9,
                  PNA_lag10,NAO_lag10,AO_lag10,QBO_lag10,AMO_lag10,PDO_lag10,TNA_lag10,nino34_lag10,
                  PNA_lag11,NAO_lag11,AO_lag11,QBO_lag11,AMO_lag11,PDO_lag11,TNA_lag11,nino34_lag11,))
elif analysis == 2:
    xx = np.array((PNA,NAO,AO,QBO,PNA_lag1,NAO_lag1,AO_lag1,QBO_lag1,
                   PNA_lag2,NAO_lag2,AO_lag2,QBO_lag2,PNA_lag3,NAO_lag3,AO_lag3,QBO_lag3,
                   PNA_lag4,NAO_lag4,AO_lag4,QBO_lag4,PNA_lag5,NAO_lag5,AO_lag5,QBO_lag5,
                   PNA_lag6,NAO_lag6,AO_lag6,QBO_lag6))
elif analysis == 3:
    xx = np.array((AMO,PDO,TNA,nino34,AMO_lag1,PDO_lag1,TNA_lag1,nino34_lag1,
                   AMO_lag2,PDO_lag2,TNA_lag2,nino34_lag2,AMO_lag3,PDO_lag3,TNA_lag3,nino34_lag3,
                   AMO_lag4,PDO_lag4,TNA_lag4,nino34_lag4,AMO_lag5,PDO_lag5,TNA_lag5,nino34_lag5,
                   AMO_lag6,PDO_lag6,TNA_lag6,nino34_lag6))
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
    if analysis == 1:
        filename = 'real_case_study_liang_lags.npy'
    elif analysis == 2:
        filename = 'real_case_study_liang_atm_lags.npy'
    elif analysis == 3:
        filename = 'real_case_study_liang_oce_lags.npy'
    np.save(filename,[T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R])