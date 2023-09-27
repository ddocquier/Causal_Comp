#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results from Lorenz (1963) model - Correlation, Liang index and PCMCI with lag

Focusing on x^2 and z

Last updated: 01/09/2023

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Options
save_fig = True
conf = 1.96 # 1.96
sampling = 1

# Load results Liang and R
tau,R,error_tau,error_R,sig_tau,sig_R = np.load('Lorenz_xsquare.npy',allow_pickle=True)
tau_xlag10,R_xlag10,error_tau_xlag10,error_R_xlag10,sig_tau_xlag10,sig_R_xlag10 = np.load('Lorenz_xsquare_xlag10.npy',allow_pickle=True)
tau_xlag20,R_xlag20,error_tau_xlag20,error_R_xlag20,sig_tau_xlag20,sig_R_xlag20 = np.load('Lorenz_xsquare_xlag20.npy',allow_pickle=True)
tau_xlag30,R_xlag30,error_tau_xlag30,error_R_xlag30,sig_tau_xlag30,sig_R_xlag30 = np.load('Lorenz_xsquare_xlag30.npy',allow_pickle=True)
tau_xlag40,R_xlag40,error_tau_xlag40,error_R_xlag40,sig_tau_xlag40,sig_R_xlag40 = np.load('Lorenz_xsquare_xlag40.npy',allow_pickle=True)
tau_xlag50,R_xlag50,error_tau_xlag50,error_R_xlag50,sig_tau_xlag50,sig_R_xlag50 = np.load('Lorenz_xsquare_xlag50.npy',allow_pickle=True)
tau_xlag60,R_xlag60,error_tau_xlag60,error_R_xlag60,sig_tau_xlag60,sig_R_xlag60 = np.load('Lorenz_xsquare_xlag60.npy',allow_pickle=True)
tau_xlag70,R_xlag70,error_tau_xlag70,error_R_xlag70,sig_tau_xlag70,sig_R_xlag70 = np.load('Lorenz_xsquare_xlag70.npy',allow_pickle=True)
tau_xlag80,R_xlag80,error_tau_xlag80,error_R_xlag80,sig_tau_xlag80,sig_R_xlag80 = np.load('Lorenz_xsquare_xlag80.npy',allow_pickle=True)
tau_xlag90,R_xlag90,error_tau_xlag90,error_R_xlag90,sig_tau_xlag90,sig_R_xlag90 = np.load('Lorenz_xsquare_xlag90.npy',allow_pickle=True)
tau_xlag100,R_xlag100,error_tau_xlag100,error_R_xlag100,sig_tau_xlag100,sig_R_xlag100 = np.load('Lorenz_xsquare_xlag100.npy',allow_pickle=True)
tau_xlag_10,R_xlag_10,error_tau_xlag_10,error_R_xlag_10,sig_tau_xlag_10,sig_R_xlag_10 = np.load('Lorenz_xsquare_xlag-10.npy',allow_pickle=True)
tau_xlag_20,R_xlag_20,error_tau_xlag_20,error_R_xlag_20,sig_tau_xlag_20,sig_R_xlag_20 = np.load('Lorenz_xsquare_xlag-20.npy',allow_pickle=True)
tau_xlag_30,R_xlag_30,error_tau_xlag_30,error_R_xlag_30,sig_tau_xlag_30,sig_R_xlag_30 = np.load('Lorenz_xsquare_xlag-30.npy',allow_pickle=True)
tau_xlag_40,R_xlag_40,error_tau_xlag_40,error_R_xlag_40,sig_tau_xlag_40,sig_R_xlag_40 = np.load('Lorenz_xsquare_xlag-40.npy',allow_pickle=True)
tau_xlag_50,R_xlag_50,error_tau_xlag_50,error_R_xlag_50,sig_tau_xlag_50,sig_R_xlag_50 = np.load('Lorenz_xsquare_xlag-50.npy',allow_pickle=True)
tau_xlag_60,R_xlag_60,error_tau_xlag_60,error_R_xlag_60,sig_tau_xlag_60,sig_R_xlag_60 = np.load('Lorenz_xsquare_xlag-60.npy',allow_pickle=True)
tau_xlag_70,R_xlag_70,error_tau_xlag_70,error_R_xlag_70,sig_tau_xlag_70,sig_R_xlag_70 = np.load('Lorenz_xsquare_xlag-70.npy',allow_pickle=True)
tau_xlag_80,R_xlag_80,error_tau_xlag_80,error_R_xlag_80,sig_tau_xlag_80,sig_R_xlag_80 = np.load('Lorenz_xsquare_xlag-80.npy',allow_pickle=True)
tau_xlag_90,R_xlag_90,error_tau_xlag_90,error_R_xlag_90,sig_tau_xlag_90,sig_R_xlag_90 = np.load('Lorenz_xsquare_xlag-90.npy',allow_pickle=True)
tau_xlag_100,R_xlag_100,error_tau_xlag_100,error_R_xlag_100,sig_tau_xlag_100,sig_R_xlag_100 = np.load('Lorenz_xsquare_xlag-100.npy',allow_pickle=True)

# Create tau vectors - xlag
tau_xz_xlag = np.array([tau_xlag100[0,2],tau_xlag90[0,2],tau_xlag80[0,2],tau_xlag70[0,2],tau_xlag60[0,2],tau_xlag50[0,2],tau_xlag40[0,2],tau_xlag30[0,2],tau_xlag20[0,2],tau_xlag10[0,2],tau[0,2],tau_xlag_10[0,2],tau_xlag_20[0,2],tau_xlag_30[0,2],tau_xlag_40[0,2],tau_xlag_50[0,2],tau_xlag_60[0,2],tau_xlag_70[0,2],tau_xlag_80[0,2],tau_xlag_90[0,2],tau_xlag_100[0,2]])
tau_zx_xlag = np.array([tau_xlag100[2,0],tau_xlag90[2,0],tau_xlag80[2,0],tau_xlag70[2,0],tau_xlag60[2,0],tau_xlag50[2,0],tau_xlag40[2,0],tau_xlag30[2,0],tau_xlag20[2,0],tau_xlag10[2,0],tau[2,0],tau_xlag_10[2,0],tau_xlag_20[2,0],tau_xlag_30[2,0],tau_xlag_40[2,0],tau_xlag_50[2,0],tau_xlag_60[2,0],tau_xlag_70[2,0],tau_xlag_80[2,0],tau_xlag_90[2,0],tau_xlag_100[2,0]])

# Create R vectors - xlag
R_xz_xlag = np.array([R_xlag100[0,2],R_xlag90[0,2],R_xlag80[0,2],R_xlag70[0,2],R_xlag60[0,2],R_xlag50[0,2],R_xlag40[0,2],R_xlag30[0,2],R_xlag20[0,2],R_xlag10[0,2],R[0,2],R_xlag_10[0,2],R_xlag_20[0,2],R_xlag_30[0,2],R_xlag_40[0,2],R_xlag_50[0,2],R_xlag_60[0,2],R_xlag_70[0,2],R_xlag_80[0,2],R_xlag_90[0,2],R_xlag_100[0,2]])

# Load results PCMCI
beta = np.load('PCMCI/tig5_LIANG_LM_all_x2_X2YZ_tau0-100',allow_pickle=True)

# Lag vector
lag = np.linspace(-100,100,21) / 100.
lag2 = np.array([0,10,20,30,40,50,60,70,80,90,100]) / 100. # PCMCI

# Plots
fig = plt.figure(figsize=(18,12))
fig.subplots_adjust(left=0.08,bottom=0.1,right=0.96,top=0.95,wspace=0.22,hspace=0.3)

# Scatter plot of R as a function of x lag
ax1 = fig.add_subplot(2,2,1)
ax1.plot(lag,R_xz_xlag,'bo--',markersize=8,label=r'$R_{x_{t+l}^2,z_t}$')
ax1.legend(loc='upper right',fontsize=20)
ax1.tick_params(axis='both',labelsize=16)
ax1.set_xticks(np.arange(-1,1.1,0.2))
ax1.axes.grid(linestyle='--')
ax1.axhline(linestyle='--',color='k')
ax1.set_xlabel('Lag $l$ in $x$ (unit times)',fontsize=20)
ax1.set_ylabel('Correlation coefficient $R$',fontsize=20)
ax1.axis([-1.05,1.05,-1,1])
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of tau as a function of x lag
ax2 = fig.add_subplot(2,2,2)
ax2.plot(lag,tau_xz_xlag,'bo--',markersize=8,label=r'$\tau_{x_{t+l}^2 \longrightarrow z_t}$')
ax2.plot(lag,tau_zx_xlag,'rx--',markersize=8,label=r'$\tau_{z_t \longrightarrow x_{t+l}^2}$')
ax2.legend(loc='upper right',ncol=2,fontsize=20)
ax2.tick_params(axis='both',labelsize=16)
ax2.set_xticks(np.arange(-1,1.1,0.2))
ax2.axes.grid(linestyle='--')
ax2.axhline(linestyle='--',color='k')
ax2.set_xlabel('Lag $l$ in $x$ (unit times)',fontsize=20)
ax2.set_ylabel(r'Rate of information transfer $\tau$ ($\%$)',fontsize=20)
ax2.axis([-1.05,1.05,-70,70])
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of beta as a function of x lag
ax4 = fig.add_subplot(2,2,4)
#ax4.plot(lag2,beta[0,2,::10],'bo--',markersize=8,label=r'$\beta_{x_{t+lag}^2 \longrightarrow z_t}$')
#ax4.plot(lag2,beta[2,0,::10],'rx--',markersize=8,label=r'$\beta_{z_t \longrightarrow x_{t+lag}^2}$')
ax4.plot(np.arange(101)/100,beta[0,2,:],'b--',markersize=8,label=r'$\beta_{x_{t+l}^2 \longrightarrow z_t}$')
ax4.plot(np.arange(101)/100,beta[2,0,:],'r--',markersize=8,label=r'$\beta_{z_t \longrightarrow x_{t+l}^2}$')
ax4.legend(loc='upper right',fontsize=20)
ax4.tick_params(axis='both',labelsize=16)
ax4.set_xticks(np.arange(-1,1.1,0.2))
#ax4.axes.grid(linestyle='--')
ax4.axhline(linestyle='--',color='gray',linewidth=0.5)
ax4.set_xlabel('Lag $l$ in $x$ (unit times)',fontsize=20)
ax4.set_ylabel(r'Path coefficient $\beta$',fontsize=20)
ax4.axis([-0.05,1.05,-2,2])
ax4.set_title('(c)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig7.png')