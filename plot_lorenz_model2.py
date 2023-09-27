#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results from Lorenz (1963) model - Correlation, Liang index with lag and PCMCI

Last updated: 18/09/2023

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Options
save_fig = True
conf = 1.96 # 1.96

# Load results R and Liang
tau,R,error_tau,error_R,sig_tau,sig_R = np.load('Lorenz_nolag.npy',allow_pickle=True)
tau_xlag10,R_xlag10,error_tau_xlag10,error_R_xlag10,sig_tau_xlag10,sig_R_xlag10 = np.load('Lorenz_xlag10.npy',allow_pickle=True)
tau_xlag20,R_xlag20,error_tau_xlag20,error_R_xlag20,sig_tau_xlag20,sig_R_xlag20 = np.load('Lorenz_xlag20.npy',allow_pickle=True)
tau_xlag30,R_xlag30,error_tau_xlag30,error_R_xlag30,sig_tau_xlag30,sig_R_xlag30 = np.load('Lorenz_xlag30.npy',allow_pickle=True)
tau_xlag40,R_xlag40,error_tau_xlag40,error_R_xlag40,sig_tau_xlag40,sig_R_xlag40 = np.load('Lorenz_xlag40.npy',allow_pickle=True)
tau_xlag50,R_xlag50,error_tau_xlag50,error_R_xlag50,sig_tau_xlag50,sig_R_xlag50 = np.load('Lorenz_xlag50.npy',allow_pickle=True)
tau_xlag100,R_xlag100,error_tau_xlag100,error_R_xlag100,sig_tau_xlag100,sig_R_xlag100 = np.load('Lorenz_xlag100.npy',allow_pickle=True)
tau_xlag_10,R_xlag_10,error_tau_xlag_10,error_R_xlag_10,sig_tau_xlag_10,sig_R_xlag_10 = np.load('Lorenz_xlag-10.npy',allow_pickle=True)
tau_xlag_20,R_xlag_20,error_tau_xlag_20,error_R_xlag_20,sig_tau_xlag_20,sig_R_xlag_20 = np.load('Lorenz_xlag-20.npy',allow_pickle=True)
tau_xlag_30,R_xlag_30,error_tau_xlag_30,error_R_xlag_30,sig_tau_xlag_30,sig_R_xlag_30 = np.load('Lorenz_xlag-30.npy',allow_pickle=True)
tau_xlag_40,R_xlag_40,error_tau_xlag_40,error_R_xlag_40,sig_tau_xlag_40,sig_R_xlag_40 = np.load('Lorenz_xlag-40.npy',allow_pickle=True)
tau_xlag_50,R_xlag_50,error_tau_xlag_50,error_R_xlag_50,sig_tau_xlag_50,sig_R_xlag_50 = np.load('Lorenz_xlag-50.npy',allow_pickle=True)
tau_xlag_100,R_xlag_100,error_tau_xlag_100,error_R_xlag_100,sig_tau_xlag_100,sig_R_xlag_100 = np.load('Lorenz_xlag-100.npy',allow_pickle=True)
tau_ylag10,R_ylag10,error_tau_ylag10,error_R_ylag10,sig_tau_ylag10,sig_R_ylag10 = np.load('Lorenz_ylag10.npy',allow_pickle=True)
tau_ylag20,R_ylag20,error_tau_ylag20,error_R_ylag20,sig_tau_ylag20,sig_R_ylag20 = np.load('Lorenz_ylag20.npy',allow_pickle=True)
tau_ylag30,R_ylag30,error_tau_ylag30,error_R_ylag30,sig_tau_ylag30,sig_R_ylag30 = np.load('Lorenz_ylag30.npy',allow_pickle=True)
tau_ylag40,R_ylag40,error_tau_ylag40,error_R_ylag40,sig_tau_ylag40,sig_R_ylag40 = np.load('Lorenz_ylag40.npy',allow_pickle=True)
tau_ylag50,R_ylag50,error_tau_ylag50,error_R_ylag50,sig_tau_ylag50,sig_R_ylag50 = np.load('Lorenz_ylag50.npy',allow_pickle=True)
tau_ylag100,R_ylag100,error_tau_ylag100,error_R_ylag100,sig_tau_ylag100,sig_R_ylag100 = np.load('Lorenz_ylag100.npy',allow_pickle=True)
tau_ylag_10,R_ylag_10,error_tau_ylag_10,error_R_ylag_10,sig_tau_ylag_10,sig_R_ylag_10 = np.load('Lorenz_ylag-10.npy',allow_pickle=True)
tau_ylag_20,R_ylag_20,error_tau_ylag_20,error_R_ylag_20,sig_tau_ylag_20,sig_R_ylag_20 = np.load('Lorenz_ylag-20.npy',allow_pickle=True)
tau_ylag_30,R_ylag_30,error_tau_ylag_30,error_R_ylag_30,sig_tau_ylag_30,sig_R_ylag_30 = np.load('Lorenz_ylag-30.npy',allow_pickle=True)
tau_ylag_40,R_ylag_40,error_tau_ylag_40,error_R_ylag_40,sig_tau_ylag_40,sig_R_ylag_40 = np.load('Lorenz_ylag-40.npy',allow_pickle=True)
tau_ylag_50,R_ylag_50,error_tau_ylag_50,error_R_ylag_50,sig_tau_ylag_50,sig_R_ylag_50 = np.load('Lorenz_ylag-50.npy',allow_pickle=True)
tau_ylag_100,R_ylag_100,error_tau_ylag_100,error_R_ylag_100,sig_tau_ylag_100,sig_R_ylag_100 = np.load('Lorenz_ylag-100.npy',allow_pickle=True)
tau_zlag10,R_zlag10,error_tau_zlag10,error_R_zlag10,sig_tau_zlag10,sig_R_zlag10 = np.load('Lorenz_zlag10.npy',allow_pickle=True)
tau_zlag20,R_zlag20,error_tau_zlag20,error_R_zlag20,sig_tau_zlag20,sig_R_zlag20 = np.load('Lorenz_zlag20.npy',allow_pickle=True)
tau_zlag30,R_zlag30,error_tau_zlag30,error_R_zlag30,sig_tau_zlag30,sig_R_zlag30 = np.load('Lorenz_zlag30.npy',allow_pickle=True)
tau_zlag40,R_zlag40,error_tau_zlag40,error_R_zlag40,sig_tau_zlag40,sig_R_zlag40 = np.load('Lorenz_zlag40.npy',allow_pickle=True)
tau_zlag50,R_zlag50,error_tau_zlag50,error_R_zlag50,sig_tau_zlag50,sig_R_zlag50 = np.load('Lorenz_zlag50.npy',allow_pickle=True)
tau_zlag100,R_zlag100,error_tau_zlag100,error_R_zlag100,sig_tau_zlag100,sig_R_zlag100 = np.load('Lorenz_zlag100.npy',allow_pickle=True)
tau_zlag_10,R_zlag_10,error_tau_zlag_10,error_R_zlag_10,sig_tau_zlag_10,sig_R_zlag_10 = np.load('Lorenz_zlag-10.npy',allow_pickle=True)
tau_zlag_20,R_zlag_20,error_tau_zlag_20,error_R_zlag_20,sig_tau_zlag_20,sig_R_zlag_20 = np.load('Lorenz_zlag-20.npy',allow_pickle=True)
tau_zlag_30,R_zlag_30,error_tau_zlag_30,error_R_zlag_30,sig_tau_zlag_30,sig_R_zlag_30 = np.load('Lorenz_zlag-30.npy',allow_pickle=True)
tau_zlag_40,R_zlag_40,error_tau_zlag_40,error_R_zlag_40,sig_tau_zlag_40,sig_R_zlag_40 = np.load('Lorenz_zlag-40.npy',allow_pickle=True)
tau_zlag_50,R_zlag_50,error_tau_zlag_50,error_R_zlag_50,sig_tau_zlag_50,sig_R_zlag_50 = np.load('Lorenz_zlag-50.npy',allow_pickle=True)
tau_zlag_100,R_zlag_100,error_tau_zlag_100,error_R_zlag_100,sig_tau_zlag_100,sig_R_zlag_100 = np.load('Lorenz_zlag-100.npy',allow_pickle=True)

# Create tau vectors - xlag
tau_xy_xlag = np.array([tau_xlag100[0,1],tau_xlag50[0,1],tau_xlag40[0,1],tau_xlag30[0,1],tau_xlag20[0,1],tau_xlag10[0,1],tau[0,1],tau_xlag_10[0,1],tau_xlag_20[0,1],tau_xlag_30[0,1],tau_xlag_40[0,1],tau_xlag_50[0,1],tau_xlag_100[0,1]])
tau_yx_xlag = np.array([tau_xlag100[1,0],tau_xlag50[1,0],tau_xlag40[1,0],tau_xlag30[1,0],tau_xlag20[1,0],tau_xlag10[1,0],tau[1,0],tau_xlag_10[1,0],tau_xlag_20[1,0],tau_xlag_30[1,0],tau_xlag_40[1,0],tau_xlag_50[1,0],tau_xlag_100[1,0]])
tau_xz_xlag = np.array([tau_xlag100[0,2],tau_xlag50[0,2],tau_xlag40[0,2],tau_xlag30[0,2],tau_xlag20[0,2],tau_xlag10[0,2],tau[0,2],tau_xlag_10[0,2],tau_xlag_20[0,2],tau_xlag_30[0,2],tau_xlag_40[0,2],tau_xlag_50[0,2],tau_xlag_100[0,2]])
tau_zx_xlag = np.array([tau_xlag100[2,0],tau_xlag50[2,0],tau_xlag40[2,0],tau_xlag30[2,0],tau_xlag20[2,0],tau_xlag10[2,0],tau[2,0],tau_xlag_10[2,0],tau_xlag_20[2,0],tau_xlag_30[2,0],tau_xlag_40[2,0],tau_xlag_50[2,0],tau_xlag_100[2,0]])
tau_yz_xlag = np.array([tau_xlag100[1,2],tau_xlag50[1,2],tau_xlag40[1,2],tau_xlag30[1,2],tau_xlag20[1,2],tau_xlag10[1,2],tau[1,2],tau_xlag_10[1,2],tau_xlag_20[1,2],tau_xlag_30[1,2],tau_xlag_40[1,2],tau_xlag_50[1,2],tau_xlag_100[1,2]])
tau_zy_xlag = np.array([tau_xlag100[2,1],tau_xlag50[2,1],tau_xlag40[2,1],tau_xlag30[2,1],tau_xlag20[2,1],tau_xlag10[2,1],tau[2,1],tau_xlag_10[2,1],tau_xlag_20[2,1],tau_xlag_30[2,1],tau_xlag_40[2,1],tau_xlag_50[2,1],tau_xlag_100[2,1]])

# Create tau vectors - ylag
tau_xy_ylag = np.array([tau_ylag100[0,1],tau_ylag50[0,1],tau_ylag40[0,1],tau_ylag30[0,1],tau_ylag20[0,1],tau_ylag10[0,1],tau[0,1],tau_ylag_10[0,1],tau_ylag_20[0,1],tau_ylag_30[0,1],tau_ylag_40[0,1],tau_ylag_50[0,1],tau_ylag_100[0,1]])
tau_yx_ylag = np.array([tau_ylag100[1,0],tau_ylag50[1,0],tau_ylag40[1,0],tau_ylag30[1,0],tau_ylag20[1,0],tau_ylag10[1,0],tau[1,0],tau_ylag_10[1,0],tau_ylag_20[1,0],tau_ylag_30[1,0],tau_ylag_40[1,0],tau_ylag_50[1,0],tau_ylag_100[1,0]])
tau_xz_ylag = np.array([tau_ylag100[0,2],tau_ylag50[0,2],tau_ylag40[0,2],tau_ylag30[0,2],tau_ylag20[0,2],tau_ylag10[0,2],tau[0,2],tau_ylag_10[0,2],tau_ylag_20[0,2],tau_ylag_30[0,2],tau_ylag_40[0,2],tau_ylag_50[0,2],tau_ylag_100[0,2]])
tau_zx_ylag = np.array([tau_ylag100[2,0],tau_ylag50[2,0],tau_ylag40[2,0],tau_ylag30[2,0],tau_ylag20[2,0],tau_ylag10[2,0],tau[2,0],tau_ylag_10[2,0],tau_ylag_20[2,0],tau_ylag_30[2,0],tau_ylag_40[2,0],tau_ylag_50[2,0],tau_ylag_100[2,0]])
tau_yz_ylag = np.array([tau_ylag100[1,2],tau_ylag50[1,2],tau_ylag40[1,2],tau_ylag30[1,2],tau_ylag20[1,2],tau_ylag10[1,2],tau[1,2],tau_ylag_10[1,2],tau_ylag_20[1,2],tau_ylag_30[1,2],tau_ylag_40[1,2],tau_ylag_50[1,2],tau_ylag_100[1,2]])
tau_zy_ylag = np.array([tau_ylag100[2,1],tau_ylag50[2,1],tau_ylag40[2,1],tau_ylag30[2,1],tau_ylag20[2,1],tau_ylag10[2,1],tau[2,1],tau_ylag_10[2,1],tau_ylag_20[2,1],tau_ylag_30[2,1],tau_ylag_40[2,1],tau_ylag_50[2,1],tau_ylag_100[2,1]])

# Create tau vectors - zlag
tau_xy_zlag = np.array([tau_zlag100[0,1],tau_zlag50[0,1],tau_zlag40[0,1],tau_zlag30[0,1],tau_zlag20[0,1],tau_zlag10[0,1],tau[0,1],tau_zlag_10[0,1],tau_zlag_20[0,1],tau_zlag_30[0,1],tau_zlag_40[0,1],tau_zlag_50[0,1],tau_zlag_100[0,1]])
tau_yx_zlag = np.array([tau_zlag100[1,0],tau_zlag50[1,0],tau_zlag40[1,0],tau_zlag30[1,0],tau_zlag20[1,0],tau_zlag10[1,0],tau[1,0],tau_zlag_10[1,0],tau_zlag_20[1,0],tau_zlag_30[1,0],tau_zlag_40[1,0],tau_zlag_50[1,0],tau_zlag_100[1,0]])
tau_xz_zlag = np.array([tau_zlag100[0,2],tau_zlag50[0,2],tau_zlag40[0,2],tau_zlag30[0,2],tau_zlag20[0,2],tau_zlag10[0,2],tau[0,2],tau_zlag_10[0,2],tau_zlag_20[0,2],tau_zlag_30[0,2],tau_zlag_40[0,2],tau_zlag_50[0,2],tau_zlag_100[0,2]])
tau_zx_zlag = np.array([tau_zlag100[2,0],tau_zlag50[2,0],tau_zlag40[2,0],tau_zlag30[2,0],tau_zlag20[2,0],tau_zlag10[2,0],tau[2,0],tau_zlag_10[2,0],tau_zlag_20[2,0],tau_zlag_30[2,0],tau_zlag_40[2,0],tau_zlag_50[2,0],tau_zlag_100[2,0]])
tau_yz_zlag = np.array([tau_zlag100[1,2],tau_zlag50[1,2],tau_zlag40[1,2],tau_zlag30[1,2],tau_zlag20[1,2],tau_zlag10[1,2],tau[1,2],tau_zlag_10[1,2],tau_zlag_20[1,2],tau_zlag_30[1,2],tau_zlag_40[1,2],tau_zlag_50[1,2],tau_zlag_100[1,2]])
tau_zy_zlag = np.array([tau_zlag100[2,1],tau_zlag50[2,1],tau_zlag40[2,1],tau_zlag30[2,1],tau_zlag20[2,1],tau_zlag10[2,1],tau[2,1],tau_zlag_10[2,1],tau_zlag_20[2,1],tau_zlag_30[2,1],tau_zlag_40[2,1],tau_zlag_50[2,1],tau_zlag_100[2,1]])

# Create R vectors - xlag
R_xy_xlag = np.array([R_xlag100[0,1],R_xlag50[0,1],R_xlag40[0,1],R_xlag30[0,1],R_xlag20[0,1],R_xlag10[0,1],R[0,1],R_xlag_10[0,1],R_xlag_20[0,1],R_xlag_30[0,1],R_xlag_40[0,1],R_xlag_50[0,1],R_xlag_100[0,1]])
R_xz_xlag = np.array([R_xlag100[0,2],R_xlag50[0,2],R_xlag40[0,2],R_xlag30[0,2],R_xlag20[0,2],R_xlag10[0,2],R[0,2],R_xlag_10[0,2],R_xlag_20[0,2],R_xlag_30[0,2],R_xlag_40[0,2],R_xlag_50[0,2],R_xlag_100[0,2]])
R_yz_xlag = np.array([R_xlag100[1,2],R_xlag50[1,2],R_xlag40[1,2],R_xlag30[1,2],R_xlag20[1,2],R_xlag10[1,2],R[1,2],R_xlag_10[1,2],R_xlag_20[1,2],R_xlag_30[1,2],R_xlag_40[1,2],R_xlag_50[1,2],R_xlag_100[1,2]])

# Create R vectors - ylag
R_xy_ylag = np.array([R_ylag100[0,1],R_ylag50[0,1],R_ylag40[0,1],R_ylag30[0,1],R_ylag20[0,1],R_ylag10[0,1],R[0,1],R_ylag_10[0,1],R_ylag_20[0,1],R_ylag_30[0,1],R_ylag_40[0,1],R_ylag_50[0,1],R_ylag_100[0,1]])
R_xz_ylag = np.array([R_ylag100[0,2],R_ylag50[0,2],R_ylag40[0,2],R_ylag30[0,2],R_ylag20[0,2],R_ylag10[0,2],R[0,2],R_ylag_10[0,2],R_ylag_20[0,2],R_ylag_30[0,2],R_ylag_40[0,2],R_ylag_50[0,2],R_ylag_100[0,2]])
R_yz_ylag = np.array([R_ylag100[1,2],R_ylag50[1,2],R_ylag40[1,2],R_ylag30[1,2],R_ylag20[1,2],R_ylag10[1,2],R[1,2],R_ylag_10[1,2],R_ylag_20[1,2],R_ylag_30[1,2],R_ylag_40[1,2],R_ylag_50[1,2],R_ylag_100[1,2]])

# Create R vectors - zlag
R_xy_zlag = np.array([R_zlag100[0,1],R_zlag50[0,1],R_zlag40[0,1],R_zlag30[0,1],R_zlag20[0,1],R_zlag10[0,1],R[0,1],R_zlag_10[0,1],R_zlag_20[0,1],R_zlag_30[0,1],R_zlag_40[0,1],R_zlag_50[0,1],R_zlag_100[0,1]])
R_xz_zlag = np.array([R_zlag100[0,2],R_zlag50[0,2],R_zlag40[0,2],R_zlag30[0,2],R_zlag20[0,2],R_zlag10[0,2],R[0,2],R_zlag_10[0,2],R_zlag_20[0,2],R_zlag_30[0,2],R_zlag_40[0,2],R_zlag_50[0,2],R_zlag_100[0,2]])
R_yz_zlag = np.array([R_zlag100[1,2],R_zlag50[1,2],R_zlag40[1,2],R_zlag30[1,2],R_zlag20[1,2],R_zlag10[1,2],R[1,2],R_zlag_10[1,2],R_zlag_20[1,2],R_zlag_30[1,2],R_zlag_40[1,2],R_zlag_50[1,2],R_zlag_100[1,2]])

# Load results PCMCI
beta = np.load('PCMCI/tig5_LIANG_LM_all_XYZ_tau0-100',allow_pickle=True)

# Lag vectors
lag = np.array([-100,-50,-40,-30,-20,-10,0,10,20,30,40,50,100]) / 100. # Liang index and R
lag2 = np.array([0,10,20,30,40,50,60,70,80,90,100]) / 100. # PCMCI

# Plots
fig = plt.figure(figsize=(24,18))
fig.subplots_adjust(left=0.06,bottom=0.07,right=0.95,top=0.95,wspace=0.2,hspace=0.25)

# Scatter plot of R as a function of x lag
ax1 = fig.add_subplot(3,3,1)
ax1.plot(-lag[0:7],R_xy_xlag[0:7],'bo-',markersize=8,label=r'$R_{x_{t-l},y_t}$')
ax1.plot(-lag[0:7],R_xz_xlag[0:7],'ro-',markersize=8,label=r'$R_{x_{t-l},z_t}$')
ax1.tick_params(axis='both',labelsize=20)
ax1.legend(loc='upper right',fontsize=24)
ax1.set_ylabel('Correlation coefficient $R$',fontsize=24)
ax1.set_ylim([-0.08,1.05])
ax1.set_title('(a)',loc='left',fontsize=24,fontweight='bold')
    
# Scatter plot of R as a function of y lag
ax2 = fig.add_subplot(3,3,2) 
ax2.plot(-lag[0:7],R_xy_ylag[0:7],'bo-',markersize=8,label=r'$R_{y_{t-l},x_t}$')
ax2.plot(-lag[0:7],R_yz_ylag[0:7],'ko-',markersize=8,label=r'$R_{y_{t-l},z_t}$')
ax2.tick_params(axis='both',labelsize=20)
ax2.legend(loc='upper right',fontsize=24)
ax2.set_ylim([-0.08,1.05])
ax2.set_title('(b)',loc='left',fontsize=24,fontweight='bold')
    
# Scatter plot of R as a function of z lag
ax3 = fig.add_subplot(3,3,3)
ax3.plot(-lag[0:7],R_xz_zlag[0:7],'ro-',markersize=8,label=r'$R_{z_{t-l},x_t}$')
ax3.plot(-lag[0:7],R_yz_zlag[0:7],'ko-',markersize=8,label=r'$R_{z_{t-l},y_t}$')
ax3.tick_params(axis='both',labelsize=20)
ax3.legend(loc='upper right',fontsize=24)
ax3.set_ylim([-0.08,1.05])
ax3.set_title('(c)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of tau as a function of x lag
ax4 = fig.add_subplot(3,3,4)
ax4.plot(-lag[0:7],np.abs(tau_xy_xlag[0:7]),'bo-',markersize=8,label=r'$\|\tau\|_{x_{t-l} \longrightarrow y_t}$')
ax4.plot(-lag[0:7],np.abs(tau_xz_xlag[0:7]),'ro-',markersize=8,label=r'$\|\tau\|_{x_{t-l} \longrightarrow z_t}$')
ax4.legend(loc='upper right',fontsize=24)
ax4.tick_params(axis='both',labelsize=20)
ax4.set_ylabel(r'Rate of information transfer $\|\tau\|$ ($\%$)',fontsize=24)
ax4.set_ylim([-3,55])
ax4.set_title('(d)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of tau as a function of y lag
ax5 = fig.add_subplot(3,3,5)
ax5.plot(-lag[0:7],np.abs(tau_yx_ylag[0:7]),'bo-',markersize=8,label=r'$\|\tau\|_{y_{t-l} \longrightarrow x_t}$')
ax5.plot(-lag[0:7],np.abs(tau_yz_ylag[0:7]),'ko-',markersize=8,label=r'$\|\tau\|_{y_{t-l} \longrightarrow z_t}$')
ax5.legend(loc='upper right',fontsize=24)
ax5.tick_params(axis='both',labelsize=20)
ax5.set_ylim([-3,55])
ax5.set_title('(e)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of tau as a function of z lag
ax6 = fig.add_subplot(3,3,6)
ax6.plot(-lag[0:7],np.abs(tau_zx_zlag[0:7]),'rx-',markersize=8,label=r'$\|\tau\|_{z_{t-l} \longrightarrow x_t}$')
ax6.plot(-lag[0:7],np.abs(tau_zy_zlag[0:7]),'kx-',markersize=8,label=r'$\|\tau\|_{z_{t-l} \longrightarrow y_t}$')
ax6.legend(loc='upper right',fontsize=24)
ax6.tick_params(axis='both',labelsize=20)
ax6.set_ylim([-3,55])
ax6.set_title('(f)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of beta as a function of x lag
ax7 = fig.add_subplot(3,3,7)
ax7.plot(lag2,beta[0,1,::10],'bo--',markersize=8,label=r'$\beta_{x_{t-l} \longrightarrow y_t}$')
ax7.plot(lag2,beta[0,2,::10],'ro--',markersize=8,label=r'$\beta_{x_{t-l} \longrightarrow z_t}$')
#ax7.plot(np.arange(101)/100,beta[0,1,:],'b--',label=r'$\beta_{x_{t-l} \longrightarrow y_t}$')
#ax7.plot(np.arange(101)/100,beta[0,2,:],'r--',label=r'$\beta_{x_{t-l} \longrightarrow z_t}$')
ax7.legend(loc='upper right',fontsize=24)
ax7.tick_params(axis='both',labelsize=20)
ax7.set_xlabel('Lag $l$ in $x$ (unit times)',fontsize=24)
ax7.set_ylabel(r'Path coefficient $\beta$',fontsize=24)
ax7.set_ylim([-0.35,1])
ax7.set_title('(g)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of beta as a function of y lag
ax8 = fig.add_subplot(3,3,8)
ax8.plot(lag2,beta[1,0,::10],'bo--',markersize=8,label=r'$\beta_{y_{t-l} \longrightarrow x_t}$')
ax8.plot(lag2,beta[1,2,::10],'ko--',markersize=8,label=r'$\beta_{y_{t-l} \longrightarrow z_t}$')
#ax8.plot(np.arange(101)/100,beta[1,0,:],'b--',label=r'$\beta_{y_{t-l} \longrightarrow x_t}$')
#ax8.plot(np.arange(101)/100,beta[1,2,:],'k--',label=r'$\beta_{y_{t-l} \longrightarrow z_t}$')
ax8.legend(loc='upper right',fontsize=20)
ax8.tick_params(axis='both',labelsize=16)
ax8.set_xlabel('Lag $l$ in $y$ (unit times)',fontsize=24)
ax8.set_ylim([-0.015,0.02])
ax8.set_title('(h)',loc='left',fontsize=24,fontweight='bold')

# Scatter plot of beta as a function of z lag
ax9 = fig.add_subplot(3,3,9)
ax9.plot(lag2,beta[2,0,::10],'ro--',markersize=8,label=r'$\beta_{z_{t-l} \longrightarrow x_t}$')
ax9.plot(lag2,beta[2,1,::10],'ko--',markersize=8,label=r'$\beta_{z_{t-l} \longrightarrow y_t}$')
#ax9.plot(np.arange(101)/100,beta[2,0,:],'r--',label=r'$\beta_{z_{t-l} \longrightarrow x_t}$')
#ax9.plot(np.arange(101)/100,beta[2,1,:],'k--',label=r'$\beta_{z_{t-l} \longrightarrow y_t}$')
ax9.legend(loc='upper right',fontsize=20)
ax9.tick_params(axis='both',labelsize=16)
ax9.set_xlabel('Lag $l$ in $z$ (unit times)',fontsize=24)
ax9.set_ylim([-0.015,0.02])
ax9.set_title('(i)',loc='left',fontsize=24,fontweight='bold')

# Save figure
if save_fig == True:
    fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig6.png')