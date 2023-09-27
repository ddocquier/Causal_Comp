#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results from Lorenz model - Correlation, LKIF and PCMCI

Last updated: 19/09/2023

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for creating a matrix plot
from matplotlib.patches import Rectangle # for drawing rectangles around elements in a matrix

# Options
save_fig = True
nvar = 3
sampling = 10

# Load results LKIF x,y,z
if sampling == 1:
    filename = 'Lorenz_nolag.npy'
elif sampling == 10:
    filename = 'Lorenz_nolag_samp10.npy'
tau,R,error_tau,error_R,sig_tau,sig_R = np.load(filename,allow_pickle=True)

# Load results LKIF x^2,y,z
if sampling == 1:
    filename = 'Lorenz_xsquare.npy'
elif sampling == 10:
    filename = 'Lorenz_xsquare_samp10.npy'
tau_xsquare,R_xsquare,error_tau_xsquare,error_R_xsquare,sig_tau_xsquare,sig_R_xsquare = np.load(filename,allow_pickle=True)

# Load results PCMCI x,y,z
if sampling == 1:
    beta = np.load('PCMCI/tig5_LIANG_LM_all_XYZ_tau0-100',allow_pickle=True)
elif sampling == 10:
    beta = np.load('PCMCI/tig5_LIANG_LM_XYZ_tau0-12',allow_pickle=True)
beta_max = np.nanmax(beta[:,:,0:4],axis=2)
beta_min = np.nanmin(beta[:,:,0:4],axis=2)
beta_val = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_max[j,i] >= np.abs(beta_min[j,i]):
            beta_val[j,i] = beta_max[j,i]
        else:
            beta_val[j,i] = beta_min[j,i]

# Load results PCMCI x^2,y,z
beta2 = np.load('PCMCI/tig5_LIANG_LM_all_x2_X2YZ_tau0-100',allow_pickle=True)
beta_max2 = np.nanmax(beta2[:,:,0:4],axis=2)
beta_min2 = np.nanmin(beta2[:,:,0:4],axis=2)
beta_val2 = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_max2[j,i] >= np.abs(beta_min2[j,i]):
            beta_val2[j,i] = beta_max2[j,i]
        else:
            beta_val2[j,i] = beta_min2[j,i]

# Matrices of correlation and causality
fig,ax = plt.subplots(2,3,figsize=(20,15))
fig.subplots_adjust(left=0.05,bottom=0.05,right=0.92,top=0.88,wspace=0.3,hspace=0.4)
cmap_tau = plt.cm.YlOrRd._resample(16)
cmap_beta = plt.cm.bwr._resample(16)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)
label_names = ['$x$','$y$','$z$']
label_names_xsquare = ['$x^2$','$y$','$z$']

# Matrix of R
R_annotations_init = np.round(R,2)
R_annotations = R_annotations_init.astype(str)
R[sig_R==0] = np.nan
R_plot = sns.heatmap(R,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':20},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,0])
R_plot.set_title('Correlation coefficient $R$ \n ',fontsize=20)
R_plot.set_title('(a) \n',loc='left',fontsize=20,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if sig_R[j,i] == 1: 
#            R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=20)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=20)

# Matrix of tau
tau_annotations_init = np.round(np.abs(tau),2)
tau_annotations = tau_annotations_init.astype(str)
tau[sig_tau==0] = np.nan
tau_plot = sns.heatmap(np.abs(tau),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':20},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=20,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,1])
tau_plot.set_title(r'Rate of information transfer $\|\tau\|$' + '\n',fontsize=18)
tau_plot.set_title('(b) \n',loc='left',fontsize=20,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if sig_tau[j,i] == 1:
#            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)

# Matrix of beta
beta_annotations_init = np.round(beta_val,2)
beta_annotations = beta_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_val[j,i] < 0.005 and beta_val[j,i] > -0.005:
            beta_annotations[j,i] = '{:.0e}'.format(beta_val[j,i])
beta_val[beta_val==0] = np.nan
beta_plot = sns.heatmap(beta_val,annot=beta_annotations,fmt='',annot_kws={'color':'k','fontsize':20},cmap=cmap_beta,
    cbar_kws={'label':r'$\beta$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,2])
beta_plot.set_title(r'Path coefficient $\beta$ (PCMCI)' + '\n',fontsize=20)
beta_plot.set_title('(c) \n',loc='left',fontsize=20,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if np.abs(beta_val[j,i]) > 0:
#            beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
beta_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
beta_plot.xaxis.set_ticks_position('top')
beta_plot.set_xlabel('TO...',loc='left',fontsize=20)
beta_plot.xaxis.set_label_position('top')
beta_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
beta_plot.set_ylabel('FROM...',loc='top',fontsize=20)   

# Matrix of R - x^2
R_annotations_init = np.round(R_xsquare,2)
R_annotations = R_annotations_init.astype(str)
R_xsquare[sig_R_xsquare==0] = np.nan
R_plot = sns.heatmap(R_xsquare,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':20},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names_xsquare,yticklabels=label_names_xsquare,linewidths=0.1,linecolor='gray',ax=ax[1,0])
R_plot.set_title('Correlation coefficient $R$ \n ',fontsize=20)
R_plot.set_title('(d) \n',loc='left',fontsize=20,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if sig_R_xsquare[j,i] == 1: 
#            R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=20)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=20)

# Matrix of tau - x^2
tau_annotations_init = np.round(np.abs(tau_xsquare),2)
tau_annotations = tau_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if np.abs(tau_xsquare[j,i]) < 0.004:
            tau_annotations[j,i] = '{:.0e}'.format(np.abs(tau_xsquare[j,i]))
tau_xsquare[sig_tau_xsquare==0] = np.nan
tau_plot = sns.heatmap(np.abs(tau_xsquare),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':20},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=20,
    xticklabels=label_names_xsquare,yticklabels=label_names_xsquare,linewidths=0.1,linecolor='gray',ax=ax[1,1])
tau_plot.set_title(r'Rate of information transfer $\|\tau\|$' + '\n',fontsize=18)
tau_plot.set_title('(e) \n',loc='left',fontsize=20,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if sig_tau_xsquare[j,i] == 1:
#            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)

# Matrix of beta
beta_annotations_init = np.round(beta_val2,2)
beta_annotations = beta_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_val2[j,i] < 0.004 and beta_val2[j,i] > -0.004:
            beta_annotations[j,i] = '{:.0e}'.format(beta_val2[j,i])
beta_val2[beta_val2==0] = np.nan
beta_plot = sns.heatmap(beta_val2,annot=beta_annotations,fmt='',annot_kws={'color':'k','fontsize':20},cmap=cmap_beta,
    cbar_kws={'label':r'$\beta$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names_xsquare,yticklabels=label_names_xsquare,linewidths=0.1,linecolor='gray',ax=ax[1,2])
beta_plot.set_title(r'Path coefficient $\beta$ (PCMCI)' + '\n',fontsize=20)
beta_plot.set_title('(f) \n',loc='left',fontsize=20,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if np.abs(beta_val2[j,i]) > 0:
#            beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
beta_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=20)
beta_plot.xaxis.set_ticks_position('top')
beta_plot.set_xlabel('TO...',loc='left',fontsize=20)
beta_plot.xaxis.set_label_position('top')
beta_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=20)
beta_plot.set_ylabel('FROM...',loc='top',fontsize=20) 

# Save figure
if save_fig == True:
    if sampling == 1:
        fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig5.png')
    elif sampling == 10:
        fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig5_samp10.png')