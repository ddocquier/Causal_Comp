#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results from extension of Vannitsem & Liang (2022) - Correlation, LKIF and PCMCI
With variables downloaded by DD on 20/01/2023

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
nvar = 8

# Load results
filename = 'real_case_study_liang.npy'
T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R = np.load(filename,allow_pickle=True)

# Load results PCMCI
beta = np.load('PCMCI/tig5_LIANG_all_PNANAOAOQBOAMOPDOTNAnino34_causal_links_array_tau0-11',allow_pickle=True)
beta_max = np.nanmax(beta[:,:,0:3],axis=2)
beta_min = np.nanmin(beta[:,:,0:3],axis=2)
beta_val = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_max[j,i] >= np.abs(beta_min[j,i]):
            beta_val[j,i] = beta_max[j,i]
        else:
            beta_val[j,i] = beta_min[j,i]

# Matrices of correlation and causality
fig,ax = plt.subplots(2,2,figsize=(24,24))
fig.subplots_adjust(left=0.05,bottom=0.01,right=0.95,top=0.9,wspace=0.15,hspace=None)
cmap_tau = plt.cm.YlOrRd._resample(16)
cmap_beta = plt.cm.bwr._resample(16)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)
label_names = ['PNA','NAO','AO','QBO','AMO','PDO','TNA','ENSO']

# Matrix of R
R_annotations_init = np.round(R,2)
R_annotations = R_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if R[j,i] < 0.004 and R[j,i] > -0.004:
            R_annotations[j,i] = '{:.0e}'.format(R[j,i])
R[sig_R==0] = np.nan
text_size = 24
R_plot = sns.heatmap(R,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':text_size},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,0])
R_plot.set_title('Correlation coefficient $R$ \n ',fontsize=30)
R_plot.set_title('(a) \n',loc='left',fontsize=30,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if sig_R[j,i] == 1: 
#            R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=24)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=24)

# Matrix of tau
tau_annotations_init = np.round(np.abs(tau),2)
tau_annotations = tau_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if np.abs(tau[j,i]) < 0.005:
            tau_annotations[j,i] = '{:.0e}'.format(np.abs(tau[j,i]))
tau[sig_tau==0] = np.nan
tau_plot = sns.heatmap(np.abs(tau),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':text_size},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=20,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,1])
tau_plot.set_title(r'Rate of information transfer $\|\tau\|$' + '\n',fontsize=30)
tau_plot.set_title('(b) \n',loc='left',fontsize=30,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if sig_tau[j,i] == 1:
#            tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=24)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=24)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=24)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=24)

fig.delaxes(ax[1,0])

# Matrix of beta
beta_annotations_init = np.round(beta_val,2)
beta_annotations = beta_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_val[j,i] < 0.005 and beta_val[j,i] > -0.005:
            beta_annotations[j,i] = '{:.0e}'.format(beta_val[j,i])
beta_val[beta_val==0] = np.nan
beta_plot = sns.heatmap(beta_val,annot=beta_annotations,fmt='',annot_kws={'color':'k','fontsize':text_size},cmap=cmap_beta,
    cbar_kws={'label':r'$\beta$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[1,1])
beta_plot.set_title(r'Path coefficient $\beta$ (PCMCI)' + '\n',fontsize=30)
beta_plot.set_title('(c) \n',loc='left',fontsize=30,fontweight='bold')
#for j in np.arange(nvar):
#    for i in np.arange(nvar):
#        if np.abs(beta_val[j,i]) > 0:
#            beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
beta_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=24)
beta_plot.xaxis.set_ticks_position('top')
beta_plot.set_xlabel('TO...',loc='left',fontsize=24)
beta_plot.xaxis.set_label_position('top')
beta_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=24)
beta_plot.set_ylabel('FROM...',loc='top',fontsize=24)   

# Save figure
if save_fig == True:
    fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig8.png')