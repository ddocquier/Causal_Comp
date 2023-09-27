#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results from 2D model - Correlation, LKIF and PCMCI

Last updated: 19/09/2023

@author: David Docquier
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # for creating a matrix plot
from matplotlib.patches import Rectangle # for drawing rectangles around elements in a matrix
import networkx as nx

# Options
save_fig = True
nvar = 2 # number of variables

# Load numerical results LKIF
T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R = np.load('2D_liang.npy',allow_pickle=True)

# Load numerical results PCMCI with original dt = 0.001
beta = np.load('PCMCI/tig5_LIANG_2D_all_X1X2_tau0-12',allow_pickle=True)
beta_max = np.nanmax(beta[:,:,0:3],axis=2)
beta_min = np.nanmin(beta[:,:,0:3],axis=2)
beta_val = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_max[j,i] >= np.abs(beta_min[j,i]):
            beta_val[j,i] = beta_max[j,i]
        else:
            beta_val[j,i] = beta_min[j,i]
            
# Load numerical results PCMCI with dt = 0.1 (every 100 time steps)
beta2 = np.load('PCMCI/tig5_LIANG_2D_100tmask_X1X2_tau0-2',allow_pickle=True)
beta_max2 = np.nanmax(beta2[:,:,:],axis=2)
beta_min2 = np.nanmin(beta2[:,:,:],axis=2)
beta_val2 = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_max2[j,i] >= np.abs(beta_min2[j,i]):
            beta_val2[j,i] = beta_max2[j,i]
        else:
            beta_val2[j,i] = beta_min2[j,i]

# Load analytical results LKIF
dir_anal = '/home/dadocq/Documents/Codes/Liang/'
tau21_anal,tau12_anal,R12_anal = np.load(dir_anal+'2D_Analytical.npy',allow_pickle=True)
tau_anal = np.zeros((nvar,nvar))
R_anal = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        if i == 0 and j == 1:
            tau_anal[i,j] = tau12_anal
            R_anal[i,j] = R12_anal
        elif i == 1 and j == 0:
            tau_anal[i,j] = tau21_anal
            R_anal[i,j] = R12_anal
        elif i == j:
            tau_anal[i,j] = np.nan
            R_anal[i,j] = 1

# Create graph
G = nx.DiGraph()
G.add_edges_from(
    [('$x_1$','$x_2$')])
significant_edges = [('$x_2$','$x_1$')]

# Matrix of correct links
matrix_correct = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        if ((i == 0 and j == 1) or (i == j)):
            matrix_correct[j,i] = 1

# Matrices of correlation and causality
fig,ax = plt.subplots(2,3,figsize=(20,15))
fig.subplots_adjust(left=0.1,bottom=0.08,right=0.95,top=0.87,wspace=0.3,hspace=0.5)
cmap_tau = plt.cm.YlOrRd._resample(16)
cmap_beta = plt.cm.bwr._resample(16)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)
label_names = ['$x_1$','$x_2$']

# Matrix of numerical R
R_annotations_init = np.round(R,2)
R_annotations = R_annotations_init.astype(str)
R_annotations[sig_R==0] = '-'
R_plot = sns.heatmap(R,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,0])
R_plot.set_title('Numerical $R$ \n',fontsize=25)
R_plot.set_title('(a) \n',loc='left',fontsize=25,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if sig_R[j,i] == 1:
            if matrix_correct[j,i] == 1:
                R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
#            else:
#                R_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3,linestyle='--'))
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=28)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=28)

# Matrix of numerical tau
tau_annotations_init = np.round(np.abs(tau),2)
tau_annotations = tau_annotations_init.astype(str)
tau[sig_tau==0] = np.nan
tau_plot = sns.heatmap(np.abs(tau),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=20,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,1])
tau_plot.set_title(r'Numerical $\|\tau\|$ (LKIF)' + '\n',fontsize=25)
tau_plot.set_title('(b) \n',loc='left',fontsize=25,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if sig_tau[j,i] == 1:
            if matrix_correct[j,i] == 1:
                tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
#            else:
#                tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3,linestyle='--'))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=28)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=28)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)

# Matrix of numerical beta
beta_annotations_init = np.round(beta_val,2)
beta_annotations = beta_annotations_init.astype(str)
beta_val[1,0] = beta_val2[1,0]
beta_val[beta_val==0] = np.nan
beta_annotations[1,0] = np.round(beta_val2[1,0],2).astype(str) + '\n with d$t$=0.1'
beta_plot = sns.heatmap(beta_val,annot=beta_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_beta,
    cbar_kws={'label':r'$\beta$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,2])
beta_plot.set_title(r'Numerical $\beta$ (PCMCI)' + '\n',fontsize=25)
beta_plot.set_title('(c) \n',loc='left',fontsize=25,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_val[j,i] > 0:
            if j == 1 and i == 0:
                beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3,linestyle='--'))
            else:
                if matrix_correct[j,i] == 1:
                    beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
#            else:
#                beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3,linestyle='--'))
beta_plot.set_xticklabels(beta_plot.get_xmajorticklabels(),fontsize=28)
beta_plot.xaxis.set_ticks_position('top')
beta_plot.set_xlabel('TO...',loc='left',fontsize=20)
beta_plot.xaxis.set_label_position('top')
beta_plot.set_yticklabels(beta_plot.get_ymajorticklabels(),fontsize=28)
beta_plot.set_ylabel('FROM...',loc='top',fontsize=20)

# Matrix of analytical R
R_annotations = np.round(R_anal,2)
R_plot = sns.heatmap(R_anal,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_R,
    cbar_kws={'label':'$R$ \n','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[1,0])
R_plot.set_title('Analytical $R$ \n',fontsize=25)
R_plot.set_title('(d) \n',loc='left',fontsize=25,fontweight='bold')
R_plot.set_xticklabels(R_plot.get_xmajorticklabels(),fontsize=28)
R_plot.xaxis.set_ticks_position('top')
R_plot.set_yticklabels(R_plot.get_ymajorticklabels(),fontsize=28)

# Matrix of analytical tau
tau_annotations = np.round(np.abs(tau_anal),2)
tau_plot = sns.heatmap(np.abs(tau_anal),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=20,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[1,1])
tau_plot.set_title(r'Analytical $\|\tau\|$ (LKIF)' + '\n',fontsize=25)
tau_plot.set_title('(e) \n',loc='left',fontsize=25,fontweight='bold')
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=28)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=20)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=28)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=20)

#fig.delaxes(ax[1,2])

# Correct links 
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,node_color='lightblue',node_size=3000,node_shape='o',alpha=0.4,ax=ax[1,2])
nx.draw_networkx_labels(G,pos,font_size=40,ax=ax[1,2])
nx.draw_networkx_edges(G,pos,edgelist=significant_edges,edge_color='k',arrows=True,arrowsize=40,connectionstyle='arc3,rad=0.2',width=3,min_source_margin=30,min_target_margin=30,ax=ax[1,2])
ax[1,2].set_title('(f) \n',loc='left',fontsize=25,fontweight='bold')
ax[1,2].set_title('Correct links \n',loc='center',fontsize=25)

# Save figure
if save_fig == True:
    fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig1.png')