#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot results from 6D model - Correlation, LKIF and PCMCI

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
nvar = 6 # number of variables

# Load results LKIF
T,tau,R,error_T,error_tau,error_R,sig_T,sig_tau,sig_R = np.load('VAR_liang.npy',allow_pickle=True)

# Load results PCMCI
beta = np.load('PCMCI/tig5_LIANG_6D_all_X1X2X3X4X5X6_tau0-4',allow_pickle=True)
beta_max = np.nanmax(beta,axis=2)
beta_min = np.nanmin(beta,axis=2)
beta_val = np.zeros((nvar,nvar))
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if beta_max[j,i] >= np.abs(beta_min[j,i]):
            beta_val[j,i] = beta_max[j,i]
        else:
            beta_val[j,i] = beta_min[j,i]

# Create graph
G = nx.DiGraph()
G.add_edges_from(
    [('$x_1$','$x_2$'),('$x_2$','$x_3$'),('$x_3$','$x_4$'),('$x_4$','$x_5$'),
     ('$x_5$','$x_6$'),('$x_6$','$x_1$')])
significant_edges = [('$x_1$','$x_2$'),('$x_2$','$x_3$'),('$x_3$','$x_1$'),('$x_4$','$x_5$'),
                    ('$x_5$','$x_4$'),('$x_6$','$x_2$'),('$x_6$','$x_5$')]

# Matrix of correct links
matrix_correct = np.zeros((nvar,nvar))
for i in np.arange(nvar):
    for j in np.arange(nvar):
        if ((i == 0 and j == 2)
            or (i == 1 and (j == 0 or j == 5))
            or (i == 2 and j == 1)
            or (i == 3 and (j == 3 or j == 4))
            or (i == 4 and (j == 3 or j == 5))
            or (i == 5 and j == 5)):
                matrix_correct[j,i] = 1

# Matrices of correlation and causality
fig,ax = plt.subplots(2,2,figsize=(24,24))
fig.subplots_adjust(left=0.05,bottom=0.01,right=0.95,top=0.9,wspace=0.15,hspace=None)
cmap_tau = plt.cm.YlOrRd._resample(16)
cmap_beta = plt.cm.bwr._resample(16)
cmap_R = plt.cm.bwr._resample(16)
sns.set(font_scale=1.8)
label_names = ['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','$x_6$']

# Matrix of R
R_annotations_init = np.round(R,2)
R_annotations = R_annotations_init.astype(str)
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if R[j,i] < 0.005 and R[j,i] > -0.005:
            R_annotations[j,i] = '{:.0e}'.format(R[j,i])
R[sig_R==0] = np.nan
R_plot = sns.heatmap(R,annot=R_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_R,
    cbar_kws={'label':'$R$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,0])
R_plot.set_title('Correlation coefficient $R$ \n ',fontsize=32)
R_plot.set_title('(a) \n',loc='left',fontsize=32,fontweight='bold')
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

# Matrix of tau
tau_annotations_init = np.round(np.abs(tau),2)
tau_annotations = tau_annotations_init.astype(str)
tau[sig_tau==0] = np.nan
tau_plot = sns.heatmap(np.abs(tau),annot=tau_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_tau,
    cbar_kws={'label':r'$\|\tau\|$ ($\%$)','orientation':'horizontal','pad':0.05},vmin=0,vmax=20,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[0,1])
tau_plot.set_title(r'Rate of information transfer $\|\tau\|$ (LKIF)' + '\n',fontsize=32)
tau_plot.set_title('(b) \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if sig_tau[j,i] == 1:
            if matrix_correct[j,i] == 1:
                tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3))
#            else:
#                tau_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='blue',linewidth=3,linestyle='--'))
tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=28)
tau_plot.xaxis.set_ticks_position('top')
tau_plot.set_xlabel('TO...',loc='left',fontsize=24)
tau_plot.xaxis.set_label_position('top')
tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=28)
tau_plot.set_ylabel('FROM...',loc='top',fontsize=24)       

# Matrix of beta
beta_annotations_init = np.round(beta_val,2)
beta_annotations = beta_annotations_init.astype(str)
beta_val[beta_val==0] = np.nan
beta_plot = sns.heatmap(beta_val,annot=beta_annotations,fmt='',annot_kws={'color':'k','fontsize':24},cmap=cmap_beta,
    cbar_kws={'label':r'$\beta$','orientation':'horizontal','pad':0.05},vmin=-1,vmax=1,
    xticklabels=label_names,yticklabels=label_names,linewidths=0.1,linecolor='gray',ax=ax[1,0])
beta_plot.set_title(r'Path coefficient $\beta$ (PCMCI)' + '\n',fontsize=32)
beta_plot.set_title('(c) \n',loc='left',fontsize=32,fontweight='bold')
for j in np.arange(nvar):
    for i in np.arange(nvar):
        if np.abs(beta_val[j,i]) > 0:
            if matrix_correct[j,i] == 1:
                beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3))
#            else:
#                beta_plot.add_patch(Rectangle((i+0.05,j+0.2),0.9,0.6,fill=False,edgecolor='black',linewidth=3,linestyle='--'))
beta_plot.set_xticklabels(tau_plot.get_xmajorticklabels(),fontsize=28)
beta_plot.xaxis.set_ticks_position('top')
beta_plot.set_xlabel('TO...',loc='left',fontsize=24)
beta_plot.xaxis.set_label_position('top')
beta_plot.set_yticklabels(tau_plot.get_ymajorticklabels(),fontsize=28)
beta_plot.set_ylabel('FROM...',loc='top',fontsize=24)

# Correct causal links
pos=nx.spring_layout(G)
nx.draw_networkx_nodes(G,pos,node_color='lightblue',node_size=5000,node_shape='o',alpha=0.4,ax=ax[1,1])
nx.draw_networkx_labels(G,pos,font_size=40,ax=ax[1,1])
nx.draw_networkx_edges(G,pos,edgelist=significant_edges,edge_color='k',arrows=True,arrowsize=40,connectionstyle='arc3,rad=0.2',width=3,min_source_margin=30,min_target_margin=30,ax=ax[1,1])
ax[1,1].set_title('(d) \n',loc='left',fontsize=32,fontweight='bold')
ax[1,1].set_title('Correct links \n',loc='center',fontsize=32) 

# Save figure
if save_fig == True:
    fig.savefig('/home/dadocq/Documents/Papers/My_Papers/Causal_Comp/LaTeX/fig2.png')