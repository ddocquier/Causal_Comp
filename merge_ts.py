#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge time series in 1 file + post-processing // Vannitsem & Liang (2022) with QBO added
Files downloaded from https://psl.noaa.gov/data/climateindices/list/ on 20/01/2023

Atmospheric indices:
PNA (Pacific North American index): difference in 500hPa geopotential height anomalies between western and eastern US, https://www.cpc.ncep.noaa.gov/data/teledoc/pna.shtml
NAO (North Atlantic Oscillation): difference in SLP between Subtropical High (Azores) and Subpolar Low (Iceland), https://www.cpc.ncep.noaa.gov/data/teledoc/nao.shtml
AO (Arctic Oscillation or Northern Annular Mode [NAM]): projection of 1000hPa geopotential height anomalies poleward of 20N onto the loading pattern of the AO, https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/ao.shtml
QBO (Quasi-Biennal Oscillation): zonal average of 30hPa zonal wind at the equator, https://www.metoffice.gov.uk/weather/learn-about/weather/atmosphere/quasi-biennial-oscillation

Ocean indices:
AMO (Atlantic Multidecadal Oscillation or Variability [AMV]): SST average over North Atlantic (0-70N), https://psl.noaa.gov/data/timeseries/AMO/
PDO (Pacific Decadal Oscillation or Variability [PDV]): projection of Pacific SST anomalies on the dominant EOF poleward of 20N, https://www.ncei.noaa.gov/access/monitoring/pdo/
TNA (Tropical North Atlantic index): dominant EOF-based reconstruction of SST anomalies in the Tropical North Atlantic
ENSO (El Niño Southern Oscillation), Niño3.4 (East Central Tropical Pacific SST): standardized anomaly of SST in the eastern Tropical Pacific, https://www.cpc.ncep.noaa.gov/products/precip/CWlink/MJO/enso.shtml

Last updated: 02/02/2023

@author: David Docquier
"""

# Import libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Options
save_var = False # True: save variables
save_fig = False

# Time parameters
nmy = 12 # number of months in a year

# Working directory
dir_input = '/home/dadocq/Documents/Observations/NOAA_Indices/'


# Load PNA
filename = dir_input + 'pna.data'
pna_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
nyears = np.size(pna_init,0)
nm = nyears * nmy
t = np.linspace(0,nm-1,nm)
pna = np.reshape(pna_init,nyears*nmy)

# Load NAO
filename = dir_input + 'nao.data'
nao_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
nao = np.reshape(nao_init,nyears*nmy)

# Load AO
filename = dir_input + 'ao.data'
ao_init = np.loadtxt(filename,skiprows=1,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
ao = np.reshape(ao_init,nyears*nmy)

# Load QBO
filename = dir_input + 'qbo.data'
qbo_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
qbo = np.reshape(qbo_init,nyears*nmy)

# Load AMO
filename = dir_input + 'amo.us.data'
amo_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
amo = np.reshape(amo_init,nyears*nmy)

# Load PDO
filename = dir_input + 'pdo.data'
pdo_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
pdo = np.reshape(pdo_init,nyears*nmy)

# Load TNA
filename = dir_input + 'tna.data'
tna_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
tna = np.reshape(tna_init,nyears*nmy)

# Load Nino3.4
filename = dir_input + 'nino34.anom.data'
nino34_init = np.loadtxt(filename,skiprows=3,max_rows=72,usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
nino34 = np.reshape(nino34_init,nyears*nmy)

# Remove global trend
pna_resid = pna - (stats.linregress(t,pna).intercept + stats.linregress(t,pna).slope * t)
nao_resid = nao - (stats.linregress(t,nao).intercept + stats.linregress(t,nao).slope * t)
ao_resid = ao - (stats.linregress(t,ao).intercept + stats.linregress(t,ao).slope * t)
qbo_resid = qbo - (stats.linregress(t,qbo).intercept + stats.linregress(t,qbo).slope * t)
amo_resid = amo - (stats.linregress(t,amo).intercept + stats.linregress(t,amo).slope * t)
pdo_resid = pdo - (stats.linregress(t,pdo).intercept + stats.linregress(t,pdo).slope * t)
tna_resid = tna - (stats.linregress(t,tna).intercept + stats.linregress(t,tna).slope * t)
nino34_resid = nino34 - (stats.linregress(t,nino34).intercept + stats.linregress(t,nino34).slope * t)

# Save variables
if save_var == True:
    filename = 'Indices_wotrend.npy'
    np.save(filename,[pna_resid,nao_resid,ao_resid,qbo_resid,amo_resid,pdo_resid,tna_resid,nino34_resid])

# Time series of original variables
fig,ax = plt.subplots(4,2,figsize=(24,28))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)
xrange = np.arange(0,864,120)
name_xticks = ['1950','1960','1970','1980','1990','2000','2010','2020']

# PNA
ax[0,0].plot(t,pna,'r')
ax[0,0].plot(t,stats.linregress(t,pna).intercept + stats.linregress(t,pna).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,pna).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,pna).pvalue) + ')')
ax[0,0].set_ylabel('PNA',fontsize=26)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].legend(loc='upper right',frameon=False,fontsize=18)
ax[0,0].tick_params(axis='both',labelsize=20)
ax[0,0].grid(linestyle='--')
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# NAO
ax[0,1].plot(t,nao,'r')
ax[0,1].plot(t,stats.linregress(t,nao).intercept + stats.linregress(t,nao).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,nao).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,nao).pvalue) + ')')
ax[0,1].set_ylabel('NAO',fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].legend(loc='upper right',frameon=False,fontsize=18)
ax[0,1].tick_params(axis='both',labelsize=20)
ax[0,1].grid(linestyle='--')
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# AO
ax[1,0].plot(t,ao,'r')
ax[1,0].plot(t,stats.linregress(t,ao).intercept + stats.linregress(t,ao).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,ao).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,ao).pvalue) + ')')
ax[1,0].set_ylabel('AO',fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].legend(loc='upper left',frameon=False,fontsize=18)
ax[1,0].tick_params(axis='both',labelsize=20)
ax[1,0].grid(linestyle='--')
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# QBO
ax[1,1].plot(t,qbo,'r')
ax[1,1].plot(t,stats.linregress(t,qbo).intercept + stats.linregress(t,qbo).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,qbo).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,qbo).pvalue) + ')')
ax[1,1].set_ylabel('QBO',fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].legend(loc='lower left',frameon=False,fontsize=18)
ax[1,1].tick_params(axis='both',labelsize=20)
ax[1,1].grid(linestyle='--')
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# AMO
ax[2,0].plot(t,amo,'r')
ax[2,0].plot(t,stats.linregress(t,amo).intercept + stats.linregress(t,amo).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,amo).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,amo).pvalue) + ')')
ax[2,0].set_ylabel('AMO',fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].legend(loc='upper left',frameon=False,fontsize=18)
ax[2,0].tick_params(axis='both',labelsize=20)
ax[2,0].grid(linestyle='--')
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# PDO
ax[2,1].plot(t,pdo,'r')
ax[2,1].plot(t,stats.linregress(t,pdo).intercept + stats.linregress(t,pdo).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,pdo).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,pdo).pvalue) + ')')
ax[2,1].set_ylabel('PDO',fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].legend(loc='lower right',frameon=False,fontsize=18)
ax[2,1].tick_params(axis='both',labelsize=20)
ax[2,1].grid(linestyle='--')
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# TNA
ax[3,0].plot(t,tna,'r')
ax[3,0].plot(t,stats.linregress(t,tna).intercept + stats.linregress(t,tna).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,tna).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,tna).pvalue) + ')')
ax[3,0].set_ylabel('TNA',fontsize=26)
ax[3,0].set_xlabel('Year',fontsize=26)
ax[3,0].set_xticks(xrange)
ax[3,0].set_xticklabels(name_xticks)
ax[3,0].legend(loc='upper left',frameon=False,fontsize=18)
ax[3,0].tick_params(axis='both',labelsize=20)
ax[3,0].grid(linestyle='--')
ax[3,0].set_title('g',loc='left',fontsize=30,fontweight='bold')

# Niño3.4
ax[3,1].plot(t,nino34,'r')
ax[3,1].plot(t,stats.linregress(t,nino34).intercept + stats.linregress(t,nino34).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,nino34).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,nino34).pvalue) + ')')
ax[3,1].set_ylabel('Niño3.4',fontsize=26)
ax[3,1].set_xlabel('Year',fontsize=26)
ax[3,1].set_xticks(xrange)
ax[3,1].set_xticklabels(name_xticks)
ax[3,1].legend(loc='upper left',frameon=False,fontsize=18)
ax[3,1].tick_params(axis='both',labelsize=20)
ax[3,1].grid(linestyle='--')
ax[3,1].set_title('h',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
if save_fig == True:
    fig.savefig('TimeSeries_original.jpg')


# Time series of detrended variables
fig,ax = plt.subplots(4,2,figsize=(24,28))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)
xrange = np.arange(0,864,120)
name_xticks = ['1950','1960','1970','1980','1990','2000','2010','2020']

# PNA
ax[0,0].plot(t,pna_resid,'r')
ax[0,0].plot(t,stats.linregress(t,pna_resid).intercept + stats.linregress(t,pna_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,pna_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,pna_resid).pvalue) + ')')
ax[0,0].set_ylabel('PNA',fontsize=26)
ax[0,0].set_xticks(xrange)
ax[0,0].set_xticklabels(name_xticks)
ax[0,0].legend(loc='upper right',frameon=False,fontsize=18)
ax[0,0].tick_params(axis='both',labelsize=20)
ax[0,0].grid(linestyle='--')
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# NAO
ax[0,1].plot(t,nao_resid,'r')
ax[0,1].plot(t,stats.linregress(t,nao_resid).intercept + stats.linregress(t,nao_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,nao_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,nao_resid).pvalue) + ')')
ax[0,1].set_ylabel('NAO',fontsize=26)
ax[0,1].set_xticks(xrange)
ax[0,1].set_xticklabels(name_xticks)
ax[0,1].legend(loc='upper right',frameon=False,fontsize=18)
ax[0,1].tick_params(axis='both',labelsize=20)
ax[0,1].grid(linestyle='--')
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# AO
ax[1,0].plot(t,ao_resid,'r')
ax[1,0].plot(t,stats.linregress(t,ao_resid).intercept + stats.linregress(t,ao_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,ao_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,ao_resid).pvalue) + ')')
ax[1,0].set_ylabel('AO',fontsize=26)
ax[1,0].set_xticks(xrange)
ax[1,0].set_xticklabels(name_xticks)
ax[1,0].legend(loc='lower left',frameon=False,fontsize=18)
ax[1,0].tick_params(axis='both',labelsize=20)
ax[1,0].grid(linestyle='--')
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# QBO
ax[1,1].plot(t,qbo_resid,'r')
ax[1,1].plot(t,stats.linregress(t,qbo_resid).intercept + stats.linregress(t,qbo_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,qbo_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,qbo_resid).pvalue) + ')')
ax[1,1].set_ylabel('QBO',fontsize=26)
ax[1,1].set_xticks(xrange)
ax[1,1].set_xticklabels(name_xticks)
ax[1,1].legend(loc='lower left',frameon=False,fontsize=18)
ax[1,1].tick_params(axis='both',labelsize=20)
ax[1,1].grid(linestyle='--')
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# AMO
ax[2,0].plot(t,amo_resid,'r')
ax[2,0].plot(t,stats.linregress(t,amo_resid).intercept + stats.linregress(t,amo_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,amo_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,amo_resid).pvalue) + ')')
ax[2,0].set_ylabel('AMO',fontsize=26)
ax[2,0].set_xticks(xrange)
ax[2,0].set_xticklabels(name_xticks)
ax[2,0].legend(loc='lower right',frameon=False,fontsize=18)
ax[2,0].tick_params(axis='both',labelsize=20)
ax[2,0].grid(linestyle='--')
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# PDO
ax[2,1].plot(t,pdo_resid,'r')
ax[2,1].plot(t,stats.linregress(t,pdo_resid).intercept + stats.linregress(t,pdo_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,pdo_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,pdo_resid).pvalue) + ')')
ax[2,1].set_ylabel('PDO',fontsize=26)
ax[2,1].set_xticks(xrange)
ax[2,1].set_xticklabels(name_xticks)
ax[2,1].legend(loc='lower left',frameon=False,fontsize=18)
ax[2,1].tick_params(axis='both',labelsize=20)
ax[2,1].grid(linestyle='--')
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# TNA
ax[3,0].plot(t,tna_resid,'r')
ax[3,0].plot(t,stats.linregress(t,tna_resid).intercept + stats.linregress(t,tna_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,tna_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,tna_resid).pvalue) + ')')
ax[3,0].set_ylabel('TNA',fontsize=26)
ax[3,0].set_xlabel('Year',fontsize=26)
ax[3,0].set_xticks(xrange)
ax[3,0].set_xticklabels(name_xticks)
ax[3,0].legend(loc='lower right',frameon=False,fontsize=18)
ax[3,0].tick_params(axis='both',labelsize=20)
ax[3,0].grid(linestyle='--')
ax[3,0].set_title('g',loc='left',fontsize=30,fontweight='bold')

# Niño3.4
ax[3,1].plot(t,nino34_resid,'r')
ax[3,1].plot(t,stats.linregress(t,nino34_resid).intercept + stats.linregress(t,nino34_resid).slope * t,'k--',linewidth=2,label='b = ' + '{:.2e}'.format(stats.linregress(t,nino34_resid).slope) + ' (pval = ' + '{:.2e}'.format(stats.linregress(t,nino34_resid).pvalue) + ')')
ax[3,1].set_ylabel('Niño3.4',fontsize=26)
ax[3,1].set_xlabel('Year',fontsize=26)
ax[3,1].set_xticks(xrange)
ax[3,1].set_xticklabels(name_xticks)
ax[3,1].legend(loc='lower right',frameon=False,fontsize=18)
ax[3,1].tick_params(axis='both',labelsize=20)
ax[3,1].grid(linestyle='--')
ax[3,1].set_title('h',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
if save_fig == True:
    fig.savefig('TimeSeries_detrended.jpg')
    
    
# Time series of detrended variables - first 24 months
fig,ax = plt.subplots(4,2,figsize=(24,28))
fig.subplots_adjust(left=0.08,bottom=0.05,right=0.95,top=0.95,hspace=0.2,wspace=0.2)

# PNA
ax[0,0].plot(t[0:24],pna_resid[0:24],'r')
ax[0,0].set_ylabel('PNA',fontsize=26)
ax[0,0].legend(loc='upper right',frameon=False,fontsize=18)
ax[0,0].tick_params(axis='both',labelsize=20)
ax[0,0].grid(linestyle='--')
ax[0,0].set_title('a',loc='left',fontsize=30,fontweight='bold')

# NAO
ax[0,1].plot(t[0:24],nao_resid[0:24],'r')
ax[0,1].set_ylabel('NAO',fontsize=26)
ax[0,1].legend(loc='upper right',frameon=False,fontsize=18)
ax[0,1].tick_params(axis='both',labelsize=20)
ax[0,1].grid(linestyle='--')
ax[0,1].set_title('b',loc='left',fontsize=30,fontweight='bold')

# AO
ax[1,0].plot(t[0:24],ao_resid[0:24],'r')
ax[1,0].set_ylabel('AO',fontsize=26)
ax[1,0].legend(loc='lower left',frameon=False,fontsize=18)
ax[1,0].tick_params(axis='both',labelsize=20)
ax[1,0].grid(linestyle='--')
ax[1,0].set_title('c',loc='left',fontsize=30,fontweight='bold')

# QBO
ax[1,1].plot(t[0:24],qbo_resid[0:24],'r')
ax[1,1].set_ylabel('QBO',fontsize=26)
ax[1,1].legend(loc='lower left',frameon=False,fontsize=18)
ax[1,1].tick_params(axis='both',labelsize=20)
ax[1,1].grid(linestyle='--')
ax[1,1].set_title('d',loc='left',fontsize=30,fontweight='bold')

# AMO
ax[2,0].plot(t[0:24],amo_resid[0:24],'r')
ax[2,0].set_ylabel('AMO',fontsize=26)
ax[2,0].legend(loc='lower right',frameon=False,fontsize=18)
ax[2,0].tick_params(axis='both',labelsize=20)
ax[2,0].grid(linestyle='--')
ax[2,0].set_title('e',loc='left',fontsize=30,fontweight='bold')

# PDO
ax[2,1].plot(t[0:24],pdo_resid[0:24],'r')
ax[2,1].set_ylabel('PDO',fontsize=26)
ax[2,1].legend(loc='lower left',frameon=False,fontsize=18)
ax[2,1].tick_params(axis='both',labelsize=20)
ax[2,1].grid(linestyle='--')
ax[2,1].set_title('f',loc='left',fontsize=30,fontweight='bold')

# TNA
ax[3,0].plot(t[0:24],tna_resid[0:24],'r')
ax[3,0].set_ylabel('TNA',fontsize=26)
ax[3,0].set_xlabel('Year',fontsize=26)
ax[3,0].legend(loc='lower right',frameon=False,fontsize=18)
ax[3,0].tick_params(axis='both',labelsize=20)
ax[3,0].grid(linestyle='--')
ax[3,0].set_title('g',loc='left',fontsize=30,fontweight='bold')

# Niño3.4
ax[3,1].plot(t[0:24],nino34_resid[0:24],'r')
ax[3,1].set_ylabel('Niño3.4',fontsize=26)
ax[3,1].set_xlabel('Year',fontsize=26)
ax[3,1].legend(loc='lower right',frameon=False,fontsize=18)
ax[3,1].tick_params(axis='both',labelsize=20)
ax[3,1].grid(linestyle='--')
ax[3,1].set_title('h',loc='left',fontsize=30,fontweight='bold')

# Save Fig.
fig.savefig('TimeSeries_detrended_first24m.jpg')