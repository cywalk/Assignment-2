#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pylab 
import seaborn as sns


# In[2]:


# importing file as .csv

hist_i = 'C://users/cyril/documents/uzh/FS20/Risk Management/Assignments/Assignment 2/Valueatrisk.csv'
df = pd.read_csv(hist_i)


# In[3]:


# Preparing the data

df = df.drop(['Unnamed: 10', 'Date', 'US 1Y', 'US 2Y', 'US 3Y', 'CH 1Y', 'CH 3Y', 'CH 5Y'], axis=1)
df = df.dropna()


# In[4]:


df1 = df.diff(periods=9)
df1 = df1.dropna()


# In[5]:


# Standard deviation of variables used

std_US = df1['US 5Y'].std()
std_CH = df1['CH 2Y'].std()
std_XR = df1['CHF/USD'].std()


# In[6]:


# Correlation matrix

corr = df1.corr()

# Cholesky decomposition 

C = np.linalg.cholesky(corr)


# In[7]:


# sampling random normal data with specific std

MC_US = np.random.normal(0, std_US, (1, 10000))
MC_US = [val for sublist in MC_US for val in sublist]
MC_CH = np.random.normal(0, std_CH, (1, 10000))
MC_CH = [val for sublist in MC_CH for val in sublist]
MC_XR = np.random.normal(0, std_XR, (1, 10000))
MC_XR = [val for sublist in MC_XR for val in sublist]

MC = np.array([MC_US, MC_CH, MC_XR])


# In[8]:


# Multiplying the Cholesky with sampled data

x_cor = np.dot(C, MC)
X = pd.DataFrame(x_cor.T)


# In[9]:


# Applying simulated values to last known value

X['adj_US'] = X[0]+ df['US 5Y'].iloc[-1]
X['adj_CH'] = X[1] + df['CH 2Y'].iloc[-1]
X['adj_XR'] = X[2] + df['CHF/USD'].iloc[-1]


# In[10]:


X['delta_US'] = (X['adj_US']-df['US 5Y'].iloc[-1])/100
X['delta_CH'] = (X['adj_CH']-df['CH 2Y'].iloc[-1])/100
X['delta_XR'] = (X['adj_XR']-df['CHF/USD'].iloc[-1])/df['CHF/USD'].iloc[-1]


# In[14]:


# Calculating modified Bond Duration

D_CH = ((1*2000/1.02 + 2*102000/1.02**2)/100000)/1.02

D_US = []
L = np.arange(0.5,5,0.5)
for i in L:
    d = (i*13500/1.045**(i*2))/300000
    D_US.append(d)
    if i == 4.5:
        d = (5*313500/1.045**(5*2))/300000
        D_US.append(d)
        D_US = sum(D_US)/1.045


# In[16]:


X['dValue_US'] = -D_US*X['delta_US']*300000*1.6
X['dValue_CH'] = -D_CH*X['delta_CH']*100000
X['dCurrency'] = X['delta_XR']*(1.6*(100000+300000))
X['dtotal'] = X['dValue_US']+X['dValue_CH']+X['dCurrency']


# In[17]:


# Finally, computing the VaR 99%

x0 = X['dtotal'].quantile(.01)


# In[21]:


plt.figure(figsize=(18,9))
ax = sns.distplot(X['dtotal'], color = 'c')
kde_x, kde_y = ax.lines[0].get_data()

p1 = plt.axvline(x=x0,color='b', label = 'VaR 99% = -38123.96 CHF')
ax.fill_between(kde_x, kde_y, where=(kde_x<x0) ,interpolate=True, color='b')
plt.xlabel('Portfolio value change', fontsize = 20)
ax.tick_params(labelsize=16)
plt.legend(loc = 'upper right', frameon=False, fontsize = 16)
plt.savefig('PF_dist.png')


# In[17]:


# Computing Jarque Bera tests

jb_US = stats.jarque_bera(df['US 5Y'])
jb_CH = stats.jarque_bera(df['CH 2Y'])
jb_XR = stats.jarque_bera(df['CHF/USD'])

print([jb_US, jb_CH, jb_XR])


# In[18]:


# Testing for normality

f = plt.figure(figsize=(15,4))
ax1 = f.add_subplot(1,3,1)
ax2 = f.add_subplot(1,3,2)
ax3 = f.add_subplot(1,3,3)

plt.subplot(1,3,1)
ax1 = stats.probplot(df['US 5Y'], dist= 'norm', plot=pylab)

plt.subplot(1,3,2)
ax2 = stats.probplot(df['CH 2Y'], dist= 'norm', plot=pylab)

plt.subplot(1,3,3)
ax3 = stats.probplot(df['CHF/USD'], dist= 'norm', plot=pylab)


#plt.savefig('QQ_plots.png')

