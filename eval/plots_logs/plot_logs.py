# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:21:55 2020

@author: jacqu


Plots the training curves for the pretraining step of the VAE 

"""

import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 


logs = np.load('../saved_models/baseline_logs.npy',allow_pickle=True)

print(logs.keys())

for k in logs.keys():
    print(k,':',len(logs[k]), 'values in list')
    
t = np.arange(len(logs[k]))

# Plotting 

plt.figure()
plt.title('Reconstruction loss')
sns.lineplot(t,logs['train_rec'], label='train rec')
sns.lineplot(t,logs['test_rec'], label='test_rec')
plt.legend()
plt.show()

plt.figure()
plt.title('Divergence to prior')
sns.lineplot(t,logs['train_kl'], label='train kl')
sns.lineplot(t,logs['test_kl'], label='test kl')
plt.legend()
plt.show()

plt.figure()
plt.title('Molecular properties prediction error')
sns.lineplot(t,logs['train_pmse'], label='train mse')
sns.lineplot(t,logs['test_pmse'], label='test mse')
plt.ylim(0,100) # first train loss is huge , shrinks the plot 
plt.legend()
plt.show()