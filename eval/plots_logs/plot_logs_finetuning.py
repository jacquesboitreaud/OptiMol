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


logs = np.load('../saved_models/logs_pairs.npy',allow_pickle=True)

print(logs.keys())

for k in logs.keys():
    print(k,':',len(logs[k]), 'values in list')
    
t = np.arange(len(logs['train_rec']))

# Plotting 

plt.figure()
plt.title('Reconstruction loss')
sns.lineplot(t,logs['train_rec'], label='train rec')
plt.legend()
plt.show()

plt.figure()
plt.title('Divergence to prior')
sns.lineplot(t,logs['train_simLoss'], label='contrastive Loss')
plt.legend()
plt.show()

plt.figure()
plt.title('Molecular properties prediction error')
sns.lineplot(t,logs['train_pmse'], label='train mse')
plt.legend()
plt.show()