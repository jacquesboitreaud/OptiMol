# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 20:00:30 2020

@author: jacqu
"""

import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

p_range = np.arange(0,1,0.01)
q_range = np.arange(0,1,0.01)

mat = np.zeros((100,100))
mat_log = np.zeros((100,100))

for i in range(p_range.shape[0]):
    for j in range(p_range.shape[0]):
        
        mat[i,j] = p_range[i]/ q_range[j]
        
        mat_log[i,j] = np.log(p_range[i])/ np.log(q_range[j])
    
plt.title('Quotient des probas')    
sns.heatmap(mat)
plt.figure()
plt.title('Quotient des logs')
sns.heatmap(mat_log)