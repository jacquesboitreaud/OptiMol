# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 19:17:00 2020

@author: jacqu

Samples around z and plots samples in pca space 
"""

drd_a = Z[1]

idx = 700

active= drd_a[idx].reshape(1,-1)
tensor_a = torch.tensor(active)

plt.xlim(-5,6)
plt.ylim(-2.8,2.8)
p = ['lightblue','lightgreen','orange']
for i,z in enumerate(Z) :
    prof = profiles[i]
    pca_plot_color(z= Z[i], pca = fitted_pca, color = p[i], label = f'{prof}') 

pca_plot_color( active, pca = fitted_pca, color = 'red', label = 'active')

samples = model.sample_around_z(tensor_a, dist=2, attempts = 10 )
samples=samples.cpu().numpy()

pca_plot_color( samples, pca = fitted_pca, color = 'purple', label = 'SAMPLES')

tr = fitted_pca.transform(drd_a)


easy, hard = [], []
for i in range(drd_a.shape[0]):
    x=tr[i,0]
    y=tr[i,1]
    if(x>0.8 and y<0):
        easy.append(i)
    if(x<0.5):
        hard.append(i)
        
df_a = df[df['fold']==2]
df_a= df_a[df_a['profile']==1]

easy = df_a.iloc[easy]

hard = df_a.iloc[hard]

easy.to_csv('easy_seeds.csv')
hard.to_csv('hard_seeds.csv')


    
