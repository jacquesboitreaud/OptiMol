# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 11:54:32 2020

@author: jacqu
"""

import pandas as pd 

df = pd.read_csv('violin_data_lgbm_tolerant.csv') # change here to load different QSAR scores 

dfz = df[df['subset']=='ZINC']
df_skal = df[df['subset']=='Skalic et al.']
df_opt = df[df['subset']=='OptiMol']
df_mult = df[df['subset']=='OptiMol-multiobj']

# Gianni enrichment : 

gianni = pd.concat((dfz,df_skal))
gianni = gianni.sort_values(by='qsar')

THRESHOLD = 0.5 # percentage for enrichment

# Proportion of Gianni samples in top N scoring
N=int(gianni.shape[0]*THRESHOLD/100)
gianni = gianni[-N:]
gianni.to_csv('gianni_top1pct.csv')
pct = gianni[gianni['subset']=='Skalic et al.'].shape[0]/N


# Same for OptiMol 

optimol = pd.concat((dfz,df_opt))
optimol = optimol.sort_values(by='qsar')


N=int(optimol.shape[0]*THRESHOLD/100)
optimol = optimol[-N:]
optimol.to_csv('optimol_top1pct.csv')
pct_opt = optimol[optimol['subset']=='OptiMol'].shape[0]/N

# same for multiobj

mult = pd.concat((dfz,df_mult))
mult = mult.sort_values(by='qsar')

N=int(mult.shape[0]*THRESHOLD/100)
mult.to_csv('multiobj_top1pct.csv')
mult = mult[-N:]
pct_mult = mult[mult['subset']=='OptiMol-multiobj'].shape[0]/N

print(f'Fractions of samples in top {THRESHOLD} % : ')
print('Gianni:', pct)
print('Optimol:', pct_opt)
print('Mult:', pct_mult)
