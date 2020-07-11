# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:02:16 2020

@author: jacqu
"""

import os, sys
import pandas as pd 
import seaborn as sns
script_dir = os.path.dirname(os.path.realpath(__file__))

df = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/big_new_lr/5k_samples_scored.csv'))

first, last = df[:2500], df[2500:]

sns.distplot(first.score)
sns.distplot(last.score)

df = pd.read_csv(os.path.join(script_dir,'..', 'cbas/slurm/results/multiobj_big/5k_samples_scored.csv'))

first, last = df[:2500], df[2500:]

plt.figure()
sns.distplot(first.score)
sns.distplot(last.score)