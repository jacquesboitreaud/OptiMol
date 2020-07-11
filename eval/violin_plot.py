# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 09:48:47 2020

@author: jacqu
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os,sys 
script_dir = os.path.dirname(os.path.realpath(__file__))


optimol = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'optimol_samples_scored.csv'))
optimol['subset']='optimol'
try:
    multiobj = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'multiobj_samples_scored.csv'))
    multiobj['subset']='multiobj'
except:
    pass
gianni =  pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'fabritiis_docked.csv'))
gianni['subset']='Skalic et al.'
zinc = pd.read_csv(os.path.join(script_dir,'scores_for_plot', 'zinc_docked.csv'))
zinc['subset']='ZINC'

df= pd.concat([zinc, gianni, optimol, multiobj])

sns.violinplot(x="subset", y="score", data=df, color="0.8")
sns.stripplot(x="subset", y="score", data=df, jitter=True, zorder=1)
plt.ylim(-14, -4)
plt.show()