# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:32:07 2020

@author: jacqu

Sampling around actives in latent space 
"""

import pandas as pd 
import numpy as np 


if __name__ == "__main__":
    
    # ================= Select actives ================================
    df = pd.read_csv('../data/exp/offtarget/herg_drd.csv')
    
    df=df[df['profile']==1]
    f1, f2 = df[df['fold']==1],  df[df['fold']==2]
    
    # Sampling 50 from fold 1 and 50 from fold 2 
    
    f1 = f1.sample(50)
    f2 = f2.sample(50)
    
    d= pd.concat((f1,f2))
    
    d.to_csv('drd3_seeds.csv')