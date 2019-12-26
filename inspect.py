# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 16:50:44 2019

@author: jacqu

Inspect data and deal with non labeled affinities 
"""

import pandas as pd 

df = pd.read_csv('../data/CHEMBL_18t.csv', nrows=100)

#TODO : change nan to zeros, either here or in the dataloading. 