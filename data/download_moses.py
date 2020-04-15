# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:13:51 2020

@author: jacqu

Download moses datasets and saves them 
"""

import moses
import pandas as pd

print('>>> Loading data from moses')
train = moses.get_dataset('train')
test = moses.get_dataset('test')
test_scaffolds = moses.get_dataset('test_scaffolds')

train = pd.DataFrame(train)
test = pd.DataFrame(test)
scaf = pd.DataFrame(test_scaffolds)

print('>>> Saving data to csv files in ./data')
train.to_csv('data/moses_train.csv')
test.to_csv('data/moses_test.csv')
scaf.to_csv('data/moses_test_scaffolds.csv')