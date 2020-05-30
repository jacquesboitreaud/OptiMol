# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:41:29 2020

@author: jacqu
"""

import pandas as pd 

f= '../data/moses_train.csv'

for chunk in pd.read_csv(f, chunksize=5000):
       # we are going to append to each table by group
       # we are not going to create indexes at this time
       # but we *ARE* going to create (some) data_columns
       
       print(chunk)

       """
       # figure out the field groupings
       for g, v in group_map.items():
             # create the frame for this group
             frame = chunk.reindex(columns = v['fields'], copy = False)    

             # append it
             store.append(g, frame, index=False, data_columns = v['dc'])
        """