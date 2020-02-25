# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:07:32 2020

@author: jacqu


Cleans text file for formatting errors (smiles which contained '\n' and got split into two when writing)

Run it before using the .txt file for docking or anyother downstream task !
"""


with open('samp/hard_samp_batch.txt','r') as f:
        lines = f.readlines()
        
print(len(lines))
with open('samp/hard_samp_batch.txt', "w") as f:
        for l in lines : 
            try:
                m= l.split()[1]
                name = l.split()[0]
                f.write(name)
                f.write('\t')
                f.write(m)
                f.write('\n')
            except: 
                next
        
