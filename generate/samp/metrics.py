# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:04:06 2020

@author: jacqu
"""

results_file = 'hard_herg_out/Final_results.txt'

with open(results_file,'r') as f:
    lines = f.readlines()
  
cpt, safe=0,0
for l in lines : 
    cpt+=1
    if('non' in l):
        safe +=1
        
print(safe/cpt)


    