# -*- coding: utf-8 -*-
"""
Created on Tue May  7 23:58:52 2019

@author: Sachin
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('data.csv')
df = dataset.iloc[:,:-1].values 
X = pd.DataFrame(df)         
x = X.loc[:,X.columns != 1]
y = X.loc[:,1]

for i in range(len(y)) :
    if y[i] == 'M' :
        y[i] = 1 
        
for i  in range(len(y)) :
    if y[i] == 'B' :
        y[i] = 0 
        

y 

rows = 10 
col = 10 
t = rows*col 
nd = 31 

#network dimensions 
n_d = np.array([10,10])
n_it = 200 
n_lr = 0.5 
#weigth
w = np.random.random((3,21,1))
