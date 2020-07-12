# -*- coding: utf-8 -*-
"""
Created on Sun May  5 17:36:08 2019

@author: Sidarth
"""
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 


#importing the data set : wisconsin dataset 
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

#Feature Scaling 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
x1 = sc.fit_transform(x) 
 
#TRAINING THE SELF ORGANISING MAP 

from minisom import MiniSom
som = MiniSom (x =10, y = 10, input_len = 31, sigma = 2.0, learning_rate = 0.5) 
som.random_weights_init(x1)
som.train_random(data = x1, num_iteration = 100 )

#Visualisation 

from pylab import bone,pcolor, colorbar, plot,show 
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for j, m  in enumerate(x1):
    w = som.winner(m)
    plot(w[0]+0.5 , 
         w[1]+0.5 ,
         markers[y[j]],
         markeredgecolor = colors[y[j]], 
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2)
    
show()
    



