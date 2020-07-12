# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:36:16 2019

@author: Sidarth
"""


from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd 
from scipy.spatial import distance
####################### FUNCTIONS ##########################################
############ BEST MATCHING UNIT#################

def best_match_unit(t, net, m):
    
    bmu_idx = np.array([0, 0])
    min_dist = np.iinfo(np.int).max
    
    # calculate the high-dimensional distance between each neuron and the input
    for x11 in range(net.shape[0]):
        for y11 in range(net.shape[1]):
            w = net[x11, y11, :].reshape(m, 1)
            sq_dist = distance.euclidean(w,t)
            if sq_dist < min_dist :
                min_dist = sq_dist
                bmu_idx = np.array([x11, y11])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(m, 1)
    # return the (bmu, bmu_idx) tuple
    return (bmu, bmu_idx)

##### RADIUS DECAY , LEARNING RATE DECAY, INFLUENCE DECAY#########

def radius_decay(i,rad, tc):
    i = i + 1 ;
    return rad * np.exp(-i / tc)

def learning_rate_decay(i, lr, nir):
    i = i + 1 ;
    return lr * np.exp(-i / nir)

def influence_cal(dt, rad):
    return np.exp(-dt/ (2* (rad**2)))

############################################################################
    
#################### DATASET #############################################
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


x = np.transpose(x)

###########################################################################

############################FEATURE SCALING################################
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
data = sc.fit_transform(x)         

x1 = np.transpose(data)
##########################################################################

##################### Parameter Initialisation############################
rows = 10 
col = 10 
numd = 31 
m = 31
n = 569
n_it = 1000
n_lr = 0.5 
nd = np.array([10,10])

#Radius 
init_radius = max(nd[0], nd[1]) / 2
print(init_radius)

# radius decay parameter
time_constant = n_it / np.log(init_radius)
print(time_constant)


#############################################################################

####################### WEIGHT MATRIX########################################
weight = np.random.random((nd[0],nd[1],numd))
weight.shape

print((weight.shape[0]))
print((weight.shape[1]))

#############################################################################

######################### CODE BODY##########################################
for i in range(n_it):
    # select a training example at random
    t = data[:, np.random.randint(0, n)].reshape(np.array([m, 1]))

    
    # find its Best Matching Unit
    bmu, bmu_idx = best_match_unit(t, weight, m)
    
    # decaying parameters
    r = radius_decay(i, init_radius, time_constant)
    l = learning_rate_decay(i, n_lr, n_it)
    
    # update weight vector to move closer to input
    # and move the neighbour 2-D vector space closer
    
    for xx2 in range(weight.shape[0]):
        for yy2 in range(weight.shape[1]):
            w = weight[xx2, yy2, :].reshape(m, 1)
            w_dist = np.sum((np.array([xx2, yy2]) - bmu_idx) ** 2)
            w_dist = np.sqrt(w_dist)
           
            if w_dist <= r:
                # calculate the degree of influence (based on the 2-D distance)
                influence = influence_cal(w_dist, r)
                
                # new w = old w + (learning rate * influence * delta)
                # where delta = input vector (t) - old w
                new_w = w + (l * influence * (t - w))
                weight[xx2, yy2, :] = new_w.reshape(1,m)

                
print(weight.shape)
##########################################################################


##################################VISUALISATION###########################
mapping = np.empty(shape=(rows,col))
from pylab import bone,pcolor, colorbar, plot,show 
bone()
pcolor(mapping)
markers = ['o','s']
colors = ['r','g']

for j, m  in enumerate(data):
    plot(weight[0]+0.5 , 
         weight[1]+0.5 ,
         markers[y[j]],
         markeredgecolor = 'None', 
         markerfacecolor = 'None',
         markersize = 10, 
         markeredgewidth = 2)


#############################################################################
  

 
