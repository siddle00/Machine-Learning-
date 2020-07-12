# som_iris.py
# SOM for Iris dataset
# Anaconda3 5.2.0 (Python 3.6.5)

# ==================================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance


def closest_node(data, t, map, m_rows, m_cols):
  # (row,col) of map node closest to data[t]
  result = (0,0) 
  for i in range(m_rows):
    for j in range(m_cols):
      ed = distance.euclidean(map[i][j], data[t])
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result

def manhattan_dist(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)

def most_common(lst, n):
  # lst is a list of values 0 . . n
  if len(lst) == 0: return -1
  counts = np.zeros(shape=n, dtype=np.int)
  for i in range(len(lst)):
    counts[lst[i]] += 1
  return np.argmax(counts)


  # 0. get started
  np.random.seed(1)
  Dim = 31
  Rows = 100; Cols = 100
  RangeMax = Rows + Cols
  LearnMax = 0.5
  StepsMax = 100

  # 1. load data
  dataset = pd.read_csv('data.csv')
  df = dataset.iloc[:,:-1].values 
  X = pd.DataFrame(df)         
  data_x = X.loc[:,X.columns != 1]
  data_y = X.loc[:,1]

  for i in range(len(data_y)) :
      if data_y[i] == 'M' :
          data_y[i] = 1 
        
  for i  in range(len(data_y)) :
      if data_y[i] == 'B' :
          data_y[i] = 0 
        
  data_y 


  # option: normalize data  

  # 2. construct the SOM
  print("Constructing a 30x30 SOM from the iris data")
  map = np.random.random_sample(size=(Rows,Cols,Dim))
  for s in range(StepsMax):
    if s % (StepsMax/10) == 0: print("step = ", str(s))
    pct_left = 1.0 - ((s * 1.0) / StepsMax)
    curr_range = (int)(pct_left * RangeMax)
    curr_rate = pct_left * LearnMax

    t = np.random.randint(len(data_x))
    (bmu_row, bmu_col) = closest_node(data_x, t, map, Rows, Cols)
    for i in range(Rows):
      for j in range(Cols):
        if manhattan_dist(bmu_row, bmu_col, i, j) < curr_range:
          map[i][j] = map[i][j] + curr_rate * (data_x[t] - map[i][j])
  print("SOM construction complete \n")

  # 3. construct U-Matrix
  print("Constructing U-Matrix from SOM")
  u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
  for i in range(Rows):
    for j in range(Cols):
      v = map[i][j]  # a vector 
      sum_dists = 0.0; ct = 0
     
      if i-1 >= 0:    # above
        sum_dists += distance.euclidean(v, map[i-1][j]); ct += 1
      if i+1 <= Rows-1:   # below
        sum_dists += distance.euclidean(v, map[i+1][j]); ct += 1
      if j-1 >= 0:   # left
        sum_dists += distance.euclidean(v, map[i][j-1]); ct += 1
      if j+1 <= Cols-1:   # right
        sum_dists += distance.euclidean(v, map[i][j+1]); ct += 1
      
      u_matrix[i][j] = sum_dists / ct
  print("U-Matrix constructed \n")

  # display U-Matrix
  plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
  plt.show()

  # 4. because the data has labels, another possible visualization:
  # associate each data label with a map node
  print("Associating each data label to one map node ")
  mapping = np.empty(shape=(Rows,Cols), dtype=object)
  for i in range(Rows):
    for j in range(Cols):
      mapping[i][j] = []

  for t in range(len(data_x)):
    (m_row, m_col) = closest_node(data_x, t, map, Rows, Cols)
    mapping[m_row][m_col].append(data_y[t])

  label_map = np.zeros(shape=(Rows,Cols), dtype=np.int)
  for i in range(Rows):
    for j in range(Cols):
      label_map[i][j] = most_common(mapping[i][j], 3)
 
  plt.imshow(label_map, cmap=plt.cm.get_cmap('terrain_r', 4)) 
  plt.colorbar()
  plt.show()
  

