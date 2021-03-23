#This is the K-means algorithm for simple dataset 

import math
import cv2
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt

class KMeansGeneral:
  def start(self,n,dataset):
    k_list = [] #list of centroids
    clusters = [] #list of clusters
  
    #randomly chosen the centroids
    for i in range(n):
      k_list.append(dataset[0])
      clusters.append([dataset[0]])
      #dataset.remove(dataset[0]) #normal list
      dataset = np.delete(dataset,0,0)
    for i in range(len(dataset)):
      ed_list = []  #euclidean distance list
      observed = dataset[i]
      #calculate euclidean distance
      for j in range(len(k_list)):
        ed = self.euclidean_distance(observed,k_list[j])
        ed_list.append(ed)
      
      #get index of min ed
      min_idx = ed_list.index(min(ed_list))
      clusters[min_idx].append(observed)
      k_list[min_idx] = self.new_centroid(observed,k_list[min_idx])
    return clusters

  def new_centroid(self,observed,k):
    z = []
    for i in range(len(k)):
      x = (observed[i]+k[i])/2
      z.append(x)
    return z

  def euclidean_distance(self,observed,k): 
    sum = 0
    for i in range(len(k)):
      sum += math.pow(observed[i]-k[i],2)
    return math.sqrt(sum)


def __main__():
  dataset = [[185,72],[170,56],[168,60],[179,68],[182,72],[188,77],[180,71],[180,70],[183,84],[180,88],[180,67],[177,76]]
  dataset2 = np.array(dataset)
  k = 2   #no. of clusters

  obj2 = KMeansGeneral()
  clusters = obj2.start(k,dataset2)
  for i in range(k):
    print('cluster:'+str(i)+' :: ',clusters[i])

__main__()