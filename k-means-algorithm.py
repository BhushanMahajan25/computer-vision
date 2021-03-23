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
      x = np.float32((observed[i]+k[i])/2)
      z.append(x)
    return z

  def euclidean_distance(self,observed,k): 
    sum = np.float64(0)
    for i in range(len(k)):
      diff = np.float64(round(np.float64(observed[i])-np.float64(k[i]),3))
      sq = np.float64(round(math.pow(diff,2),3))
      sum += sq
    return math.sqrt(sum)


def __main__():
  dataset = [[185,72],[170,56],[168,60],[179,68],[182,72],[188,77],[180,71],[180,70],[183,84],[180,88],[180,67],[177,76]]
  dataset2 = np.array(dataset)
  #print('dataset2:',dataset2)

  pancakes=r'CVND_Exercises-master\1_3_Types_of_Features_Image_Segmentation\images\pancakes.jpg'
  rainbow=r'CVND_Exercises-master\1_3_Types_of_Features_Image_Segmentation\images\rainbow_icon.png'
  #print('image',image)
  image = mpimg.imread(rainbow)

  image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
  image_reshape = image.reshape(-1,3)
  np.float32(image_reshape)
  print('len(image_reshape)',len(image_reshape))
  plt.imshow(image)
  plt.show()

  k = 3   #no. of clusters

  obj2 = KMeansGeneral()
  clusters = obj2.start(k,image_reshape)
  temp = []
  for i in range(len(clusters)):
    li = clusters[i]
    for j in range(len(li)):
        temp.append(li[j])

  #clusters = np.uint8(temp)
  clusters = np.array(temp)

  final = clusters.reshape((image.shape))
  print('showing final image')
  plt.imshow(final)
  plt.show()
  #clusters = obj2.start(k,dataset2)
  #for i in range(k):
  #  print('cluster:'+str(i)+' :: ',clusters[i])

__main__()