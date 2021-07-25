#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
from math import sqrt
import tensorflow as tf


class Knn:
    def __init__(self):
        self.x=None
        self.y=None
        
    def Find_neighbors(self,train_x,data_point,neighbors):
        vector_distances=[]
        for i in train_x:
            
            dist=self.find_distance(i,data_point)
            vector_distances.append((i,dist))
        vector_distances.sort(key=lambda tup: tup[1])
        neighbor_list=[]
        for label in range(neighbors):
            neighbor_list.append(vector_distances[label][0])
        return neighbor_list
    def find_label(self,neighbor_list):
        output=[data_point[-1] for data_point in neighbor_list]
        return max(set(output), key=output.count)
        
    def find_distance(self,data_point1,data_point2):
        distance=0.0
        
        for j in range(len(data_point1)-1):
            distance += math.pow(data_point1[j]-data_point2[j],2)
            
        return np.sqrt(distance)
            
    def predict(self,test_x,test_y,neighbors):
        pred=[]
        with tf.device('/device:gpu:0'):
            for idx,i in enumerate(test_x):
                print(idx)
                neighbor_list=self.Find_neighbors(self.x,i,neighbors)
                pred.append(self.find_label(neighbor_list))

            acc=np.mean(np.array(test_x)[:,-1]==pred)
            print("model Acc: ",acc)
    def fit(self,x,y,neighbors):
        self.x=x
        self.y=y
        print("Knearest(neighbors=%d)",neighbors)
        self.predict(x,y,neighbors)



model=Knn()

df=pd.read_csv("labeleddata.csv")
df=df.drop(columns=['Unnamed: 0','src_ip','dst_ip'])
df.loc[df['flags']=='DF','flags']=0
df=df.astype('int')
df=np.array(df)

train_y=df[35000:35500,-1]
test_y=df[39900:40000,-1]

train_X=df[35000:35500,:-1]
test_X=df[39900:40000,:-1]


model.fit(df[35000:35500,:],df[39900:40000],neighbors=3)


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(train_X, train_y)
print("prebuilt-library Acc: ",neigh.score(test_X,test_y))

