# -*- coding: utf-8 -*-
"""
Created on Sat May 02 10:15:09 2015
"""

from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from scipy.cluster.vq import whiten

# data generation
data = vstack((rand(150,2)+ array([.5,.5]),rand(150,2)))

# normalization on a per feature basis. Rescale each feature dimension by
#dividing by its standard deviation across all observations to get unit variance
data = whiten(data)

# computing K-Means with K = 2 (2 clusters)
# Returns a centroid for the 2x2 array 
centroids,_ = kmeans(data, k_or_guess=2)

#Assign the data to a relevant centroid 
idx,_=vq(data,centroids)

#Indexing based on the shape of the array (centroid shape)
plot(data[idx==0,0],data[idx==0,1],'ob', data[idx==1,0],data[idx==1,1],'or')

#Plotting centroids only (2) as columns
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
