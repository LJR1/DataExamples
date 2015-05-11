# -*- coding: utf-8 -*-
"""
Created on Sat May 02 09:59:49 2015

"""


#pyplot interface is generally preferred for non-interactive plotting
#(i.e., scripting). The pylab interface is convenient for interactive 
#calculations and plotting, as it minimizes typing. 
import matplotlib.pyplot as plt
from pylab import plot,show
import numpy as np
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from scipy.cluster.vq import whiten
from sklearn.datasets import load_iris
from sklearn import neighbors

#Load the iris data
iris = load_iris()

#Describe the Iris Data
n_samples, n_features = iris.data.shape
print (n_samples, n_features)
print iris.data[0]
print iris.data.shape
print iris.target.shape
print iris.target
print iris.target_names
print iris.feature_names

#Plot the data to visualise 
#Set the x and y index values for plotting matched against iris.feature_names
#Index starts at [0], 2 and 3 reference petal length and petal width
x_index=2
y_index=3
# Label the colorbar with the correct target names
formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
#c sets the colour of the points against the number of targets-3
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)
plt.colorbar(ticks=[0, 1, 2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])

######################################

#Supervised Learning-Classification & Regression
#One or more unknown quantities to be determined from other observed 
#quantities off the object
#Classification-Discrete Regression-Continuous
#KNN Classification
#Data already loaded above in iris
#K nearest neighbors (kNN): given a new, unknown observation, look up in your 
#reference database which ones have the closest features and assign the 
#predominant class.

X, y = iris.data, iris.target

# create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=1)

#fit the model
knn.fit(X,y)

#Predict the value
result = knn.predict([ 5.1,  3.5,  1.4,  5.0])

Output = iris.target_names[result]



