# -*- coding: utf-8 -*-
"""
Created on Wed May 06 11:03:21 2015

"""

#Unsupervised Learning
#The data has no labels
#We are interested in finding similarities between the objects. 
#Unsupervised learning is discovering labels from the data itself. 
#Unsupervised learning comprises tasks such as 
#dimensionality reduction, clustering, and density estimation. 
#We can used unsupervised methods to determine combinations of 
#the measurements which best display the structure of the data. 
#A projection of the data can be used to visualize the four-dimensional 
#dataset in two dimensions.

#Principal Component Analysis (PCA) 
#A dimension reduction technique that can find the 
#combinations of variables that explain the most variance

import matplotlib.pyplot as plt
from pylab import plot,show
import numpy as np
from numpy import vstack,array
from numpy.random import rand
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pylab as pl

iris = load_iris()
X, y = iris.data, iris.target

pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)
print "Reduced dataset shape:", X_reduced.shape

pl.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)

print "Meaning of the 2 components:"
for component in pca.components_:
    print " + ".join("%.3f x %s" % (value, name)
                     for value, name in zip(component,
                                            iris.feature_names))





