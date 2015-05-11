# -*- coding: utf-8 -*-
"""
Created on Mon May 04 16:37:39 2015

"""
import matplotlib.pyplot as plt
from pylab import plot,show
import numpy as np
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq
from scipy.cluster.vq import whiten
from sklearn.datasets import load_iris
from sklearn.svm import SVC

#For a classification problem using SVM

#Load the iris data
iris = load_iris()

X, y = iris.data, iris.target

SVC=SVC()

SVC.fit(X,y)

result=SVC.predict([1,2,3,4])

Output=iris.target_names[result]