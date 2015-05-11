# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:46:04 2015

"""
import matplotlib.pyplot as plt
from pylab import plot,show
import numpy as np
from numpy import vstack,array
from numpy.random import rand
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pylab as pl
from sklearn.decomposition import PCA

iris = load_iris()
X, y = iris.data, iris.target

k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X)

y_pred = k_means.predict(X)

pca = PCA(n_components=2)
pca.fit(X)
X_reduced = pca.transform(X)

pl.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_pred)