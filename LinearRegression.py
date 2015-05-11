# -*- coding: utf-8 -*-
"""
Created on Mon May 04 17:06:26 2015

"""
import matplotlib.pyplot as plt
from pylab import plot,show
import numpy as np
from numpy import vstack,array
from numpy.random import rand
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate random data
np.random.seed(0)
X = np.random.random(size=(20, 1))
y = 3 * X.squeeze() + 2 + np.random.normal(size=20)

#Fit the Linear Regression
model=LinearRegression(fit_intercept=True)
model.fit(X,y)

print ("Model coefficient: %.5f, and intercept: %.5f" % (model.coef_, model.intercept_))

# Plot the data and the model prediction
X_test = np.linspace(0, 1, 100)[:, np.newaxis]

y_test = model.predict(X_test)

plt.plot(X.squeeze(), y, 'o')
plt.plot(X_test.squeeze(), y_test)
