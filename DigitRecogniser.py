# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:42:03 2015

@author: Laurie.Richardson
"""

import pandas as pd
from pandas import DataFrame
import numpy as np
import sklearn
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import matplotlib.pyplot as plt


#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    
target = [x[0] for x in dataset]
train = [x[1:] for x in dataset]
test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]

#create and train the random forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train, target)

#Predict based on test data and create as a DataFrame
PTestData=rf.predict(test)    
PTestDataDF=DataFrame(PTestData)
PTestDataDF.index +=1

PTestDataDF=PTestDataDF.rename(columns={0:'Label'})
PTestDataDF.index.names=['ImageId']


#print casual.feature_importances_
#print "Accuracy score: %0.2f" % rf.score(train, target)

PTestDataDF.to_csv('DigitSubmission.csv')

#savetxt('DigitRecogniserSubmission.csv',predicted_probs, header='ImageID,Prediction', delimiter=',', fmt='%f')










#def main():
#    #create the training & test sets, skipping the header row with [1:]
#    dataset = genfromtxt(open('train.csv','r'), delimiter=',', dtype='f8')[1:]    
#    target = [x[0] for x in dataset]
#    train = [x[1:] for x in dataset]
#    test = genfromtxt(open('test.csv','r'), delimiter=',', dtype='f8')[1:]
#    
#    #create and train the random forest
#    rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
#    rf.fit(train, target)
#
#    #Predict based on test data and create an index
#    z=rf.predict(test)    
#    y=DataFrame(z)    
#    #predicted_probs = [[index+1, x[1]] for index, x in enumerate(rf.predict(test))]
#    
#
#    #savetxt('DigitRecogniserSubmission.csv',predicted_probs, header='ImageID,Prediction', delimiter=',', fmt='%f')
#
#if __name__=="__main__":
#    main()
