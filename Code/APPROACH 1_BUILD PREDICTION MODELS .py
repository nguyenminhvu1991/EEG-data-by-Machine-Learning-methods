'''
EEG DATA PROJECT
PURPOSE: DIFFERENTIATE H AND P SUBJECTS

APPROACH 1: BUILD PREDICTION MODELS TO PREDICT H OR P SUBJECT
Use groups for data from 12 different subjects to prevent data leakage and Cross Validation in SVM and ANN models to access the test accuracy

RESULT: FAIL
'''

import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import cross_validate, GroupShuffleSplit
from sklearn.svm import SVC


initial_path = "D:/SP19/project eeg/Python/"

#initialize first empty maxtrix
X= np.empty((0,50000), dtype = float) 
#import  6  Healthy (H) subjects (31x50000) and 6 PTSD (P) subjects (31x50000)
for path in glob.glob(initial_path + "H/*.txt"):  
    table= pd.read_table(path)
    X1 =np.array(table)
    X= np.concatenate((X,X1), axis=0)
for path in glob.glob(initial_path + "P/*.txt"): #import the P chanels
    table= pd.read_table(path)
    X1 =np.array(table)
    X= np.concatenate((X,X1), axis=0)

#Careate the labels
y = np.repeat(['H','P'],X.shape[0]/2) 

#Add groups column for 12 groups of data to prevent data leakage
groups = np.repeat(range(0, 12),31)

#Can split by groups column to prevent data leakage by another way
#from sklearn.model_selection import GroupKFold
#gkf = GroupKFold(n_splits=12)"""

# Cross Validation by SVM
acc = cross_validate(SVC(max_iter=-1, 
                         kernel ='linear' , 
                         class_weight='balanced', 
                         gamma ='scale'), 
                     X, y, groups=groups, 
                     cv=GroupShuffleSplit(test_size=0.2, random_state=50))

print("Mean test accuracy: {}".format(np.mean(acc["test_score"])))
#0.4086021505376344
'''UNACCEPTABLE PREDICTION MODEL'''

# CV Cross Validation by NNet
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(8,2), random_state=50)
acc2 = cross_validate(clf, 
                     X, y, groups=groups, 
                     cv=GroupShuffleSplit(test_size=0.2, random_state=50))

print("Mean test accuracy: {}".format(np.mean(acc2["test_score"])))
#0.5333333333333333
'''UNACCEPTABLE PREDICTION MODEL'''
 


# CV Cross Validation by NNet with scaled data

#SCALE BY NORMINALIZATION 
#https://stackoverflow.com/questions/13324071/scaling-data-in-scikit-learn-svm
#https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
from sklearn import preprocessing
X_scaled = preprocessing.scale(X)
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,  hidden_layer_sizes=(8,2), random_state=50)
acc3 = cross_validate(clf, 
                     X_scaled, y, groups=groups, 
                     cv=GroupShuffleSplit(test_size=0.2, random_state=50))

print("Mean test accuracy: {}".format(np.mean(acc3["test_score"])))
#0.5440860215053763
'''UNACCEPTABLE PREDICTION MODEL'''
