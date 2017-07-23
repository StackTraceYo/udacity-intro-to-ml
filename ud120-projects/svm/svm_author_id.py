#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

from sklearn.svm import SVC

cl = SVC(kernel="rbf", C=10000)
t0 = time()
cl.fit(features_train, labels_train)
print "training time:", round(time() - t0, 3), "s"

t0 = time()
pred = cl.predict(features_test)
print "predict time:", round(time() - t0, 3), "s"

one = 0
zero = 0
for x in pred:
    if x == 1:
        one += 1
    elif x == 0:
        zero += 1

print "Number of Zeroes: ", zero
print "Number of Ones: ", one

from sklearn.metrics import accuracy_score

print "accuracy:", accuracy_score(pred, labels_test)


#########################################################


